#!/usr/bin/env python3
"""
Four-pool semantic LLM judge experiment (Phase-2 mini fulltest).

Hypothesis
----------
The selection disadvantage of heterogeneous tree search is partly due to
frequency aggregation over correlated candidates. Typed structural verification
recovers much of the loss; semantic verification may further convert higher
Oracle coverage into superior end-to-end Selected.

Pipeline (identical for CoT / Five / r75 / BoN)
-----------------------------------------------
  Candidate pool
    -> typed structural hard filter (hard_violation_count == 0)
    -> exact-plan dedup (judge never sees frequency / method / MCTS metadata)
    -> if |unique|==0: fallback = raw exact-JSON majority on full pool
       if |unique|==1: select that plan (no LLM call)
       if |unique|>=2: blind semantic LLM judge over shuffled labels C0..Ck
    -> Selected

Judge is blind to: method name, frequency, node_value, path, leaf action.
Judge sees: task_desc, compact agent I/O cards, candidate plans.

Outputs under artifacts/phase2_fulltest/semantic_judge/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from utils.answer_selector import (  # noqa: E402
    hard_violation_count,
    load_tools_manifest,
    parse_plan,
)


def make_client(api: str, model_name: str):
    """api: openai | openrouter"""
    if api == "openai":
        from interfaces.OpenAI_client import OpenAIClient

        return OpenAIClient(model_name=model_name)
    if api == "openrouter":
        from interfaces.OpenRouter_client import OpenRouterClient

        return OpenRouterClient(model_name=model_name)
    raise ValueError(f"Unknown api={api}")


class JudgeQuotaError(RuntimeError):
    pass

MANIFEST = ROOT / "artifacts" / "phase2_fulltest" / "manifest_mini.json"
TOOLS = ROOT / "src" / "interfaces" / "tools" / "tools_manifest.json"
TEST_JSONL = ROOT / "data" / "test.jsonl"
OUT = ROOT / "artifacts" / "phase2_fulltest" / "semantic_judge"
SEED_P2 = 20260808
COT_K = 8
POOLS = ("CoT", "Five", "r75", "BoN")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def resolve(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def load_phase2_runs() -> Dict[str, Path]:
    m = json.loads(MANIFEST.read_text())
    out = {}
    for r in m.get("runs", []):
        if int(r.get("seed", -1)) != SEED_P2:
            continue
        out[r["key"]] = resolve(r["dir"])
    need = ["CoT-only-M", "Five-M", "Five-H24", "BoN-CoT-M"]
    missing = [k for k in need if k not in out]
    if missing:
        raise SystemExit(f"Missing phase2 runs: {missing}")
    return out


def load_tasks(run_dir: Path) -> Dict[int, dict]:
    out = {}
    for p in (run_dir / "answer_sheets").glob("Task * - Answer.json"):
        a = json.loads(p.read_text())
        out[int(a["task_id"])] = a
    return out


def load_task_descs() -> Dict[int, str]:
    out = {}
    for line in TEST_JSONL.open():
        o = json.loads(line)
        out[int(o["task_id"])] = o.get("task_desc") or ""
    return out


def cands_sorted(task: dict) -> List[dict]:
    c = list(task.get("all_candidates") or [])
    c.sort(key=lambda x: int(x.get("candidate_id", 0)))
    return c


def plan_key(c: dict) -> Optional[str]:
    ans = c.get("model_answer")
    if ans is None:
        return None
    if isinstance(ans, str):
        return ans
    try:
        return json.dumps(ans, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(ans)


def oracle(cands: List[dict]) -> bool:
    return any(bool(c.get("correct")) for c in cands)


def majority_pick(cands: List[dict]) -> Optional[dict]:
    if not cands:
        return None
    groups: Dict[str, List[dict]] = defaultdict(list)
    for c in cands:
        k = plan_key(c)
        if k is None:
            continue
        groups[k].append(c)
    if not groups:
        return None
    best_k = max(
        groups.keys(),
        key=lambda k: (len(groups[k]), -min(int(c.get("candidate_id", 0)) for c in groups[k])),
    )
    return sorted(groups[best_k], key=lambda c: int(c.get("candidate_id", 0)))[0]


def native_selected_ok(task: dict) -> bool:
    return bool(((task.get("selected_best") or {}).get("metrics") or {}).get("correct"))


# ---------------------------------------------------------------------------
# Structural violation taxonomy
# ---------------------------------------------------------------------------

def classify_violations(plan: Optional[List[dict]], ctx) -> List[str]:
    """Return list of violation tags (may be empty)."""
    if plan is None:
        return ["unparseable_json"]
    if not isinstance(plan, list) or not plan:
        return ["empty_or_nonlist_plan"]

    tags: List[str] = []
    agents = set(ctx.keys()) if isinstance(ctx, dict) else None
    n = len(plan)
    # cycle / forward-dep approximated via dependence indices
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            tags.append("non_object_step")
            continue
        agent = step.get("agent")
        if not isinstance(agent, str) or not agent:
            tags.append("missing_agent")
        elif agents is not None and agent not in agents:
            tags.append("invalid_agent")

        deps = step.get("dependence", [-1])
        if deps is None:
            deps = [-1]
        if not isinstance(deps, list) or not deps:
            tags.append("invalid_dependency")
            deps = []
        dep_ids = []
        for d in deps:
            try:
                di = int(d)
            except (TypeError, ValueError):
                tags.append("invalid_dependency")
                continue
            dep_ids.append(di)
            if di != -1 and (di < 0 or di >= i):
                tags.append("forward_or_oob_dependency")

        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        agent_spec = (ctx or {}).get(agent) if isinstance(agent, str) else None
        expected_in = set((agent_spec.get("input") or {}).keys()) if agent_spec else None
        if expected_in is not None:
            missing = expected_in - set(inputs.keys())
            if missing:
                tags.append("missing_required_input")

        # GENERATED ref checks
        non_neg = [d for d in dep_ids if d != -1]
        gen_refs = []
        gen_re = re.compile(r"^<GENERATED>-(\d+)(?:-<([A-Za-z0-9_\-]+)>)?$")
        for val in inputs.values():
            if not isinstance(val, str) or "GENERATED" not in val:
                continue
            m = gen_re.match(val)
            if not m:
                tags.append("malformed_generated_ref")
                continue
            src = int(m.group(1))
            out_key = m.group(2)
            gen_refs.append(src)
            if src < 0 or src >= i:
                tags.append("bad_generated_src")
                continue
            if out_key is not None and ctx is not None:
                src_agent = plan[src].get("agent") if isinstance(plan[src], dict) else None
                src_outs = set((((ctx.get(src_agent) or {}).get("output")) or {}).keys())
                src_step_outs = plan[src].get("outputs") if isinstance(plan[src], dict) else None
                if isinstance(src_step_outs, list):
                    src_outs |= set(str(x) for x in src_step_outs)
                if out_key not in src_outs:
                    tags.append("type_mismatch_generated_out")
            if non_neg and src not in non_neg:
                tags.append("generated_not_in_dependence")
        if non_neg and not gen_refs:
            tags.append("deps_without_generated_inputs")
        if (not non_neg) and gen_refs:
            tags.append("generated_without_deps")

    # soft cycle hint: repeated agent+identical dependence chains not enforced here
    return sorted(set(tags))


def compact_tools_card(ctx: dict, max_desc: int = 120) -> str:
    lines = []
    for name, spec in ctx.items():
        desc = (spec.get("desc") or "").replace("\n", " ")
        if len(desc) > max_desc:
            desc = desc[: max_desc - 3] + "..."
        inns = list((spec.get("input") or {}).keys())
        outs = list((spec.get("output") or {}).keys())
        lines.append(f"- {name}\n  desc: {desc}\n  inputs: {inns}\n  outputs: {outs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

def build_pool(name: str, tid: int, runs_tasks: Dict[str, Dict[int, dict]]) -> List[dict]:
    cot, five, h24, bon = (
        runs_tasks["CoT"],
        runs_tasks["Five"],
        runs_tasks["H24"],
        runs_tasks["BoN"],
    )
    if name == "CoT":
        return cands_sorted(cot[tid])
    if name == "Five":
        return cands_sorted(five[tid])
    if name == "BoN":
        return cands_sorted(bon[tid])
    if name == "r75":
        pool = []
        for c in cands_sorted(cot[tid])[:COT_K]:
            cc = dict(c)
            cc["_src"] = "cot8"
            pool.append(cc)
        for c in cands_sorted(h24[tid]):
            cc = dict(c)
            cc["_src"] = "h24"
            pool.append(cc)
        return pool
    raise ValueError(name)


def structural_filter(cands: List[dict], ctx) -> Tuple[List[dict], Dict[str, Any]]:
    kept = []
    rejected = 0
    tag_counter: Counter = Counter()
    for c in cands:
        plan = parse_plan(c.get("model_answer"))
        tags = classify_violations(plan, ctx)
        # align with hard_violation_count == 0
        hv = hard_violation_count(plan, ctx)
        if hv == 0 and plan is not None:
            kept.append(c)
        else:
            rejected += 1
            if not tags:
                tags = ["hard_violation_other"]
            for t in tags:
                tag_counter[t] += 1
    # exact-plan dedup, keep first (stable by candidate_id order of kept)
    uniq = []
    seen = set()
    for c in kept:
        k = plan_key(c)
        if k is None or k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    stats = {
        "n_raw": len(cands),
        "n_hard0": len(kept),
        "n_unique_hard0": len(uniq),
        "reject_rate": rejected / max(len(cands), 1),
        "reject_tags": dict(tag_counter),
    }
    return uniq, stats


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert at multi-agent geospatial workflow planning.
Given a natural-language disaster task and a catalog of typed agents (tools),
choose which candidate structured plan is most likely to be the correct workflow.

Rules:
- Prefer plans that use the right agents, wire inputs/outputs correctly via
  <GENERATED>-i-<output_key> references, and match the task requirements.
- Ignore how candidates are labeled; labels are arbitrary.
- Respond with ONLY a JSON object:
  {"chosen_id":"C<k>","confidence":0.0-1.0,"reason":"<one short sentence>"}
"""


def build_judge_prompt(task_desc: str, tools_card: str, labeled: List[Tuple[str, dict]]) -> str:
    blocks = []
    for lab, c in labeled:
        plan = parse_plan(c.get("model_answer"))
        plan_txt = json.dumps(plan, ensure_ascii=False, indent=2)
        blocks.append(f"### Candidate {lab}\n```json\n{plan_txt}\n```")
    return (
        f"{JUDGE_SYSTEM}\n\n"
        f"## Task\n{task_desc}\n\n"
        f"## Available agents (typed I/O)\n{tools_card}\n\n"
        f"## Candidates\n" + "\n\n".join(blocks) + "\n\n"
        f"Return JSON only."
    )


def parse_judge_response(text: str, valid_labels: List[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    # try fenced json
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        # bare Ck
        m2 = re.search(r"\b(C\d+)\b", text)
        return m2.group(1) if m2 and m2.group(1) in valid_labels else None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        m2 = re.search(r"\b(C\d+)\b", text)
        return m2.group(1) if m2 and m2.group(1) in valid_labels else None
    cid = obj.get("chosen_id") or obj.get("candidate_id") or obj.get("id")
    if isinstance(cid, int):
        cid = f"C{cid}"
    if isinstance(cid, str):
        cid = cid.strip()
        if cid in valid_labels:
            return cid
        m2 = re.search(r"C\d+", cid)
        if m2 and m2.group(0) in valid_labels:
            return m2.group(0)
    return None


def judge_unique(
    client: Any,
    task_desc: str,
    tools_card: str,
    uniq: List[dict],
    rng: random.Random,
    temperature: float = 0.0,
) -> Tuple[Optional[dict], Dict[str, Any]]:
    order = list(range(len(uniq)))
    rng.shuffle(order)
    labeled = [(f"C{i}", uniq[j]) for i, j in enumerate(order)]
    label_to_cand = {lab: c for lab, c in labeled}
    prompt = build_judge_prompt(task_desc, tools_card, labeled)
    t0 = time.time()
    try:
        raw = client.generate(prompt, temperature=temperature, max_tokens=300)
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "credit_balance_exhausted" in msg or "402" in msg:
            raise JudgeQuotaError(msg) from e
        raise
    elapsed = time.time() - t0
    if not raw:
        # OpenAIClient may swallow quota errors and return ""
        raise JudgeQuotaError(
            "Judge returned empty response (possible API quota/network failure). "
            "Check billing / OPENROUTER_API_KEY and rerun with --resume."
        )
    chosen_lab = parse_judge_response(raw or "", list(label_to_cand.keys()))
    meta = {
        "raw": raw,
        "chosen_label": chosen_lab,
        "n_unique": len(uniq),
        "elapsed_s": elapsed,
        "label_order": [lab for lab, _ in labeled],
    }
    if chosen_lab is None:
        return None, meta
    return label_to_cand[chosen_lab], meta


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def eval_task_pool(
    pool_name: str,
    tid: int,
    cands: List[dict],
    task_desc: str,
    ctx,
    tools_card: str,
    client: Optional[OpenAIClient],
    rng: random.Random,
    cache: Dict[str, Any],
    use_judge: bool,
) -> Dict[str, Any]:
    o = oracle(cands)
    maj = majority_pick(cands)
    uniq, st = structural_filter(cands, ctx)
    # filter then majority among hard0 (pre-dedup kept list reconstructed)
    hard0 = []
    for c in cands:
        plan = parse_plan(c.get("model_answer"))
        if plan is not None and hard_violation_count(plan, ctx) == 0:
            hard0.append(c)
    filt_maj = majority_pick(hard0) if hard0 else maj

    mode = "fallback_majority"
    pick = maj
    judge_meta = None

    if len(uniq) == 0:
        mode = "empty_filter_fallback_majority"
        pick = maj
    elif len(uniq) == 1:
        mode = "singleton_hard0"
        pick = uniq[0]
    elif use_judge and client is not None:
        cache_key = f"{pool_name}:{tid}"
        if cache_key in cache and cache[cache_key].get("pick_key") is not None:
            # resume from cache
            pk = cache[cache_key]["pick_key"]
            pick = next((c for c in uniq if plan_key(c) == pk), None)
            mode = cache[cache_key].get("mode", "judge")
            judge_meta = cache[cache_key].get("judge_meta")
            if pick is None:
                pick = filt_maj
                mode = "judge_miss_fallback_filter_maj"
        else:
            pick, judge_meta = judge_unique(client, task_desc, tools_card, uniq, rng)
            mode = "judge"
            if pick is None:
                pick = filt_maj
                mode = "judge_parse_fail_fallback_filter_maj"
            cache[cache_key] = {
                "pick_key": plan_key(pick) if pick else None,
                "mode": mode,
                "judge_meta": {
                    "chosen_label": (judge_meta or {}).get("chosen_label"),
                    "n_unique": (judge_meta or {}).get("n_unique"),
                    "elapsed_s": (judge_meta or {}).get("elapsed_s"),
                    "raw": (judge_meta or {}).get("raw"),
                    "label_order": (judge_meta or {}).get("label_order"),
                },
            }
    else:
        mode = "filter_majority_no_judge"
        pick = filt_maj

    def ok(c):
        return bool(c and c.get("correct"))

    return {
        "pool": pool_name,
        "task_id": tid,
        "oracle": o,
        "majority_ok": ok(maj),
        "filter_majority_ok": ok(filt_maj),
        "judge_ok": ok(pick) if use_judge else None,
        "final_ok": ok(pick),
        "mode": mode,
        "n_raw": st["n_raw"],
        "n_hard0": st["n_hard0"],
        "n_unique_hard0": st["n_unique_hard0"],
        "reject_rate": st["reject_rate"],
        "reject_tags": st["reject_tags"],
        "judge_meta": judge_meta,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(rows: List[Dict[str, Any]], native_by_pool: Dict[str, Dict[int, bool]]) -> List[Dict[str, Any]]:
    out = []
    for pool in POOLS:
        rs = [r for r in rows if r["pool"] == pool]
        if not rs:
            continue
        n = len(rs)
        oracle_n = sum(1 for r in rs if r["oracle"])
        maj_n = sum(1 for r in rs if r["majority_ok"])
        fmaj_n = sum(1 for r in rs if r["filter_majority_ok"])
        final_n = sum(1 for r in rs if r["final_ok"])
        native = native_by_pool.get(pool, {})
        native_n = sum(1 for r in rs if native.get(r["task_id"]))

        # recovery among oracle & majority wrong
        def recovery(sel_key: str) -> Optional[float]:
            denom = [r for r in rs if r["oracle"] and not r["majority_ok"]]
            if not denom:
                return None
            num = sum(1 for r in denom if r.get(sel_key))
            return 100.0 * num / len(denom)

        def util(sel_key: str) -> Optional[float]:
            if oracle_n == 0:
                return None
            return 100.0 * sum(1 for r in rs if r["oracle"] and r.get(sel_key)) / oracle_n

        judge_calls = sum(1 for r in rs if str(r.get("mode", "")).startswith("judge"))
        out.append(
            {
                "pool": pool,
                "n_tasks": n,
                "oracle_em": 100.0 * oracle_n / n,
                "native_selected_em": 100.0 * native_n / n if pool != "r75" else None,
                "majority_em": 100.0 * maj_n / n,
                "filter_majority_em": 100.0 * fmaj_n / n,
                "judge_em": 100.0 * final_n / n,
                "majority_util": util("majority_ok"),
                "filter_majority_util": util("filter_majority_ok"),
                "judge_util": util("final_ok"),
                "recovery_filter_vs_maj": recovery("filter_majority_ok"),
                "recovery_judge_vs_maj": recovery("final_ok"),
                "mean_reject_rate": sum(r["reject_rate"] for r in rs) / n,
                "mean_n_raw": sum(r["n_raw"] for r in rs) / n,
                "mean_n_hard0": sum(r["n_hard0"] for r in rs) / n,
                "mean_n_unique_hard0": sum(r["n_unique_hard0"] for r in rs) / n,
                "judge_calls": judge_calls,
                "modes": dict(Counter(r["mode"] for r in rs)),
            }
        )
    return out


def aggregate_reject_tags(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for pool in POOLS:
        rs = [r for r in rows if r["pool"] == pool]
        tag_sum: Counter = Counter()
        total_rejected = 0
        total_raw = 0
        for r in rs:
            total_raw += r["n_raw"]
            rejected = r["n_raw"] - r["n_hard0"]
            total_rejected += rejected
            for t, c in (r.get("reject_tags") or {}).items():
                tag_sum[t] += c
        for tag, c in tag_sum.most_common():
            out.append(
                {
                    "pool": pool,
                    "tag": tag,
                    "count": c,
                    "share_of_rejected": c / max(total_rejected, 1),
                    "share_of_raw": c / max(total_raw, 1),
                    "reject_rate_pool": total_rejected / max(total_raw, 1),
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
            w.writerow(flat)


def render_report(summary: List[Dict[str, Any]], tag_rows: List[Dict[str, Any]], usage: dict, args) -> str:
    lines = []
    lines.append("# Semantic Judge Experiment — Phase-2 mini")
    lines.append("")
    lines.append(f"- Judge model: `{args.judge_model}`")
    lines.append(f"- Seed (RNG shuffle): `{args.rng_seed}`")
    lines.append(f"- Pipeline: Structural hard filter → unique plans → Semantic judge (blind)")
    lines.append(f"- API usage: {usage}")
    lines.append("")
    lines.append("## End-to-end Selected")
    lines.append("")
    lines.append(
        "| Pool | Oracle | Native | Majority | Filter→Maj | **Filter→Judge** | "
        "Maj util | Filt util | Judge util | RejectRate | uniq hard0 |"
    )
    lines.append(
        "|------|-------:|-------:|---------:|-----------:|-----------------:|"
        "---------:|----------:|-----------:|-----------:|-----------:|"
    )
    for s in summary:
        native = f"{s['native_selected_em']:.2f}" if s["native_selected_em"] is not None else "—"
        lines.append(
            f"| {s['pool']} | {s['oracle_em']:.2f} | {native} | {s['majority_em']:.2f} | "
            f"{s['filter_majority_em']:.2f} | **{s['judge_em']:.2f}** | "
            f"{s['majority_util']:.1f} | {s['filter_majority_util']:.1f} | {s['judge_util']:.1f} | "
            f"{100*s['mean_reject_rate']:.1f}% | {s['mean_n_unique_hard0']:.2f} |"
        )
    lines.append("")
    lines.append("## Recovery among (Oracle=1 ∧ Majority=0)")
    lines.append("")
    lines.append("| Pool | Filter recovery | Judge recovery |")
    lines.append("|------|----------------:|---------------:|")
    for s in summary:
        fr = f"{s['recovery_filter_vs_maj']:.1f}%" if s["recovery_filter_vs_maj"] is not None else "—"
        jr = f"{s['recovery_judge_vs_maj']:.1f}%" if s["recovery_judge_vs_maj"] is not None else "—"
        lines.append(f"| {s['pool']} | {fr} | {jr} |")
    lines.append("")
    lines.append("## Structural rejection tags (top)")
    lines.append("")
    by_pool = defaultdict(list)
    for r in tag_rows:
        by_pool[r["pool"]].append(r)
    for pool in POOLS:
        rows = by_pool.get(pool, [])[:8]
        if not rows:
            continue
        lines.append(f"### {pool} (pool reject rate {100*rows[0]['reject_rate_pool']:.1f}%)")
        lines.append("")
        lines.append("| Tag | count | share of rejected |")
        lines.append("|-----|------:|------------------:|")
        for r in rows:
            lines.append(f"| {r['tag']} | {r['count']} | {100*r['share_of_rejected']:.1f}% |")
        lines.append("")
    lines.append("## Interpretation checklist")
    lines.append("")
    lines.append("- If `r75 Filter→Judge > BoN Filter→Judge`: W2 strong (coverage convertible).")
    lines.append("- If ≈ : higher coverage needs stronger verification to compete.")
    lines.append("- If < : Oracle advantage not usable even under semantic verification.")
    lines.append("- Tree pools should show higher structural reject rates than BoN if mechanism holds.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", choices=["openai", "openrouter"], default="openai")
    ap.add_argument(
        "--judge_model",
        default="gpt-4o",
        help="Judge model id (openai: gpt-4o; openrouter: openai/gpt-4o)",
    )
    ap.add_argument("--max_tasks", type=int, default=0, help="0 = all")
    ap.add_argument("--task_ids", type=str, default="", help="Comma-separated task ids")
    ap.add_argument("--pools", type=str, default="CoT,Five,r75,BoN")
    ap.add_argument("--rng_seed", type=int, default=20260914)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0, help="Alias: first N tasks")
    ap.add_argument("--no_judge", action="store_true", help="Only structural filter analysis")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--fresh", action="store_true", help="Ignore judge cache")
    args = ap.parse_args()
    if args.api == "openrouter" and "/" not in args.judge_model and args.judge_model.startswith("gpt-"):
        args.judge_model = f"openai/{args.judge_model}"

    OUT.mkdir(parents=True, exist_ok=True)
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    for p in pools:
        if p not in POOLS:
            raise SystemExit(f"Unknown pool {p}")

    runs = load_phase2_runs()
    runs_tasks = {
        "CoT": load_tasks(runs["CoT-only-M"]),
        "Five": load_tasks(runs["Five-M"]),
        "H24": load_tasks(runs["Five-H24"]),
        "BoN": load_tasks(runs["BoN-CoT-M"]),
    }
    descs = load_task_descs()
    ctx = load_tools_manifest(TOOLS)
    tools_card = compact_tools_card(ctx)

    ids = sorted(
        set(runs_tasks["CoT"])
        & set(runs_tasks["Five"])
        & set(runs_tasks["H24"])
        & set(runs_tasks["BoN"])
    )
    if args.task_ids:
        want = {int(x) for x in args.task_ids.split(",") if x.strip()}
        ids = [t for t in ids if t in want]
    if args.smoke:
        ids = ids[: args.smoke]
    elif args.max_tasks:
        ids = ids[: args.max_tasks]

    cache_path = OUT / f"judge_cache_{args.judge_model.replace('/', '_')}.json"
    cache: Dict[str, Any] = {}
    if args.resume and not args.fresh and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Loaded cache entries: {len(cache)} from {cache_path}")

    client = None
    if not args.no_judge:
        try:
            client = make_client(args.api, args.judge_model)
        except Exception as e:
            raise SystemExit(
                f"Failed to init judge client (api={args.api}, model={args.judge_model}): {e}\n"
                "Top up OpenAI credits or set OPENROUTER_API_KEY, then rerun.\n"
                "Meanwhile: python scripts/run_semantic_judge.py --no_judge"
            ) from e

    native_by_pool = {
        "CoT": {tid: native_selected_ok(runs_tasks["CoT"][tid]) for tid in ids},
        "Five": {tid: native_selected_ok(runs_tasks["Five"][tid]) for tid in ids},
        "BoN": {tid: native_selected_ok(runs_tasks["BoN"][tid]) for tid in ids},
        "r75": {},
    }

    # Build work items
    work = []
    for tid in ids:
        for pool in pools:
            work.append((pool, tid))

    rows: List[Dict[str, Any]] = []
    lock_cache_writes = 0

    def run_one(item):
        pool, tid = item
        pool_code = {"CoT": 1, "Five": 2, "r75": 3, "BoN": 4}.get(pool, 0)
        rng = random.Random(args.rng_seed + tid * 17 + pool_code * 97)
        cands = build_pool(pool, tid, runs_tasks)
        return eval_task_pool(
            pool_name=pool,
            tid=tid,
            cands=cands,
            task_desc=descs.get(tid, ""),
            ctx=ctx,
            tools_card=tools_card,
            client=client,
            rng=rng,
            cache=cache,
            use_judge=not args.no_judge,
        )

    print(f"Tasks={len(ids)} pools={pools} work={len(work)} judge={not args.no_judge} model={args.judge_model}")

    try:
        if args.threads <= 1 or args.no_judge:
            for i, item in enumerate(work, 1):
                row = run_one(item)
                rows.append(row)
                if i % 20 == 0 or i == len(work):
                    print(
                        f"  [{i}/{len(work)}] {row['pool']} task {row['task_id']} "
                        f"mode={row['mode']} final={row['final_ok']}"
                    )
                    if client and not args.fresh:
                        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                futs = {ex.submit(run_one, item): item for item in work}
                for fut in as_completed(futs):
                    row = fut.result()
                    rows.append(row)
                    done += 1
                    if done % 20 == 0 or done == len(work):
                        print(
                            f"  [{done}/{len(work)}] {row['pool']} task {row['task_id']} "
                            f"mode={row['mode']} final={row['final_ok']}"
                        )
                        if client:
                            cache_path.write_text(json.dumps(cache, ensure_ascii=False))
                            lock_cache_writes += 1
    except JudgeQuotaError as e:
        if client:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        raise SystemExit(
            f"Judge API quota/credit failure after {len(cache)} cached decisions.\n{e}\n"
            f"Cache saved to {cache_path}. Top up credits / set OPENROUTER_API_KEY, then:\n"
            f"  python scripts/run_semantic_judge.py --api openrouter --resume\n"
            f"Or structural-only: python scripts/run_semantic_judge.py --no_judge"
        ) from e

    if client:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))

    rows.sort(key=lambda r: (r["pool"], r["task_id"]))
    summary = summarize(rows, native_by_pool)
    tag_rows = aggregate_reject_tags(rows)
    usage = client.get_usage() if client else {}

    write_csv(OUT / "per_task.csv", rows)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "reject_tags.csv", tag_rows)
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "usage": usage, "args": vars(args)}, indent=2))
    report = render_report(summary, tag_rows, usage, args)
    (OUT / "SEMANTIC_JUDGE_REPORT.md").write_text(report)
    print(report)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
