"""
Gold-free answer selectors for MCTS candidates.

Used online (mcts_search) and offline (scripts/offline_reselect.py).
Never uses the `correct` flag for selection — only structural fields.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

Candidate = dict
SelectFn = Callable[[List[Candidate], Any], Optional[Candidate]]

GENERATED_RE = re.compile(r"^<GENERATED>-(\d+)(?:-<([A-Za-z0-9_\-]+)>)?$")
GENERATED_LOOSE_RE = re.compile(r"^<GENERATED>-(\d+)-([A-Za-z0-9_\-]+)$")

ONLINE_SELECTORS = (
    "majority",
    "filter_then_majority",
    "filter_then_short",
    "filter_then_nodup",
)


def load_tools_manifest(path: str | Path) -> Optional[dict]:
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def parse_plan(model_answer: Any) -> Optional[List[dict]]:
    if model_answer is None:
        return None
    if isinstance(model_answer, list):
        return model_answer if all(isinstance(x, dict) for x in model_answer) else None
    if not isinstance(model_answer, str):
        return None
    text = model_answer.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def tool_sequence(plan: Optional[List[dict]]) -> Optional[Tuple[str, ...]]:
    if not plan:
        return None
    agents = []
    for step in plan:
        if not isinstance(step, dict) or "agent" not in step:
            return None
        agents.append(str(step["agent"]))
    return tuple(agents)


def path_actions(cand: dict) -> Tuple[str, ...]:
    path = cand.get("model_path") or []
    actions = []
    for item in path:
        if isinstance(item, (list, tuple)) and item:
            actions.append(str(item[0]))
        elif isinstance(item, str):
            actions.append(item)
    return tuple(actions)


def _ctx_agents(ctx) -> Optional[set]:
    if ctx is None:
        return None
    if isinstance(ctx, set):
        return ctx
    if isinstance(ctx, dict):
        return set(ctx.keys())
    return None


def _ctx_manifest(ctx) -> Optional[dict]:
    if isinstance(ctx, dict) and ctx:
        first = next(iter(ctx.values()))
        if isinstance(first, dict) and ("input" in first or "output" in first or "desc" in first):
            return ctx
    return None


def verifier_score(plan: Optional[List[dict]], ctx=None) -> float:
    if not plan:
        return -100.0

    manifest = _ctx_manifest(ctx)
    agents = _ctx_agents(ctx)
    n = len(plan)
    hard_penalty = 0.0
    step_quality = []

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            hard_penalty += 5.0
            step_quality.append(0.0)
            continue

        q = 0.0
        agent = step.get("agent")
        if not isinstance(agent, str) or not agent:
            hard_penalty += 3.0
            step_quality.append(0.0)
            continue

        if agents is not None and agent not in agents:
            hard_penalty += 4.0
            step_quality.append(0.0)
            continue
        q += 0.2

        if step.get("step") is not None:
            try:
                if int(step["step"]) != i:
                    hard_penalty += 1.0
                else:
                    q += 0.05
            except (TypeError, ValueError):
                hard_penalty += 0.5

        deps = step.get("dependence", [-1])
        if deps is None:
            deps = [-1]
        dep_ids = []
        if not isinstance(deps, list) or not deps:
            hard_penalty += 2.0
        else:
            for d in deps:
                try:
                    di = int(d)
                except (TypeError, ValueError):
                    hard_penalty += 2.0
                    continue
                dep_ids.append(di)
                if di != -1 and (di < 0 or di >= i):
                    hard_penalty += 2.0
            q += 0.1

        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        outputs = step.get("outputs")
        agent_spec = (manifest or {}).get(agent) if manifest else None
        expected_in = set((agent_spec.get("input") or {}).keys()) if agent_spec else None
        expected_out = set((agent_spec.get("output") or {}).keys()) if agent_spec else None

        if expected_in is not None:
            got_in = set(inputs.keys())
            if got_in == expected_in:
                q += 0.35
            else:
                hard_penalty += 0.8 * len(expected_in - got_in)
                hard_penalty += 0.4 * len(got_in - expected_in)

        if expected_out is not None and outputs is not None:
            if isinstance(outputs, list):
                got_out = set(str(x) for x in outputs)
            elif isinstance(outputs, dict):
                got_out = set(outputs.keys())
            else:
                got_out = set()
            if got_out == expected_out:
                q += 0.15
            else:
                hard_penalty += 0.3 * len(expected_out - got_out)
                hard_penalty += 0.2 * len(got_out - expected_out)

        non_neg_deps = [d for d in dep_ids if d != -1]
        generated_refs = []
        for val in inputs.values():
            if not isinstance(val, str) or "GENERATED" not in val:
                continue
            m = GENERATED_RE.match(val)
            if m:
                src = int(m.group(1))
                out_key = m.group(2)
                generated_refs.append(src)
                if src < 0 or src >= i or src >= n:
                    hard_penalty += 2.0
                    continue
                if out_key is not None and manifest is not None:
                    src_agent = plan[src].get("agent") if src < len(plan) else None
                    src_spec = (manifest or {}).get(src_agent) if src_agent else None
                    src_outs = set((src_spec.get("output") or {}).keys()) if src_spec else set()
                    src_step_outs = plan[src].get("outputs")
                    if isinstance(src_step_outs, list):
                        src_outs |= set(str(x) for x in src_step_outs)
                    if out_key in src_outs:
                        q += 0.2
                    else:
                        hard_penalty += 1.5
                if non_neg_deps and src not in non_neg_deps:
                    hard_penalty += 1.0
            elif GENERATED_LOOSE_RE.match(val):
                hard_penalty += 1.0
                mm = GENERATED_LOOSE_RE.match(val)
                generated_refs.append(int(mm.group(1)))
            else:
                hard_penalty += 1.5

        if non_neg_deps:
            if not generated_refs:
                hard_penalty += 1.5
            for src in non_neg_deps:
                if src not in generated_refs:
                    hard_penalty += 0.5
            q += 0.1 if generated_refs else 0.0
        else:
            if generated_refs:
                hard_penalty += 1.0
            else:
                q += 0.1

        step_quality.append(min(q, 1.0))

    avg_q = sum(step_quality) / max(len(step_quality), 1)
    length_penalty = 0.35 * max(0, n - 1)
    return 10.0 * avg_q - hard_penalty - length_penalty


def hard_violation_count(plan: Optional[List[dict]], ctx=None) -> int:
    if not plan:
        return 99
    manifest = _ctx_manifest(ctx)
    agents = _ctx_agents(ctx)
    n = len(plan)
    viol = 0
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            viol += 1
            continue
        agent = step.get("agent")
        if not isinstance(agent, str) or (agents is not None and agent not in agents):
            viol += 1
            continue
        deps = step.get("dependence", [-1])
        if deps is None:
            deps = [-1]
        if not isinstance(deps, list) or not deps:
            viol += 1
            deps = []
        dep_ids = []
        for d in deps:
            try:
                di = int(d)
            except (TypeError, ValueError):
                viol += 1
                continue
            dep_ids.append(di)
            if di != -1 and (di < 0 or di >= i):
                viol += 1
        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        agent_spec = (manifest or {}).get(agent) if manifest else None
        expected_in = set((agent_spec.get("input") or {}).keys()) if agent_spec else None
        if expected_in is not None:
            got_in = set(inputs.keys())
            viol += len(expected_in - got_in)
        non_neg = [d for d in dep_ids if d != -1]
        gen_refs = []
        for val in inputs.values():
            if not isinstance(val, str) or "GENERATED" not in val:
                continue
            m = GENERATED_RE.match(val)
            if not m:
                viol += 1
                continue
            src = int(m.group(1))
            out_key = m.group(2)
            gen_refs.append(src)
            if src < 0 or src >= i:
                viol += 1
                continue
            if out_key is not None and manifest is not None:
                src_agent = plan[src].get("agent")
                src_outs = set((((manifest.get(src_agent) or {}).get("output")) or {}).keys())
                src_step_outs = plan[src].get("outputs")
                if isinstance(src_step_outs, list):
                    src_outs |= set(str(x) for x in src_step_outs)
                if out_key not in src_outs:
                    viol += 1
            if non_neg and src not in non_neg:
                viol += 1
        if non_neg and not gen_refs:
            viol += 1
        if (not non_neg) and gen_refs:
            viol += 1
    return viol


def prefers_direct_root(cand: dict) -> bool:
    acts = path_actions(cand)
    if len(acts) <= 1:
        return False
    return len(acts) <= 3 and "DIRECT_ANSWER" in acts[1:2]


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def pick_by_key(cands: List[Candidate], key_fn, reverse=True) -> Optional[Candidate]:
    if not cands:
        return None
    return sorted(cands, key=key_fn, reverse=reverse)[0]


def _cand_verifier(c: Candidate, ctx=None) -> float:
    return verifier_score(parse_plan(c.get("model_answer")), ctx)


def _answer_key(c: Candidate) -> Optional[str]:
    plan = parse_plan(c.get("model_answer"))
    if plan is None:
        return None
    return json.dumps(plan, sort_keys=True)


def _filter_hard(cands: List[Candidate], ctx=None, max_viol: int = 0) -> List[Candidate]:
    kept = []
    for c in cands:
        plan = parse_plan(c.get("model_answer"))
        if plan is None:
            continue
        if hard_violation_count(plan, ctx) <= max_viol:
            kept.append(c)
    return kept


def select_current_answer_majority(cands: List[Candidate], ctx=None) -> Optional[Candidate]:
    buckets = defaultdict(list)
    for c in cands:
        ans = c.get("model_answer")
        if not ans:
            continue
        plan = parse_plan(ans)
        key = json.dumps(plan, sort_keys=True) if plan is not None else str(ans)
        buckets[key].append(c)
    if not buckets:
        return None
    best_key = max(buckets.keys(), key=lambda k: (len(buckets[k]), -_safe_int(buckets[k][0].get("depth"), 99)))
    group = buckets[best_key]
    return pick_by_key(group, lambda c: (_safe_float(c.get("N")), _safe_float(c.get("Q"))), reverse=True)


def select_filter_then_majority(cands: List[Candidate], ctx=None) -> Optional[Candidate]:
    pool = _filter_hard(cands, ctx, max_viol=0)
    if len(pool) < 1:
        pool = _filter_hard(cands, ctx, max_viol=1)
    if len(pool) < 1:
        pool = [c for c in cands if c.get("model_answer")]
    return select_current_answer_majority(pool, ctx)


def select_filter_then_short(cands: List[Candidate], ctx=None) -> Optional[Candidate]:
    pool = _filter_hard(cands, ctx, max_viol=0) or _filter_hard(cands, ctx, max_viol=1)
    if not pool:
        pool = [c for c in cands if c.get("model_answer")]
    buckets = defaultdict(list)
    for c in pool:
        key = _answer_key(c)
        if key is not None:
            buckets[key].append(c)
    if not buckets:
        return None
    best_key = max(
        buckets.keys(),
        key=lambda k: (
            len(buckets[k]),
            -min(_safe_int(c.get("depth"), 99) for c in buckets[k]),
            max(_cand_verifier(c, ctx) for c in buckets[k]),
        ),
    )
    return pick_by_key(
        buckets[best_key],
        lambda c: (
            -_safe_int(c.get("depth"), 99),
            1 if prefers_direct_root(c) else 0,
            _cand_verifier(c, ctx),
        ),
        reverse=True,
    )


def select_filter_then_nodup(cands: List[Candidate], ctx=None) -> Optional[Candidate]:
    pool = _filter_hard(cands, ctx, max_viol=0) or _filter_hard(cands, ctx, max_viol=1)
    if not pool:
        pool = [c for c in cands if c.get("model_answer")]
    buckets = defaultdict(list)
    for c in pool:
        key = _answer_key(c)
        if key is not None:
            buckets[key].append(c)
    if not buckets:
        return None

    def cluster_score(k: str):
        group = buckets[k]
        plan = parse_plan(group[0].get("model_answer"))
        agents = tool_sequence(plan) or ()
        dup = len(agents) - len(set(agents))
        return (
            len(group),
            -dup,
            -len(agents),
            max(_cand_verifier(c, ctx) for c in group),
            -min(_safe_int(c.get("depth"), 99) for c in group),
        )

    best_key = max(buckets.keys(), key=cluster_score)
    return pick_by_key(
        buckets[best_key],
        lambda c: (
            _cand_verifier(c, ctx),
            -_safe_int(c.get("depth"), 99),
            1 if prefers_direct_root(c) else 0,
        ),
        reverse=True,
    )


FILTER_STRATEGIES: Dict[str, SelectFn] = {
    "filter_then_majority": select_filter_then_majority,
    "filter_then_short": select_filter_then_short,
    "filter_then_nodup": select_filter_then_nodup,
    "current_answer_majority": select_current_answer_majority,
}


def select_answer(
    strategy: str,
    cands: List[Candidate],
    ctx=None,
) -> Optional[Candidate]:
    """Dispatch gold-free selector. `majority` is an alias for exact-answer majority."""
    name = strategy or "majority"
    if name == "majority":
        name = "current_answer_majority"
    fn = FILTER_STRATEGIES.get(name)
    if fn is None:
        raise ValueError(f"Unknown answer_selector={strategy!r}. Choose from {ONLINE_SELECTORS}")
    return fn(cands, ctx)
