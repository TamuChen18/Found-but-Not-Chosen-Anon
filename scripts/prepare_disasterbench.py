#!/usr/bin/env python3
"""
Set up DisasterBench for this repository (the benchmark itself is not redistributed here).

1. Uses third_party/DisasterBench_Open if present, otherwise clones it at the pinned commit
   (DB_REPO_URL, override with --repo_url).
2. Verifies the files this repository uses against pinned sha256 hashes.
3. Writes the split used in the paper:
     data/train.jsonl   public task ids 1-50     (task_id unchanged)
     data/test.jsonl    public task ids 51-233   (task_id = public id - 50, i.e. 1-183)
   with fields task_id, task_desc, structured_plan.
4. Copies the tool manifest to src/interfaces/tools/tools_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "third_party" / "DisasterBench_Open"
DB_REPO_URL = "DisasterBench_Open repository (anonymized for review; the pinned files are included)"
DB_COMMIT = "408672bf0da489d94e6245900a87ec913b3d69f9"
PINNED = {
    "data/benchmark.jsonl": "c736a1cb879586e613a3299cff459a25919145dd2d6d2167c202c9d7a4b9b28f",
    "evaluators/evaluators.py": "087e769d19462ed67beb883b8bf3cfa9ab42b5b28f1a27640db30f77bfebcd1a",
    "interfaces/tools/tools_manifest.json": "f67b8bb65e54c525b8218f96c155b4b4ff2136db2034f88bedacd2bd861422bd",
}
N_TRAIN = 50
N_TEST = 183
# sha256 of json.dumps(rows, sort_keys=True) for the split files used in the experiments
SPLIT_SHA256 = {
    "train": "dc3f0256eafa4500842f4dbcca3f3c19ed750e7c27b027a5a563a58dfa6b2fab",
    "test": "29d7418d16481c68dd28f97d0fd927ad1b772a0a0e3dfe4195a0adef6c9c8859",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_url", default=DB_REPO_URL)
    args = ap.parse_args()

    if not DB_DIR.exists():
        DB_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", args.repo_url, str(DB_DIR)], check=True)
        subprocess.run(["git", "-C", str(DB_DIR), "checkout", "--quiet", DB_COMMIT], check=True)
    if (DB_DIR / ".git").exists():
        head = subprocess.run(["git", "-C", str(DB_DIR), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        assert head == DB_COMMIT, f"DisasterBench_Open is at {head}, expected {DB_COMMIT}"
    for rel, digest in PINNED.items():
        got = sha256(DB_DIR / rel)
        assert got == digest, f"{rel}: sha256 {got} != pinned {digest}"

    bench = [json.loads(l) for l in (DB_DIR / "data" / "benchmark.jsonl").read_text().splitlines() if l.strip()]
    by_id = {r["task_id"]: r for r in bench}
    assert sorted(by_id) == list(range(1, N_TRAIN + N_TEST + 1)), "unexpected DisasterBench task ids"
    fields = ("task_desc", "structured_plan")
    split = {
        "train": [{"task_id": i, **{k: by_id[i][k] for k in fields}} for i in range(1, N_TRAIN + 1)],
        "test": [{"task_id": i, **{k: by_id[i + N_TRAIN][k] for k in fields}} for i in range(1, N_TEST + 1)],
    }
    (ROOT / "data").mkdir(exist_ok=True)
    for name, rows in split.items():
        digest = canonical_sha256(rows)
        assert digest == SPLIT_SHA256[name], f"{name} split sha256 {digest} != {SPLIT_SHA256[name]}"
        (ROOT / "data" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote data/{name}.jsonl ({len(rows)} tasks)")

    tools = ROOT / "src" / "interfaces" / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DB_DIR / "interfaces" / "tools" / "tools_manifest.json", tools / "tools_manifest.json")
    print("wrote src/interfaces/tools/tools_manifest.json")


if __name__ == "__main__":
    main()
