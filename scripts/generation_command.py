#!/usr/bin/env python3
"""
Print the main.py command that produced a released run, from its config.json.

  python scripts/generation_command.py runs/gpt-4o-mini/no_seed/cot_tree/config.json

Only arguments that differ from the parser defaults are printed. Backend switches are environment
variables of the OpenRouter-compatible client and are not stored in config.json (see README):
OPENROUTER_SEND_SEED, OPENROUTER_ENABLE_THINKING, OPENROUTER_BASE_URL.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DERIVED = {"file", "run_outputs_dir", "logdir", "answer_sheets_dir"}


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    argv, sys.argv = sys.argv, [argv_0 := "main.py"]
    from config.args import parse_args
    defaults = vars(parse_args())
    sys.argv = argv
    parts = ["python", argv_0]
    for k, v in cfg.items():
        if k in DERIVED or k not in defaults or defaults[k] == v:
            continue
        if k == "disable_wandb":
            parts.append("--disable_wandb")
            continue
        parts += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]
    print(" ".join(shlex.quote(p) for p in parts))


if __name__ == "__main__":
    main()
