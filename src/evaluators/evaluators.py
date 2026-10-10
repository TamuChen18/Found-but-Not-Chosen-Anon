"""DisasterBench evaluator.

Loads the evaluator of DisasterBench_Open at the pinned commit (third_party/DisasterBench_Open, set up by
scripts/prepare_disasterbench.py). Exact Match (check_answers_equivalence) and the Tool / Parameter / Dependency
checks are used unchanged. Answer extraction (isolate_answer, extract_answer_from_model_completion) is the
version that produced the stored `model_answer` of every candidate pool; the pinned upstream extraction is more
lenient and parses a few dozen raw completions differently, so it is overridden here to keep generation and
offline re-extraction consistent with the released pools.
"""
import importlib.util
import json
import re
import traceback
from pathlib import Path

from json_repair import repair_json

_PATH = Path(__file__).resolve().parents[2] / "third_party" / "DisasterBench_Open" / "evaluators" / "evaluators.py"
if not _PATH.exists():
    raise FileNotFoundError(f"{_PATH} not found; run `python scripts/prepare_disasterbench.py` first")
_spec = importlib.util.spec_from_file_location("disasterbench_evaluators", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class Evaluator(_mod.Evaluator):
    def isolate_answer(self, text) -> str:
        if text is None:
            return None
        assert isinstance(text, str)
        text = self.normalize_outputs(text)
        text = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", " ")
        # 1) a JSON array of objects: [ {...}, {...}, ... ]
        pattern = r'\[\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}(?:\s*,\s*\{(?:[^{}]|\{[^{}]*\})*\})*)\s*\]'
        match = re.search(pattern, text, flags=re.S)
        if match:
            return match.group(0)
        # 2) a comma-separated sequence of JSON objects without outer brackets
        obj_seq_pattern = r'\{\s*(?:[^{}]|\{[^{}]*\})*\}\s*(?:,\s*\{\s*(?:[^{}]|\{[^{}]*\})*\}\s*)+'
        seq_match = re.search(obj_seq_pattern, text, flags=re.S)
        if seq_match:
            return "[" + seq_match.group(0).strip() + "]"
        # 3) a single JSON object that looks like a step item
        obj_pattern = r'\{\s*(?:[^{}]|\{[^{}]*\})*\}\s*'
        obj_match = re.search(obj_pattern, text, flags=re.S)
        if obj_match:
            candidate = obj_match.group(0).strip()
            if '"agent"' in candidate and '"step"' in candidate:
                return "[" + candidate + "]"
        return None

    def extract_answer_from_model_completion(self, completion) -> str:
        assert isinstance(completion, str)
        answer_split = self.isolate_answer(completion)
        if answer_split is None or len(answer_split) == 0:
            return None
        json_str = None
        try:
            json_str = json.dumps(json.loads(answer_split))
        except Exception:
            try:
                json_str = json.dumps(json.loads(repair_json(answer_split)))
            except Exception:
                print(traceback.format_exc())
        return json_str

    @staticmethod
    def _normalize_plan_steps(x):
        """Flatten occasional nested lists emitted by some models into step dicts."""
        if isinstance(x, dict):
            return [x]
        if isinstance(x, list):
            out = []
            for item in x:
                out.extend(Evaluator._normalize_plan_steps(item))
            return out
        return []
