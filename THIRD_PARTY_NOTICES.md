# Third-party notices

The MIT license in `LICENSE` covers the original code, documentation, figures and result files of this repository.
The third-party code and data listed below, including the portions of it that were adapted into files of this
repository, remain under their own licenses and copyright notices, reproduced here in full. Verbatim copies of the
license files are in `licenses/`.

## rStar

Source: rStar (<https://github.com/zhentingqi/rStar>), commit `cab41df656bddc4d4b61ac87721315dd48b45b48`.
License: MIT, Copyright (c) 2024 rStar Team (`licenses/rStar-LICENSE`).

Files adapted from rStar (modified for structured workflow planning on DisasterBench):

| File in this repository | rStar source |
|---|---|
| `src/mcts/mcts_base.py` | `run_src/MCTS_backbone.py` |
| `src/mcts/mcts_node.py` | `run_src/MCTS_for_reasoning.py` |
| `src/mcts/mcts_generator.py` | `run_src/MCTS_for_reasoning.py` |
| `src/utils/agent_utils.py` | `run_src/rstar_utils.py` |
| `src/utils/common_utils.py` | `run_src/rstar_utils.py`, `common/utils.py` |

Files containing smaller portions adapted from rStar:

| File in this repository | rStar source |
|---|---|
| `src/config/args.py` | `common/arguments.py`, `run_src/do_generate.py` |
| `src/generate.py` | `run_src/do_generate.py` |
| `src/mcts/mcts_search.py` | `run_src/MCTS_for_reasoning.py` |
| `baselines/cot.py` | `eval_src/Evaluator.py` |

The DisasterBench_Open evaluator (`evaluators/evaluators.py`, see below; loaded by `src/evaluators/evaluators.py`)
also contains answer-selection code from rStar `eval_src/Evaluator.py` (`find_most_confident_answer` and the
`stochastic_*` selection helpers).

~~~
    MIT License

    Copyright (c) 2024 rStar Team

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
~~~

## ThoughtICR

Source: ThoughtICR (<https://github.com/jinyangwu/ThoughtICR>), commit `b174a28563f1c074eb120e20be1fe470b05f67a2`,
itself derived from rStar. License: MIT, Copyright (c) 2025 Jinyang Wu (`licenses/ThoughtICR-LICENSE`).

Files containing portions adapted from ThoughtICR (in addition to the rStar portions listed above):

| File in this repository | ThoughtICR source |
|---|---|
| `src/mcts/mcts_base.py` | `run_src/MCTS_backbone.py` |
| `src/mcts/mcts_node.py` | `run_src/MCTS_for_reasoning.py` |
| `src/mcts/mcts_generator.py` | `run_src/MCTS_for_reasoning.py` |
| `src/utils/agent_utils.py` | `run_src/hiaricl_utils.py`, `run_src/MCTS_for_reasoning.py` |
| `src/config/args.py` | `common/arguments.py` |
| `src/generate.py` | `run_src/generate.py` |

~~~
    MIT License

    Copyright (c) 2025 Jinyang Wu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
~~~

## Earlier DisasterBench MCTS prototype

The adaptation of the rStar / ThoughtICR search to DisasterBench builds on an earlier, unpublished prototype by a
collaborator (name withheld for anonymous review). Parts of the following files originate from that prototype:
`src/prompts/prompts.yaml`, `src/prompts/prompt_builder.py`, `src/mcts/mcts_search.py`, `src/mcts/mcts_generator.py`,
`src/mcts/mcts_node.py`, `src/generate.py`, `src/utils/agent_utils.py`, `src/utils/logger.py`, `src/interfaces/` and
`baselines/test_baseline.py`.

## Minimal MCTS implementation (public domain)

`src/mcts/mcts_base.py` (through rStar `run_src/MCTS_backbone.py`) is based on "A minimal implementation of Monte
Carlo tree search (MCTS) in Python 3" by Luke Harold Miles, July 2019, released under a Public Domain Dedication
(<https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1>). The original attribution is kept at the top of
the file.

## DisasterBench_Open

Benchmark tasks (`data/benchmark.jsonl`), evaluator (`evaluators/evaluators.py`) and tool manifest
(`interfaces/tools/tools_manifest.json`) at commit `408672bf0da489d94e6245900a87ec913b3d69f9`, used through
`third_party/DisasterBench_Open/` and the files derived from it by `scripts/prepare_disasterbench.py`
(`data/train.jsonl`, `data/test.jsonl`, `src/interfaces/tools/tools_manifest.json`).
License: MIT (`licenses/DisasterBench_Open-LICENSE`).

~~~
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~
