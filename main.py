import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from dotenv import load_dotenv
from config.args import parse_args, post_process_args
from generate import main as mcts_main
import warnings
warnings.filterwarnings("ignore")

def main():
    """
    Entry point for candidate generation: MCTS (CoT-tree, MO-Tree) and baseline (BoN) runs.
    """
    load_dotenv()
    args = parse_args()
    args = post_process_args(args)

    if getattr(args, "seed", None) is not None:
        import random
        import numpy as np
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        print(f"[INFO] Seeded RNG with seed={seed}")

    print(f"[INFO] Starting execution with mode: {args.mode}")

    try:
        # --- MCTS: CoT-tree / MO-Tree ---
        if args.mode == "mcts":
            print("[INFO] Running MCTS reasoning pipeline...")
            assert args.if_use_cards == False, "Please set if_use_cards as False for MCTS"
            mcts_main(args)

        elif args.mode == "baseline":
            print("[INFO] Running baseline evaluation...")
            sys.path.insert(0, os.path.join(ROOT, "baselines"))
            from test_baseline import main as baseline_main
            baseline_main(args)
        else:
            print(f"[ERROR] Unknown mode '{args.mode}'. Expected one of ['mcts', 'baseline'].")

    except Exception as e:
        print(f"[FATAL] Error occurred during execution: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()