import os


def init_wandb(args):
    if getattr(args, "disable_wandb", False) or os.getenv("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            entity="agents-research",
            project="disaster-management-agent",
            config=vars(args) if hasattr(args, "__dict__") else args,
        )
    except Exception as e:
        print(f"[WARN] wandb init skipped: {e}")
        return None


def finish_wandb(run, metrics=None):
    if run is None:
        return
    try:
        if metrics:
            run.log(metrics)
        run.finish()
    except Exception as e:
        print(f"[WARN] wandb finish skipped: {e}")
