from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    raise NotImplementedError(
        "TODO(student): run matched canonical-GRPO and Dr-GRPO short continuations, keeping prompt/generation/reward/KL/token budgets fixed, then analyze length-conditioned behavior."
    )


if __name__ == "__main__":
    main()
