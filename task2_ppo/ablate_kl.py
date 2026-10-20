from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    raise NotImplementedError(
        "TODO(student): run matched short PPO continuations from the exact same midpoint for each KL beta, then implement the requested reward/drift/entropy/length analysis."
    )


if __name__ == "__main__":
    main()
