from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])
    raise NotImplementedError(
        "TODO(student): launch matched DPO beta forks from the original policy initialization and evaluate them under a common protocol."
    )


if __name__ == "__main__":
    main()
