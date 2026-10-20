from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.data import load_yaml


def read_csv_if_exists(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--results-dir", default="results/task5")
    args = ap.parse_args()

    # Load config to validate that the same course configuration is used.
    cfg = load_yaml(args.config)
    results = Path(args.results_dir)

    gsm_metrics = read_csv_if_exists(results / "gsm8k_metrics.csv")
    gsm_pairwise = read_csv_if_exists(results / "gsm8k_pairwise_summary.csv")
    gsm_agreement = read_csv_if_exists(results / "gsm8k_verifier_judge_agreement.csv")
    svamp_metrics = read_csv_if_exists(results / "svamp_metrics.csv")
    svamp_pairwise = read_csv_if_exists(results / "svamp_pairwise_summary.csv")
    diagnostic = read_csv_if_exists(results / "diagnostic_pair_summary.csv")
    sensitivity = read_csv_if_exists(results / "diagnostic_sensitivity.csv")

    required = {
        "GSM8K metrics": gsm_metrics,
        "GSM8K pairwise": gsm_pairwise,
        "GSM8K verifier/judge agreement": gsm_agreement,
        "SVAMP metrics": svamp_metrics,
        "SVAMP pairwise": svamp_pairwise,
        "diagnostic pair summary": diagnostic,
        "diagnostic sensitivity": sensitivity,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise FileNotFoundError(
            "Missing result files: " + ", ".join(missing) +
            "\nRun evaluate_math.py for --dataset gsm and --dataset transfer, "
            "and run score_perturbations.py first."
        )

    domain = gsm_metrics[["policy", "exact_accuracy", "mean_word_count"]].merge(
        svamp_metrics[["policy", "exact_accuracy", "mean_word_count"]],
        on="policy", suffixes=("_gsm8k", "_svamp")
    )
    domain["accuracy_drop"] = (
        domain["exact_accuracy_gsm8k"] - domain["exact_accuracy_svamp"]
    )
    domain["accuracy_drop_percentage_points"] = 100 * domain["accuracy_drop"]
    domain["mean_word_change"] = (
        domain["mean_word_count_svamp"] - domain["mean_word_count_gsm8k"]
    )
    domain.to_csv(results / "domain_comparison.csv", index=False)

    # A compact merged comparison table for the report.
    merged = gsm_metrics.merge(
        svamp_metrics, on="policy", suffixes=("_gsm8k", "_svamp")
    )
    merged.to_csv(results / "task5_policy_comparison.csv", index=False)

    print("\n=== GSM8K metrics ===")
    print(gsm_metrics.to_string(index=False))
    print("\n=== GSM8K AI pairwise vs SFT ===")
    print(gsm_pairwise.to_string(index=False))
    print("\n=== Verifier/judge agreement ===")
    print(gsm_agreement.to_string(index=False))
    print("\n=== Diagnostic pair results ===")
    print(diagnostic.to_string(index=False))
    print("\n=== Diagnostic sensitivity ===")
    print(sensitivity.to_string(index=False))
    print("\n=== SVAMP metrics ===")
    print(svamp_metrics.to_string(index=False))
    print("\n=== SVAMP AI pairwise vs SFT ===")
    print(svamp_pairwise.to_string(index=False))
    print("\n=== In-domain to OOD drop ===")
    print(domain.to_string(index=False))
    print(f"\nSaved final comparison tables to {results.resolve()}")


if __name__ == "__main__":
    main()
