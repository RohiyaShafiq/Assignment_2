from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            "Generate and save SFT responses first: " + str(src)
        )

    base_rows = read_jsonl(str(src))
    ids = fixed_audit_ids(
        base_rows,
        int(cfg["manual_audit_per_class"]),
        int(cfg["seed"]),
    )

    # Save the fixed, balanced subset.
    ids_path = outdir / "manual_audit_ids.csv"
    pd.DataFrame({"xstest_id": ids}).to_csv(ids_path, index=False)

    # Create one annotation row for every policy-response pair.
    annotation_rows = []
    for policy in ["sft", "dpo", "ppo", "grpo"]:
        path = outdir / f"generated_{policy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Generate all four policies before creating the audit: {path}"
            )

        df = pd.DataFrame(read_jsonl(str(path)))
        subset = df[df["xstest_id"].isin(ids)]

        if len(subset) != len(ids):
            raise ValueError(
                f"{policy}: expected {len(ids)} audit responses, "
                f"found {len(subset)}."
            )

        for _, row in subset.iterrows():
            annotation_rows.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy,
                "benchmark_class": row["benchmark_class"],
                "type": row["type"],
                "prompt": row["prompt"],
                "response": row["response"],
                "manual_label": "",
                "manual_rationale": "",
            })

    annotation = pd.DataFrame(annotation_rows)
    annotation_path = outdir / "manual_audit_annotations.csv"

    # Do not silently overwrite completed human annotations.
    if annotation_path.exists():
        raise FileExistsError(
            f"{annotation_path} already exists. Preserve your completed "
            "annotations; move it aside only if you intentionally want "
            "to create a new audit sheet."
        )

    annotation.to_csv(annotation_path, index=False)

    print("Fixed audit IDs:", ids_path)
    print("Human annotation sheet:", annotation_path)
    print("Number of prompts:", len(ids))
    print("Number of policy-response annotations:", len(annotation))
    print("Annotate without viewing AI labels first.")
    print("Allowed labels:", [
        "SAFE_ANSWER",
        "JUSTIFIED_REFUSAL",
        "UNSAFE_COMPLIANCE",
        "OVER_REFUSAL",
        "AMBIGUOUS",
    ])


if __name__ == "__main__":
    main()
