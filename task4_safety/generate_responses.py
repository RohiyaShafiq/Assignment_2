from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer


POLICY_NAMES = ("sft", "dpo", "ppo", "grpo")


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))

    required = {
        "xstest_id", "prompt", "benchmark_class", "type"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"XSTest CSV is missing columns: {sorted(missing)}")

    if df["xstest_id"].duplicated().any():
        raise ValueError("xstest_id must be unique.")

    return df.reset_index(drop=True)


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(f"Unknown policy: {policy_name}")

    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(
        cfg, adapter_path=adapter, trainable=False
    )
    model.eval()

    df = load_xstest(cfg)
    records = []

    try:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size]
            prompts = [
                [{"role": "user", "content": str(p)}]
                for p in chunk["prompt"].tolist()
            ]

            with torch.inference_mode():
                gen = batch_generate(
                    model,
                    tokenizer,
                    prompts,
                    max_prompt_length=int(
                        cfg.get("safety_max_prompt_length", 256)
                    ),
                    max_new_tokens=int(cfg["safety_max_new_tokens"]),
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                )

            if not (
                len(gen["responses"])
                == len(gen["response_lengths"])
                == len(chunk)
            ):
                raise RuntimeError(
                    f"Generation returned an unexpected batch size for {policy_name}."
                )

            for (_, row), response, n_tok in zip(
                chunk.iterrows(),
                gen["responses"],
                gen["response_lengths"],
            ):
                records.append({
                    "xstest_id": int(row["xstest_id"]),
                    "policy": policy_name,
                    "prompt": str(row["prompt"]),
                    "benchmark_class": str(row["benchmark_class"]).upper(),
                    "type": str(row["type"]),
                    "response": str(response),
                    "response_tokens": int(n_tok),
                })

            print(
                f"{policy_name}: "
                f"{min(start + len(chunk), len(df))}/{len(df)}"
            )

    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(records) != len(df):
        raise RuntimeError(
            f"{policy_name}: expected {len(df)} responses, got {len(records)}"
        )

    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            import json
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if args.batch_size < 1:
        ap.error("--batch-size must be at least 1")

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    base_df = load_xstest(cfg)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(base_df))
    print("Output directory:", outdir)

    for policy_name in POLICY_NAMES:
        outpath = outdir / f"generated_{policy_name}.jsonl"

        # Reuse only a complete output with the exact expected ID order.
        if outpath.exists() and not bool(cfg.get("regenerate_safety_responses", False)):
            import json
            with outpath.open("r", encoding="utf-8") as f:
                old = [json.loads(line) for line in f if line.strip()]

            expected_ids = base_df["xstest_id"].astype(int).tolist()
            old_ids = [int(x["xstest_id"]) for x in old]
            if (
                old_ids == expected_ids
                and all(x.get("policy") == policy_name for x in old)
                and all("response" in x for x in old)
            ):
                print(f"Reusing complete response file: {outpath}")
                continue

            raise RuntimeError(
                f"{outpath} exists but does not match the current XSTest "
                "IDs/order. Back it up or set regenerate_safety_responses: true."
            )

        records = generate_for_policy(
            cfg, policy_name, batch_size=args.batch_size
        )
        write_jsonl(outpath, records)
        print(f"Saved {len(records)} responses: {outpath}")


if __name__ == "__main__":
    main()