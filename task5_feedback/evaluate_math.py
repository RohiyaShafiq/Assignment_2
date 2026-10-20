from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str, override: str | None = None):
    """Resolve dataset paths across course config naming conventions."""
    if override:
        return override
    paths = cfg.get("paths", {})
    aliases = {
        "gsm": ("gsm_eval", "gsm8k_eval", "gsm8k_test", "gsm_test", "gsm8k"),
        "transfer": ("math_transfer_eval", "svamp_eval", "svamp_test", "math_transfer", "svamp"),
    }
    if dataset not in aliases:
        raise ValueError(dataset)
    for key in aliases[dataset]:
        if paths.get(key):
            return paths[key]
        if cfg.get(key):
            return cfg[key]
    raise KeyError(
        f"Could not resolve the {dataset!r} dataset path from config. "
        f"Checked paths keys {aliases[dataset]}. Available paths keys: {list(paths)}. "
        "Pass --data-path explicitly or add the path to configs/base.yaml."
    )


def load_math_evaluation(config_path: str, dataset: str, data_path: str | None = None):
    cfg = load_yaml(config_path)
    resolved_path = dataset_path(cfg, dataset, data_path)
    rows = read_jsonl(resolved_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    print(f"Dataset file: {repo_path(resolved_path)}")
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def first_value(row, keys, required=True, default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    if required:
        raise KeyError(
            f"Could not find any of fields {keys}. Available fields: {list(row)}"
        )
    return default


def get_prompt(row):
    # These course JSONL files provide a messages array containing the intended
    # user instruction (including the required #### output format). Prefer it
    # over a pre-rendered prompt so we do not accidentally wrap a chat template
    # inside another chat template.
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user" and message.get("content"):
                return str(message["content"])
    return str(first_value(row, ["question", "problem", "input", "prompt"]))


def get_gold_final(row):
    value = first_value(
        row,
        ["gold_final", "gold_answer", "answer", "target", "final_answer", "solution"],
    )
    # GSM8K commonly stores answer with explanation followed by #### number.
    value = str(value)
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", value)
    if matches:
        return matches[-1].replace(",", "")
    # If answer field is already just the gold number, return it.
    numeric = re.fullmatch(r"\s*[-+]?\d[\d,]*(?:\.\d+)?\s*", value)
    if numeric:
        return value.strip().replace(",", "")
    # Some datasets store an answer explanation and a separate numeric field.
    return str(first_value(row, ["gold", "numeric_answer", "answer_value"])).replace(",", "")


def get_example_id(row, index):
    return str(first_value(
        row, ["id", "problem_id", "example_id", "prompt_id", "source_index", "svamp_id", "idx"],
        required=False, default=index
    ))


def generation_settings(cfg):
    gen = dict(cfg.get("generation", {}))
    # feedback.yaml defines math_max_new_tokens at the top level; allow a
    # nested generation override if the project adds one later.
    max_new = gen.get(
        "max_new_tokens",
        cfg.get("math_max_new_tokens", cfg.get("max_new_tokens", 512)),
    )
    do_sample = gen.get("do_sample", False)
    kwargs = {"max_new_tokens": int(max_new), "do_sample": bool(do_sample)}
    for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if key in gen and (key != "temperature" or do_sample):
            kwargs[key] = gen[key]
    return kwargs


def generate_completion(model, tokenizer, prompt, cfg):
    # Preserve a chat template when available; otherwise use the raw prompt.
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": str(prompt)}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(inputs, torch.Tensor):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs
            attention_mask = None
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(device)
        prompt_length = input_ids.shape[-1]
    else:
        enc = tokenizer(str(prompt), return_tensors="pt")
        device = next(model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in enc.items()}
        prompt_length = model_inputs["input_ids"].shape[-1]

    kwargs = generation_settings(cfg)
    kwargs.setdefault("pad_token_id", tokenizer.pad_token_id or tokenizer.eos_token_id)

    model.eval()
    with torch.no_grad():
        output = model.generate(**model_inputs, **kwargs)
    completion_ids = output[0][prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def format_compliant(text):
    return int(bool(re.search(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?", str(text))))


def judge_result(judge, problem, a, b):
    result = str(judge.compare(problem, a, b)).strip().upper()
    if result not in {"A", "B", "TIE"}:
        return "TIE"
    return result


def run_dataset(cfg, rows, tokenizer, dataset_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load one policy at a time: keeping SFT, RLVR, and RLAIF checkpoints in
    # memory simultaneously can exhaust GPU RAM for large causal LMs.
    generations = []
    specs = policy_specs(cfg)
    for name in specs:
        print(f"Loading {name} policy...")
        model = load_frozen_policy(cfg, name)
        for idx, row in enumerate(rows):
            prompt = get_prompt(row)
            gold = get_gold_final(row)
            eid = get_example_id(row, idx)
            print(f"\r{dataset_name}: {idx+1}/{len(rows)} | {name}", end="")
            response = generate_completion(model, tokenizer, prompt, cfg)
            reward = exact_reward(response, gold)
            generations.append({
                "dataset": dataset_name,
                "example_id": eid,
                "policy": name,
                "prompt": prompt,
                "gold_final": gold,
                "response": response,
                "exact_correct": int(reward),
                "format_compliant": format_compliant(response),
                "word_count": len(response.split()),
                "char_count": len(response),
            })
        print()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Instantiate the judge after generation to reduce peak memory usage.
    judge = PairwiseAIJudge(cfg, out_dir / f"{dataset_name}_judge_cache.json")

    outputs = pd.DataFrame(generations)
    outputs.to_json(
        out_dir / f"{dataset_name}_responses.jsonl",
        orient="records", lines=True, force_ascii=False
    )

    metrics = outputs.groupby("policy", as_index=False).agg(
        n=("exact_correct", "size"),
        exact_accuracy=("exact_correct", "mean"),
        format_compliance=("format_compliant", "mean"),
        mean_word_count=("word_count", "mean"),
        mean_char_count=("char_count", "mean"),
    )
    metrics["exact_accuracy_pct"] = 100 * metrics["exact_accuracy"]
    metrics["format_compliance_pct"] = 100 * metrics["format_compliance"]
    metrics.to_csv(out_dir / f"{dataset_name}_metrics.csv", index=False)

    # Compare every candidate with SFT on the same prompts.
    indexed = outputs.set_index(["example_id", "policy"])
    pair_rows = []
    for eid in outputs["example_id"].drop_duplicates():
        sft = indexed.loc[(eid, "sft")]
        for candidate_name in ("rlvr", "rlaif"):
            candidate = indexed.loc[(eid, candidate_name)]
            pref = judge_result(judge, sft["prompt"], sft["response"], candidate["response"])
            pair_rows.append({
                "example_id": eid,
                "candidate_policy": candidate_name,
                "judge_preference": pref,
                "sft_win": int(pref == "A"),
                "candidate_win": int(pref == "B"),
                "tie": int(pref == "TIE"),
                "sft_correct": int(sft["exact_correct"]),
                "candidate_correct": int(candidate["exact_correct"]),
                "verifier_preference": (
                    "A" if sft["exact_correct"] > candidate["exact_correct"]
                    else "B" if candidate["exact_correct"] > sft["exact_correct"]
                    else "TIE"
                ),
            })

    pair_detail = pd.DataFrame(pair_rows)
    pair_summary = pair_detail.groupby("candidate_policy", as_index=False).agg(
        comparisons=("judge_preference", "size"),
        sft_win_rate=("sft_win", "mean"),
        candidate_win_rate=("candidate_win", "mean"),
        tie_rate=("tie", "mean"),
    )
    pair_summary["candidate_pairwise_score"] = (
        pair_summary["candidate_win_rate"] + 0.5 * pair_summary["tie_rate"]
    )
    pair_summary.to_csv(out_dir / f"{dataset_name}_pairwise_summary.csv", index=False)
    pair_detail.to_csv(out_dir / f"{dataset_name}_pairwise_detail.csv", index=False)

    # Agreement is only meaningful as pairwise rank agreement; no reward calibration assumed.
    pair_detail["verifier_preference"] = pair_detail["verifier_preference"].str.upper()
    pair_detail["rank_agreement"] = (
        pair_detail["judge_preference"] == pair_detail["verifier_preference"]
    ).astype(int)
    pair_detail["unequal_correctness"] = (
        pair_detail["sft_correct"] != pair_detail["candidate_correct"]
    ).astype(int)
    pair_detail["judge_correct_direction"] = pair_detail.apply(
        lambda r: (
            int(r["judge_preference"] == r["verifier_preference"])
            if r["unequal_correctness"] and r["judge_preference"] != "TIE"
            else float("nan")
        ), axis=1
    )
    agreement = pair_detail.groupby("candidate_policy", as_index=False).agg(
        exact_rank_agreement=("rank_agreement", "mean"),
        unequal_correctness_pairs=("unequal_correctness", "sum"),
        judge_directional_agreement=("judge_correct_direction", "mean"),
        judge_tie_rate=("tie", "mean"),
    )
    agreement.to_csv(out_dir / f"{dataset_name}_verifier_judge_agreement.csv", index=False)

    # Release cached GPU allocations after all metrics have been written.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs, metrics, pair_summary, agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--data-path", default=None, help="Override the dataset path in the merged config")
    ap.add_argument("--out-dir", default="results/task5")
    args = ap.parse_args()

    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset, args.data_path)
    seed = int(cfg.get("seed", 6304))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    name = "gsm8k" if args.dataset == "gsm" else "svamp"
    run_dataset(cfg, rows, tokenizer, name, args.out_dir)


if __name__ == "__main__":
    main()
