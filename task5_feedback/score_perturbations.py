from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def value(row, keys, required=True, default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    if required:
        raise KeyError(f"Expected one of {keys}; row keys are {list(row)}")
    return default


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        pid = str(value(row, ["problem_id", "id"]))
        variant = str(value(row, ["variant_type", "category", "variant"]))
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"Unknown variant_type={variant!r}; expected {sorted(EXPECTED_VARIANTS)}")
        if variant in by_problem[pid]:
            raise ValueError(f"Duplicate variant {variant!r} for problem {pid}")
        by_problem[pid][variant] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def get_response(row):
    return str(value(row, ["response", "completion", "text", "output"]))


def get_prompt(row):
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user" and message.get("content"):
                return str(message["content"])
    return str(value(row, ["question", "problem", "prompt"], required=False, default=""))


def get_gold(row):
    return str(value(row, ["gold_final", "gold_answer", "answer", "gold"]))


def preferred_by_verifier(a_reward, b_reward):
    if a_reward > b_reward:
        return "A"
    if b_reward > a_reward:
        return "B"
    return "TIE"


def classify(preference):
    if preference == "TIE":
        return "tie"
    return "better" if preference == "A" else "wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--out-dir", default="results/task5")
    ap.add_argument("--data-path", default=None, help="Override diagnostic JSONL path from config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    diagnostic_path = args.data_path
    if diagnostic_path is None:
        paths = cfg.get("paths", {})
        for key in ("task5_diagnostics", "diagnostic_eval", "diagnostics", "math_diagnostics"):
            if paths.get(key):
                diagnostic_path = paths[key]
                break
            if cfg.get(key):
                diagnostic_path = cfg[key]
                break
    if diagnostic_path is None:
        raise KeyError(
            "Could not resolve diagnostic JSONL path. Checked paths.task5_diagnostics, "
            "paths.diagnostic_eval, paths.diagnostics, and paths.math_diagnostics. "
            "Pass --data-path or add the path to configs/base.yaml."
        )
    groups = load_diagnostic_groups(diagnostic_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = PairwiseAIJudge(cfg, out_dir / "diagnostic_judge_cache.json")
    pair_rows = []
    response_rows = []

    for pid, variants in groups.items():
        # All variants should share the same underlying prompt and gold answer.
        clean = variants["clean_correct"]
        prompt = get_prompt(clean)
        gold = get_gold(clean)

        rewards = {}
        responses = {}
        for variant_name, row in variants.items():
            response = get_response(row)
            row_gold = get_gold(row)
            if str(row_gold).strip() != str(gold).strip():
                # Gold answer should be constant across controlled variants.
                raise ValueError(f"Gold answer mismatch for problem {pid}, variant {variant_name}")
            reward = float(exact_reward(response, gold))
            rewards[variant_name] = reward
            responses[variant_name] = response
            response_rows.append({
                "problem_id": pid,
                "variant_type": variant_name,
                "gold_final": gold,
                "response": response,
                "exact_reward": reward,
                "word_count": len(response.split()),
                "format_compliant": int(bool(re.search(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?", response))),
            })

        comparisons = [
            ("reasoning_only", "clean_correct", "corrupt_reasoning_correct_final"),
            ("outcome_change", "clean_correct", "good_reasoning_wrong_final"),
            ("persuasive_filler", "clean_correct", "persuasive_filler_correct"),
            ("gold_distractor", "clean_correct", "gold_distractor_wrong_final"),
        ]

        for category, a_name, b_name in comparisons:
            a, b = responses[a_name], responses[b_name]
            a_reward, b_reward = rewards[a_name], rewards[b_name]
            verifier_pref = preferred_by_verifier(a_reward, b_reward)
            judge_pref = str(judge.compare(prompt, a, b)).strip().upper()
            if judge_pref not in {"A", "B", "TIE"}:
                judge_pref = "TIE"

            pair_rows.append({
                "problem_id": pid,
                "category": category,
                "variant_a": a_name,
                "variant_b": b_name,
                "reward_a": a_reward,
                "reward_b": b_reward,
                "verifier_preference": verifier_pref,
                "verifier_result": classify(verifier_pref),
                "judge_preference": judge_pref,
                "judge_result": classify(judge_pref),
                "response_a_words": len(a.split()),
                "response_b_words": len(b.split()),
            })

    responses_df = pd.DataFrame(response_rows)
    pairs_df = pd.DataFrame(pair_rows)
    responses_df.to_csv(out_dir / "diagnostic_response_scores.csv", index=False)
    pairs_df.to_csv(out_dir / "diagnostic_pair_detail.csv", index=False)

    summary_rows = []
    for category, group in pairs_df.groupby("category"):
        for mechanism, col in [
            ("RLVR_verifier", "verifier_result"),
            ("RLAIF_judge", "judge_result"),
        ]:
            n = len(group)
            counts = group[col].value_counts()
            summary_rows.append({
                "category": category,
                "mechanism": mechanism,
                "n_pairs": n,
                "better_response_rate": counts.get("better", 0) / n if n else float("nan"),
                "tie_rate": counts.get("tie", 0) / n if n else float("nan"),
                "wrong_preference_rate": counts.get("wrong", 0) / n if n else float("nan"),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "diagnostic_pair_summary.csv", index=False)

    # Sensitivity metrics are restricted to valid controlled pairs.
    sensitivity_rows = []
    for metric, category, valid_condition in [
        ("Sreason", "reasoning_only", lambda g: (g["reward_a"] == 1) & (g["reward_b"] == 1)),
        ("Soutcome", "outcome_change", lambda g: (g["reward_a"] == 1) & (g["reward_b"] == 0)),
    ]:
        group = pairs_df[pairs_df["category"] == category]
        eligible = group[valid_condition(group)]
        for mechanism, col in [
            ("RLVR_verifier", "verifier_result"),
            ("RLAIF_judge", "judge_result"),
        ]:
            n = len(eligible)
            counts = eligible[col].value_counts()
            sensitivity_rows.append({
                "metric": metric,
                "mechanism": mechanism,
                "n_eligible_pairs": n,
                "sensitivity_better_rate": counts.get("better", 0) / n if n else float("nan"),
                "tie_rate": counts.get("tie", 0) / n if n else float("nan"),
                "wrong_preference_rate": counts.get("wrong", 0) / n if n else float("nan"),
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out_dir / "diagnostic_sensitivity.csv", index=False)

    print("\nDiagnostic summary:")
    print(summary.to_string(index=False))
    print("\nSensitivity:")
    print(sensitivity.to_string(index=False))
    print(f"\nSaved diagnostic outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
