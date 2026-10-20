from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from common.data import load_yaml, read_jsonl
from common.models import load_tokenizer
from task1_dpo.evaluate import generate_response, load_policy


# ============================================================
# Helpers
# ============================================================

def normalize_stratum(value):
    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    mapping = {
        "preferred_longer": "preferred-longer",
        "preferred-longer": "preferred-longer",

        "length_matched": "matched-length",
        "length-matched": "matched-length",
        "matched_length": "matched-length",
        "matched-length": "matched-length",

        "rejected_longer": "rejected-longer",
        "rejected-longer": "rejected-longer",
    }

    return mapping.get(value, value)


def extract_word_limit(prompt):
    patterns = [
        r"\bin\s+(\d+)\s+words?\b",
        r"\bwithin\s+(\d+)\s+words?\b",
        r"\bno more than\s+(\d+)\s+words?\b",
        r"\bmaximum\s+(\d+)\s+words?\b",
        r"\bmax(?:imum)?\s+(\d+)\s+words?\b",
        r"\b(\d+)\s+words?\s+or less\b",
        r"\blimited to\s+(\d+)\s+words?\b",
        r"\blimit(?:ed)?\s+to\s+(\d+)\s+words?\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            prompt,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def count_words(text):
    return len(
        re.findall(r"\b\w+\b", text)
    )


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    model_path,
    rows,
    cfg,
    label,
):

    print()
    print("=" * 70)
    print(f"EVALUATING: {label}")
    print("=" * 70)

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    model = load_policy(
        cfg,
        adapter_path=model_path,
        trainable=False,
    )

    model.eval()

    results = []

    for i, row in enumerate(rows):

        # ----------------------------------------------------
        # Build the prompt messages
        # ----------------------------------------------------

        prompt_messages = [
            {
                "role": "user",
                "content": row["prompt"],
            }
        ]

        # ----------------------------------------------------
        # Generate response
        # ----------------------------------------------------

        generated = generate_response(
            policy=model,
            tokenizer=tokenizer,
            prompt_messages=prompt_messages,
            max_length=cfg["max_sequence_length"],
            max_new_tokens=cfg["max_generation_tokens"],
            temperature=0.7,
            top_p=0.9,
        )

        # ----------------------------------------------------
        # Length
        # ----------------------------------------------------

        generated_words = count_words(
            generated
        )

        generated_tokens = len(
            tokenizer.encode(
                generated,
                add_special_tokens=False,
            )
        )

        # ----------------------------------------------------
        # Word-limit compliance
        # ----------------------------------------------------

        word_limit = extract_word_limit(
            row["prompt"]
        )

        if word_limit is not None:
            word_limit_compliant = (
                generated_words <= word_limit
            )
        else:
            word_limit_compliant = None

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "prompt_id": row.get(
                    "prompt_id"
                ),

                "length_stratum": normalize_stratum(
                    row.get("length_stratum")
                ),

                "chosen_tokens": row.get(
                    "chosen_tokens"
                ),

                "rejected_tokens": row.get(
                    "rejected_tokens"
                ),

                "length_difference": row.get(
                    "length_difference"
                ),

                "generated_tokens": generated_tokens,

                "generated_words": generated_words,

                "word_limit": word_limit,

                "word_limit_compliant": (
                    word_limit_compliant
                ),

                "generated_response": generated,
            }
        )

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/{len(rows)}"
            )

    # --------------------------------------------------------
    # Free memory
    # --------------------------------------------------------

    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================
# Per-stratum summary
# ============================================================

def summarize_by_stratum(results):

    strata = [
        "preferred-longer",
        "matched-length",
        "rejected-longer",
    ]

    summary = {}

    for stratum in strata:

        subset = [
            r
            for r in results
            if r["length_stratum"] == stratum
        ]

        if not subset:
            summary[stratum] = {
                "count": 0,
                "mean_generated_tokens": None,
                "mean_generated_words": None,
            }
            continue

        token_lengths = [
            r["generated_tokens"]
            for r in subset
        ]

        word_lengths = [
            r["generated_words"]
            for r in subset
        ]

        summary[stratum] = {
            "count": len(subset),

            "mean_generated_tokens": (
                sum(token_lengths)
                / len(token_lengths)
            ),

            "mean_generated_words": (
                sum(word_lengths)
                / len(word_lengths)
            ),
        }

    return summary


# ============================================================
# Word-limit summary
# ============================================================

def summarize_word_limits(results):

    limited = [
        r
        for r in results
        if r["word_limit"] is not None
    ]

    if not limited:
        return {
            "prompts_with_word_limit": 0,
            "compliant": 0,
            "non_compliant": 0,
            "compliance_rate": None,
        }

    compliant = [
        r
        for r in limited
        if r["word_limit_compliant"]
    ]

    non_compliant = [
        r
        for r in limited
        if not r["word_limit_compliant"]
    ]

    return {
        "prompts_with_word_limit": len(limited),
        "compliant": len(compliant),
        "non_compliant": len(non_compliant),
        "compliance_rate": (
            len(compliant)
            / len(limited)
        ),
    }


# ============================================================
# Print summary
# ============================================================

def print_summary(
    model_name,
    results,
):

    print()
    print("-" * 70)
    print(model_name)
    print("-" * 70)

    summary = summarize_by_stratum(
        results
    )

    for stratum in [
        "preferred-longer",
        "matched-length",
        "rejected-longer",
    ]:

        values = summary[stratum]

        if values["count"] == 0:
            print(
                f"{stratum}: no examples"
            )
            continue

        print(
            f"{stratum}: "
            f"n={values['count']}, "
            f"mean generated tokens="
            f"{values['mean_generated_tokens']:.2f}, "
            f"mean generated words="
            f"{values['mean_generated_words']:.2f}"
        )

    word_limit = summarize_word_limits(
        results
    )

    if word_limit["compliance_rate"] is not None:

        print(
            f"Word-limit compliance: "
            f"{word_limit['compliance_rate'] * 100:.2f}% "
            f"({word_limit['compliant']}/"
            f"{word_limit['prompts_with_word_limit']})"
        )

    else:

        print(
            "Word-limit compliance: "
            "No explicit word limits found."
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    parser.add_argument(
        "--standard-adapter",
        default="outputs/task1_dpo/standard",
    )

    parser.add_argument(
        "--balanced-adapter",
        default="outputs/task1_dpo/length_balanced",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    cfg = load_yaml(
        args.config
    )

    # --------------------------------------------------------
    # Load length-stratified evaluation set
    # --------------------------------------------------------

    rows = read_jsonl(
        cfg["paths"]["dpo_length_eval"]
    )

    print(
        "Length-stratified evaluation rows:",
        len(rows),
    )

    # --------------------------------------------------------
    # Verify strata
    # --------------------------------------------------------

    strata_counts = {}

    for row in rows:

        stratum = normalize_stratum(
            row.get("length_stratum")
        )

        strata_counts[stratum] = (
            strata_counts.get(stratum, 0) + 1
        )

    print(
        "Stratum counts:",
        strata_counts,
    )

    # --------------------------------------------------------
    # Standard DPO
    # --------------------------------------------------------

    standard_results = evaluate_model(
        model_path=args.standard_adapter,
        rows=rows,
        cfg=cfg,
        label="STANDARD DPO",
    )

    # --------------------------------------------------------
    # Length-balanced DPO
    # --------------------------------------------------------

    balanced_results = evaluate_model(
        model_path=args.balanced_adapter,
        rows=rows,
        cfg=cfg,
        label="LENGTH-BALANCED DPO",
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_summary(
        "STANDARD DPO",
        standard_results,
    )

    print_summary(
        "LENGTH-BALANCED DPO",
        balanced_results,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = Path(
        cfg["results_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "length_confounding_results.json"
    )

    output = {
        "evaluation_rows": len(rows),

        "stratum_counts": strata_counts,

        "standard_dpo": {
            "per_stratum": summarize_by_stratum(
                standard_results
            ),

            "word_limit": summarize_word_limits(
                standard_results
            ),

            "raw_results": standard_results,
        },

        "length_balanced_dpo": {
            "per_stratum": summarize_by_stratum(
                balanced_results
            ),

            "word_limit": summarize_word_limits(
                balanced_results
            ),

            "raw_results": balanced_results,
        },
    }

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(output_path)


if __name__ == "__main__":
    main()