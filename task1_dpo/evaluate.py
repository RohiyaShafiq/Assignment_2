from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path)

from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode)

def filter_long_prompts(tokenizer, rows, max_length):
    """
    Keep only examples whose prompt fits inside max_length.

    Prompts are never truncated because DPO should condition on the
    complete prompt for both chosen and rejected responses.
    """
    valid_rows = []
    removed = []

    for idx, row in enumerate(rows):
        messages = prompt_messages_from_preference(row)

        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        if len(prompt_ids) >= max_length:
            removed.append(
                {
                    "index": idx,
                    "prompt_tokens": len(prompt_ids),
                }
            )
        else:
            valid_rows.append(row)

    print(
        f"Prompt-length filtering: "
        f"{len(valid_rows)} valid / {len(rows)} total"
    )
    print(f"Removed {len(removed)} examples with prompts >= {max_length} tokens.")

    if removed:
        print("First removed examples:", removed[:10])

    return valid_rows


# Evaluation bundle
def load_evaluation_bundle(
    config_path: str,
    adapter: str):

    cfg = load_yaml(config_path)

    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    tokenizer = load_tokenizer(cfg["base_model"])

    rows = filter_long_prompts(
        tokenizer,
        rows,
        int(cfg["max_sequence_length"]),
    )
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)

    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": policy}


# Sequence log probability
def sequence_logprob(model, batch):

    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    shift_attention = attention_mask[:, 1:]
    shift_response_mask = response_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)).squeeze(-1)

    token_log_probs = (
        token_log_probs
        * shift_attention
        * shift_response_mask)

    return token_log_probs.sum(dim=-1)


# Exact token-level KL on generated sequences
def token_level_kl(
    policy,
    reference,
    batch):
    """
    Estimate KL(policy || reference) over the response tokens.

    Both models receive the same generated sequence.

    KL contribution at each response position:

        sum_v p_policy(v)
                 [log p_policy(v)
                  - log p_reference(v)]

    The value is averaged over response tokens.
    """

    device = next(policy.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"].to(device)

    with torch.no_grad():

        policy_outputs = policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False)

        with reference_mode(reference):

            reference_outputs = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False)

        policy_log_probs = F.log_softmax(
            policy_outputs.logits[:, :-1, :],
            dim=-1)

        reference_log_probs = F.log_softmax(
            reference_outputs.logits[:, :-1, :],
            dim=-1)

        policy_probs = (policy_log_probs.exp())

        token_kl = (
            policy_probs
            * (
                policy_log_probs
                - reference_log_probs)).sum(dim=-1)

        mask = (
            response_mask[:, 1:]
            * attention_mask[:, 1:])

        token_kl = token_kl * mask

        total_kl = token_kl.sum()

        token_count = mask.sum().clamp_min(1.0)

    return (total_kl / token_count).item()


# Build chosen/rejected batch
def make_pair_batch(tokenizer, rows, max_length):
    chosen = []
    rejected = []

    for row in rows:

        prompt = (
            prompt_messages_from_preference(
                row))

        yc, yr = (preference_responses(row))

        chosen.append(
            encode_prompt_response(
                tokenizer,
                prompt,
                yc,
                max_length))

        rejected.append(
            encode_prompt_response(
                tokenizer,
                prompt,
                yr,
                max_length))

    return (
        pad_batch(tokenizer, chosen),
        pad_batch(tokenizer, rejected),
    )


# Held-out preference accuracy
def evaluate_preference_accuracy(
    policy,
    reference,
    tokenizer,
    rows,
    max_length,
):
    """
    Evaluate held-out preference accuracy using mini-batches
    to avoid CUDA out-of-memory errors.

    Policy and reference models may be on different devices.
    """

    import gc

    policy.eval()
    reference.eval()

    batch_size = 1

    policy_margins = []
    reference_margins = []
    dpo_margins = []

    policy_device = next(policy.parameters()).device
    reference_device = next(reference.parameters()).device

    print(f"Evaluation mini-batch size: {batch_size}")
    print(f"Policy device: {policy_device}")
    print(f"Reference device: {reference_device}")

    with torch.inference_mode():

        for start in range(0, len(rows), batch_size):

            batch_rows = rows[start:start + batch_size]

            chosen_batch, rejected_batch = make_pair_batch(
                tokenizer,
                batch_rows,
                max_length,
            )

            # Move policy inputs to policy device
            policy_chosen_batch = {
                k: v.to(policy_device)
                for k, v in chosen_batch.items()
            }

            policy_rejected_batch = {
                k: v.to(policy_device)
                for k, v in rejected_batch.items()
            }

            # Move reference inputs to reference device
            reference_chosen_batch = {
                k: v.to(reference_device)
                for k, v in chosen_batch.items()
            }

            reference_rejected_batch = {
                k: v.to(reference_device)
                for k, v in rejected_batch.items()
            }

            # Policy log probabilities
            policy_chosen = sequence_logprob(
                policy,
                policy_chosen_batch,
            )

            policy_rejected = sequence_logprob(
                policy,
                policy_rejected_batch,
            )

            # Reference log probabilities
            with reference_mode(reference):

                ref_chosen = sequence_logprob(
                    reference,
                    reference_chosen_batch,
                )

                ref_rejected = sequence_logprob(
                    reference,
                    reference_rejected_batch,
                )

            # Margins
            policy_margin = (
                policy_chosen - policy_rejected
            )

            reference_margin = (
                ref_chosen - ref_rejected
            )

            # Move reference margin to policy device
            reference_margin_policy_device = (
                reference_margin.to(policy_device)
            )

            # Correct DPO margin
            dpo_margin = (
                policy_margin
                - reference_margin_policy_device
            )

            # Store results on CPU
            policy_margins.extend(
                policy_margin.detach()
                .cpu()
                .tolist()
            )

            reference_margins.extend(
                reference_margin.detach()
                .cpu()
                .tolist()
            )

            dpo_margins.extend(
                dpo_margin.detach()
                .cpu()
                .tolist()
            )

            # Free batch tensors
            del (
                chosen_batch,
                rejected_batch,
                policy_chosen_batch,
                policy_rejected_batch,
                reference_chosen_batch,
                reference_rejected_batch,
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                policy_margin,
                reference_margin,
                reference_margin_policy_device,
                dpo_margin,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (
                (start + batch_size) % 10 == 0
                or start + batch_size >= len(rows)
            ):
                print(
                    f"Preference evaluation: "
                    f"{min(start + batch_size, len(rows))}"
                    f"/{len(rows)}"
                )

    del gc

    policy_margins = torch.tensor(policy_margins)
    reference_margins = torch.tensor(reference_margins)
    dpo_margins = torch.tensor(dpo_margins)

    accuracy = (
        dpo_margins > 0
    ).float().mean().item()

    return {
        "preference_accuracy": accuracy,
        "mean_policy_margin": (
            policy_margins.mean().item()
        ),
        "mean_reference_margin": (
            reference_margins.mean().item()
        ),
        "mean_dpo_margin": (
            dpo_margins.mean().item()
        ),
    }

# Reward model scoring
def score_with_reward_model(
    reward_model,
    reward_tokenizer,
    prompt_messages,
    response,
    max_length):

    """
    Score a prompt + generated response using the
    course-provided UltraFeedback reward model.
    """

    messages = list(prompt_messages)

    messages.append({
            "role": "assistant",
            "content": response})

    text = reward_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False)

    encoded = reward_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length)

    device = next(
        reward_model.parameters()).device

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()}

    with torch.no_grad():

        output = reward_model(**encoded)

    return output.logits.squeeze().item()


# Generation
def generate_response(
    policy,
    tokenizer,
    prompt_messages,
    max_length,
    max_new_tokens,
    temperature,
    top_p):

    """
    Generate one response from the trained DPO policy.
    """

    device = next(
        policy.parameters()).device

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True)

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length)

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()}

    prompt_length = encoded["input_ids"].shape[1]

    with torch.no_grad():

        generated = policy.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)

    generated_tokens = generated[0, prompt_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True).strip()

    return response


# Response length
def response_length_tokens(
    tokenizer,
    response,
):
    return len(
        tokenizer(
            response,
            add_special_tokens=False)["input_ids"])


# Main evaluation
def run_evaluation(
    config_path: str,
    adapter: str,
    name: str):

    bundle = load_evaluation_bundle(
        config_path,
        adapter)

    cfg = bundle["cfg"]
    rows = bundle["rows"]

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]

    # --------------------------------------------------------
    # Load a separate frozen reference model.
    #
    # This is the original Qwen base model with no adapter.
    # --------------------------------------------------------

    reference = load_policy(
        cfg,
        trainable=False,
        fresh_lora=False)

    max_length = int(
        cfg["max_sequence_length"])

    max_generation_tokens = int(
        cfg["max_generation_tokens"])

    generation_cfg = cfg.get(
        "generation", {})

    temperature = float(
        generation_cfg.get(
            "temperature",
            0.7))

    top_p = float(
        generation_cfg.get(
            "top_p",
            0.9))

    output_dir = repo_path(
        cfg.get(
            "results_dir",
            "results/task1_dpo"))

    output_dir.mkdir(
        parents=True,
        exist_ok=True)

    print("=" * 70)
    print("STANDARD DPO EVALUATION")

    print(f"Evaluation examples: {len(rows)}")

    print(f"Adapter: {adapter}")

    print(
        f"Max generation tokens: "
        f"{max_generation_tokens}")

    print("=" * 70)


    # 1. Held-out preference accuracy
    print(
        "\n[1/4] Evaluating held-out preference accuracy..."
    )

    preference_metrics = (
        evaluate_preference_accuracy(
            policy=policy,
            reference=reference,
            tokenizer=tokenizer,
            rows=rows,
            max_length=max_length))

    print(
        f"Preference accuracy: "
        f"{preference_metrics['preference_accuracy']:.4f}")

    print(
        f"Mean policy margin: "
        f"{preference_metrics['mean_policy_margin']:.4f}")

    print(
        f"Mean reference margin: "
        f"{preference_metrics['mean_reference_margin']:.4f}")

    print(
        f"Mean DPO margin: "
        f"{preference_metrics['mean_dpo_margin']:.4f}")

    # ========================================================
    # 2. KL from reference
    #
    # Evaluate on generated responses.
    # ========================================================

    print("\n[2/4] Loading reward model...")

    reward_model, reward_tokenizer = load_reward_model(cfg)

    print("Reward model loaded.")

    print(
        "\n[2/4] Evaluating KL from reference policy...")

    generated_kl_values = []

    # Store generated responses and qualitative examples
    generated_records = []

    # Reward scores
    reward_scores = []

    # Response lengths
    response_lengths = []

    # Generate on every held-out prompt
    for idx, row in enumerate(rows):

        prompt = (
            prompt_messages_from_preference(row))

        response = generate_response(
            policy=policy,
            tokenizer=tokenizer,
            prompt_messages=prompt,
            max_length=max_length,
            max_new_tokens=max_generation_tokens,
            temperature=temperature,
            top_p=top_p
        )

        response_lengths.append(
            response_length_tokens(
                tokenizer,
                response))

        reward = score_with_reward_model(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            prompt_messages=prompt,
            response=response,
            max_length=max_length)

        reward_scores.append(reward)

        # Encode generated response for KL calculation
        encoded = encode_prompt_response(
            tokenizer,
            prompt,
            response,
            max_length)

        batch = pad_batch(
            tokenizer,
            [encoded])

        kl_value = token_level_kl(
            policy,
            reference,
            batch)

        generated_kl_values.append(kl_value)

        generated_records.append(
            {
                "index": idx,
                "prompt": prompt,
                "generated_response": response,
                "reward_score": reward,
                "response_length_tokens":
                    response_lengths[-1],
                "kl": kl_value})

        if ((idx + 1) % 10 == 0 or idx == 0):
            print(
                f"Evaluated "
                f"{idx + 1}/{len(rows)}"
            )

    # 3. Reward statistics
    reward_tensor = torch.tensor(
        reward_scores,
        dtype=torch.float32)

    mean_reward = (
        reward_tensor.mean().item()
        if len(reward_scores) > 0
        else 0.0)

    median_reward = (
        reward_tensor.median().item()
        if len(reward_scores) > 0
        else 0.0)


    # 4. KL statistics
    kl_tensor = torch.tensor(
        generated_kl_values,
        dtype=torch.float32)

    mean_kl = (
        kl_tensor.mean().item()
        if len(generated_kl_values) > 0
        else 0.0)

    median_kl = (
        kl_tensor.median().item()
        if len(generated_kl_values) > 0
        else 0.0)

    # 5. Response length statistics
    length_tensor = torch.tensor(
        response_lengths,
        dtype=torch.float32)

    mean_length = (
        length_tensor.mean().item()
        if len(response_lengths) > 0
        else 0.0)

    median_length = (
        length_tensor.median().item()
        if len(response_lengths) > 0
        else 0.0)

    min_length = (
        length_tensor.min().item()
        if len(response_lengths) > 0
        else 0.0)

    max_length_observed = (
        length_tensor.max().item()
        if len(response_lengths) > 0
        else 0.0)


    # Final metrics
    results = {
        "experiment": name,
        "adapter": adapter,
        "base_model": cfg["base_model"],
        "evaluation_dataset":
            cfg["paths"]["dpo_standard_eval"],
        "evaluation_examples": len(rows),

        "preference_accuracy":
            preference_metrics[
                "preference_accuracy"
            ],

        "mean_policy_margin":
            preference_metrics[
                "mean_policy_margin"
            ],

        "mean_reference_margin":
            preference_metrics[
                "mean_reference_margin"
            ],

        "mean_dpo_margin":
            preference_metrics[
                "mean_dpo_margin"
            ],

        "kl_from_reference_mean":
            mean_kl,

        "kl_from_reference_median":
            median_kl,

        "reward_model_score_mean":
            mean_reward,

        "reward_model_score_median":
            median_reward,

        "response_length_mean_tokens":
            mean_length,

        "response_length_median_tokens":
            median_length,

        "response_length_min_tokens":
            min_length,

        "response_length_max_tokens":
            max_length_observed,
    }

    # ========================================================
    # Print final table
    # ========================================================

    print()
    print("=" * 70)
    print("STANDARD DPO RESULTS")

    print(
        f"{'Metric':35s} {'Value':>15s}")

    print("-" * 52)

    print(
        f"{'Held-out preference accuracy':35s} "
        f"{results['preference_accuracy']:.4f}")

    print(
        f"{'Mean KL from reference':35s} "
        f"{results['kl_from_reference_mean']:.6f}")

    print(
        f"{'Median KL from reference':35s} "
        f"{results['kl_from_reference_median']:.6f}")

    print(
        f"{'Mean reward-model score':35s} "
        f"{results['reward_model_score_mean']:.4f}")

    print(
        f"{'Median reward-model score':35s} "
        f"{results['reward_model_score_median']:.4f}")

    print(
        f"{'Mean response length':35s} "
        f"{results['response_length_mean_tokens']:.2f}")

    print(
        f"{'Median response length':35s} "
        f"{results['response_length_median_tokens']:.2f}")

    print(
        f"{'Minimum response length':35s} "
        f"{results['response_length_min_tokens']:.0f}")

    print(
        f"{'Maximum response length':35s} "
        f"{results['response_length_max_tokens']:.0f}")

    print("=" * 70)

    # ========================================================
    # Save metrics
    # ========================================================

    results_path = (
        output_dir
        / f"{name}_metrics.json")

    with open(
        results_path, "w",
        encoding="utf-8") as f:json.dump(results, f, indent=2)


    # Save generated responses
    examples_path = (
        output_dir
        / f"{name}_generated.json")

    with open(
        examples_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            generated_records,
            f,
            indent=2,
            ensure_ascii=False)

    print(
        f"Metrics saved to: {results_path}")

    print(
        f"Generated examples saved to: "
        f"{examples_path}")


    # Qualitative examples
    print()
    print("=" * 70)
    print("QUALITATIVE EXAMPLES")

    for example in generated_records[:5]:

        print()
        print(
            f"Example {example['index']}")

        print(
            "-" * 50)

        prompt_text = ""

        for message in example[
            "prompt"]:
            if message.get("role") == "user":
                prompt_text = str(
                    message.get(
                        "content",
                        ""))

        print("PROMPT:")
        print(prompt_text[:1000])

        print()

        print("GENERATED RESPONSE:")

        print(
            example["generated_response"][:2000])

        print()

        print(
            f"Reward: "
            f"{example['reward_score']:.4f}")

        print(
            f"Length: "
            f"{example['response_length_tokens']} tokens")

        print(
            f"KL: "
            f"{example['kl']:.6f}")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")

    return results


# CLI
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")

    args = ap.parse_args()

    run_evaluation(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name)


if __name__ == "__main__":
    main()