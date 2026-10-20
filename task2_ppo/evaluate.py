
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from common.data import load_yaml, prompt_messages, read_jsonl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid response tokens only."""
    values = values.float()
    mask = mask.to(device=values.device, dtype=torch.float32)

    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)

    mask = mask.expand_as(values)
    denominator = mask.sum().clamp_min(1.0)

    return (values * mask).sum() / denominator


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy for each response token."""
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def pad_token_rows(
    rows: list[torch.Tensor],
    max_length: int,
) -> torch.Tensor:
    """Pad token rows to a common length."""
    result = torch.zeros((len(rows), max_length), dtype=torch.float32)

    for i, row in enumerate(rows):
        length = min(row.numel(), max_length)
        if length:
            result[i, :length] = row[:length].float()

    return result


def evaluate_policy(
    config_path: str,
    adapter: str,
    name: str = "standard",
    output: str | None = None,
    chunk_size: int = 1,
    reward_chunk_size: int = 1,
) -> dict[str, Any]:
    """Evaluate a PPO adapter on the held-out prompt set."""

    if chunk_size < 1 or reward_chunk_size < 1:
        raise ValueError("Chunk sizes must be at least 1.")

    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])

    if not rows:
        raise ValueError(
            f"No evaluation prompts found at "
            f"{cfg['paths']['rl_prompt_eval']}"
        )

    tokenizer = load_tokenizer(cfg["base_model"])

    policy = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )
    policy.eval()

    reward_model, reward_tokenizer = load_reward_model(cfg)
    reward_model.eval()

    prompts = [prompt_messages(row) for row in rows]
    generation_cfg = cfg["generation"]

    rollout = batch_generate(
        model=policy,
        tokenizer=tokenizer,
        prompts=prompts,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["eval_max_response_length"]),
        temperature=float(generation_cfg["temperature"]),
        top_p=float(generation_cfg["top_p"]),
        do_sample=bool(generation_cfg["do_sample"]),
    )

    sequences = rollout["sequences"]
    attention_mask = rollout["attention_mask"]
    response_ids = rollout["response_ids"]
    response_mask = rollout["response_mask"]
    responses = rollout["responses"]

    # Calculate policy log probabilities and entropy in chunks.
    policy_logp_rows: list[torch.Tensor] = []
    entropy_rows: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(rows), chunk_size):
            end = min(start + chunk_size, len(rows))

            policy_logp, logits = response_token_logprobs(
                policy,
                sequences[start:end],
                attention_mask[start:end],
                rollout["prompt_width"],
                response_ids[start:end],
            )

            if logits is None:
                raise RuntimeError(
                    "response_token_logprobs returned logits=None. "
                    "Check common/generation.py."
                )

            entropy = token_entropy(logits)

            policy_logp_rows.extend(
                row.detach().float().cpu()
                for row in policy_logp
            )

            entropy_rows.extend(
                row.detach().float().cpu()
                for row in entropy
            )

            del policy_logp, logits, entropy

    # Calculate reference-policy log probabilities with the adapter disabled.
    reference_logp_rows: list[torch.Tensor] = []

    with reference_mode(policy):
        with torch.inference_mode():
            for start in range(0, len(rows), chunk_size):
                end = min(start + chunk_size, len(rows))

                ref_logp, _ = response_token_logprobs(
                    policy,
                    sequences[start:end],
                    attention_mask[start:end],
                    rollout["prompt_width"],
                    response_ids[start:end],
                )

                reference_logp_rows.extend(
                    row.detach().float().cpu()
                    for row in ref_logp
                )

                del ref_logp

    # Align token rows and exclude padding from all metrics.
    max_length = response_ids.shape[1]

    policy_logp = pad_token_rows(policy_logp_rows, max_length)
    reference_logp = pad_token_rows(reference_logp_rows, max_length)
    entropy_values = pad_token_rows(entropy_rows, max_length)
    mask = response_mask.detach().float().cpu()

    kl_tokens = policy_logp - reference_logp
    heldout_kl = masked_mean(kl_tokens, mask).item()
    heldout_entropy = masked_mean(entropy_values, mask).item()
    heldout_response_length = mask.sum(dim=1).mean().item()

    # Score responses in chunks to reduce reward-model memory usage.
    reward_scores: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(rows), reward_chunk_size):
            end = min(start + reward_chunk_size, len(rows))

            chunk_scores = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts[start:end],
                responses[start:end],
                max_length=int(cfg.get("reward_max_length", 1280)),
            )

            reward_scores.extend(
                float(score)
                for score in chunk_scores.detach().float().cpu().tolist()
            )

            del chunk_scores

    heldout_reward = (
        sum(reward_scores) / len(reward_scores)
        if reward_scores
        else 0.0
    )

    examples = []

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        examples.append(
            {
                "index": i,
                "prompt": prompt,
                "response": response,
                "reward": reward_scores[i],
                "response_length": int(rollout["response_lengths"][i]),
                "terminated_with_eos": bool(
                    rollout["terminated_with_eos"][i]
                ),
                "truncated": bool(rollout["truncated"][i]),
            }
        )

    result: dict[str, Any] = {
        "name": name,
        "adapter": str(adapter),
        "num_examples": len(rows),
        "heldout_reward": heldout_reward,
        "heldout_kl": heldout_kl,
        "heldout_entropy": heldout_entropy,
        "heldout_response_length": heldout_response_length,
        "examples": examples,
    }

    print(f"\nEvaluation: {name}")
    print(f"Number of examples: {len(rows)}")
    print(f"Held-out reward: {heldout_reward:.6f}")
    print(f"Held-out sampled KL: {heldout_kl:.6f}")
    print(f"Held-out entropy: {heldout_entropy:.6f}")
    print(f"Mean response length: {heldout_response_length:.2f}")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)

        print(f"Saved evaluation results to: {output_path}")

    del policy, reward_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO policy on held-out prompts."
    )

    parser.add_argument(
        "--config",
        default="configs/ppo.yaml",
        help="Path to PPO configuration YAML.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the trained policy adapter.",
    )
    parser.add_argument(
        "--name",
        default="standard",
        help="Name for this evaluation run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON metrics and examples.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Policy/reference evaluation batch size.",
    )
    parser.add_argument(
        "--reward-chunk-size",
        type=int,
        default=1,
        help="Reward-model scoring batch size.",
    )

    args = parser.parse_args()

    evaluate_policy(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name,
        output=args.output,
        chunk_size=args.chunk_size,
        reward_chunk_size=args.reward_chunk_size,
    )


if __name__ == "__main__":
    main()
