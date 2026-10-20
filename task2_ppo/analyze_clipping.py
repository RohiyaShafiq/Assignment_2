from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.data import load_yaml, repo_path
from common.models import load_policy
from common.generation import response_token_logprobs
from task2_ppo.ppo import compute_gae, normalize_advantages


def load_cached_rollouts(path):
    rows = torch.load(
        repo_path(path),
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []

    for row in rows:
        row = dict(row)

        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]

        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]

        normalized.append(row)

    required = {
        "source_index",
        "response",
        "response_tokens",
        "old_logprobs",
        "ref_logprobs",
        "values",
        "effective_terminal_reward",
    }

    if not required.issubset(normalized[0]):
        raise ValueError(
            f"Unexpected PPO cache schema; need at least {sorted(required)}"
        )

    return normalized


def load_prompt_pool():
    path = repo_path("data/rl_prompt_pool_train.jsonl")

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def tokenize_cached_batch(rows, prompt_pool, tokenizer, max_prompt_length):
    """
    Reconstruct the exact prompt + cached response sequence.

    The cache stores:
      - response: decoded response text
      - response_tokens: response length
      - source_index: index into rl_prompt_pool_train.jsonl

    We tokenize prompt and response separately so that the response
    token positions are explicitly known.
    """

    prompt_ids_list = []
    response_ids_list = []

    for row in rows:
        source_index = int(row["source_index"])

        record = prompt_pool[source_index]

        prompt_text = record["prompt"]
        response_text = row["response"]

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )["input_ids"][0]

        response_ids = tokenizer(
            row["response"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if row["terminated_with_eos"]:
            response_ids = torch.cat([
                response_ids,
                torch.tensor([tokenizer.eos_token_id], dtype=torch.long),
            ])

        expected_len = int(row["response_tokens"])

        if len(response_ids) != expected_len:
            # The cache's response text is decoded text, so exact token
            # reconstruction can differ in edge cases. Keep the actual
            # reconstructed tokens, but report the mismatch.
            print(
                f"WARNING: source_index={source_index}: "
                f"cached response length={expected_len}, "
                f"re-tokenized length={len(response_ids)}"
            )

        prompt_ids_list.append(prompt_ids)
        response_ids_list.append(response_ids)

    return prompt_ids_list, response_ids_list


def make_batch(prompt_ids_list, response_ids_list, tokenizer):
    """
    Construct padded [prompt + response] sequences.

    response positions are aligned to the same fixed prompt width,
    matching common.generation.response_token_logprobs().
    """

    batch_size = len(prompt_ids_list)

    prompt_width = max(len(x) for x in prompt_ids_list)
    response_width = max(len(x) for x in response_ids_list)

    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    sequences = torch.full(
        (batch_size, prompt_width + response_width),
        pad_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros_like(sequences)

    response_ids = torch.full(
        (batch_size, response_width),
        pad_id,
        dtype=torch.long,
    )

    response_mask = torch.zeros(
        (batch_size, response_width),
        dtype=torch.float32,
    )

    for i, (p, r) in enumerate(
        zip(prompt_ids_list, response_ids_list)
    ):
        p_len = len(p)
        r_len = len(r)

        sequences[i, :p_len] = p
        sequences[i, prompt_width:prompt_width + r_len] = r

        attention_mask[i, :p_len] = 1
        attention_mask[
            i,
            prompt_width:prompt_width + r_len
        ] = 1

        response_ids[i, :r_len] = r
        response_mask[i, :r_len] = 1.0

    return {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "prompt_width": prompt_width,
        "response_ids": response_ids,
        "response_mask": response_mask,
    }


def pad_tensor(rows, key, width):
    x = torch.zeros(
        len(rows),
        width,
        dtype=torch.float32,
    )

    for i, row in enumerate(rows):
        values = row[key].detach().float().cpu()
        n = min(values.numel(), width)
        x[i, :n] = values[:n]

    return x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    parser.add_argument(
        "--output",
        default="results/task2_ppo/clipping_analysis.json",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # ---------------------------------------------------------
    # Load cache
    # ---------------------------------------------------------

    rows = load_cached_rollouts(
        cfg["cached_rollouts"]
    )

    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])

    # ---------------------------------------------------------
    # Load prompt pool
    # ---------------------------------------------------------

    prompt_pool = load_prompt_pool()

    print("Prompt pool:", len(prompt_pool))

    # ---------------------------------------------------------
    # Load midpoint policy
    # ---------------------------------------------------------

    print("\nLoading midpoint policy...")

    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=False,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
    )

    device = next(policy.parameters()).device

    # ---------------------------------------------------------
    # Reconstruct prompt + cached response
    # ---------------------------------------------------------

    prompt_ids_list, response_ids_list = tokenize_cached_batch(
        rows,
        prompt_pool,
        tokenizer,
        int(cfg["max_prompt_length"]),
    )

    batch = make_batch(
        prompt_ids_list,
        response_ids_list,
        tokenizer,
    )

    batch = {
        k: (
            v.to(device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in batch.items()
    }

    # ---------------------------------------------------------
    # Re-evaluate cached responses under midpoint policy
    # ---------------------------------------------------------

    print("\nComputing midpoint-policy log probabilities...")

    policy.eval()

    with torch.no_grad():
        # Evaluate the cached responses in small batches to avoid GPU OOM.
    # Evaluate the cached responses in small batches to avoid GPU OOM.
      logprob_chunks = []
      chunk_size = 4

      for start in range(0, len(prompt_ids_list), chunk_size):
          end = min(start + chunk_size, len(prompt_ids_list))

          chunk = make_batch(
              prompt_ids_list[start:end],
              response_ids_list[start:end],
              tokenizer,
          )

          chunk_sequences = chunk["sequences"].to(device)
          chunk_attention = chunk["attention_mask"].to(device)
          chunk_response_ids = chunk["response_ids"].to(device)

          with torch.no_grad():
              chunk_logprobs, _ = response_token_logprobs(
                  policy,
                  chunk_sequences,
                  chunk_attention,
                  chunk["prompt_width"],
                  chunk_response_ids,
              )

          logprob_chunks.append(chunk_logprobs.cpu())

          del chunk_sequences
          del chunk_attention
          del chunk_response_ids
          del chunk_logprobs

          if torch.cuda.is_available():
              torch.cuda.empty_cache()

      max_logprob_len = max(x.shape[1] for x in logprob_chunks)

      padded_chunks = []
      for x in logprob_chunks:
          if x.shape[1] < max_logprob_len:
              x = torch.nn.functional.pad(
                  x,
                  (0, max_logprob_len - x.shape[1]),
                  value=0.0,
              )
          padded_chunks.append(x)

      new_logprobs = torch.cat(padded_chunks, dim=0)

    new_logprobs = new_logprobs.float().cpu()

    response_mask = batch["response_mask"].float().cpu()

    print(
        "New logprob shape:",
        tuple(new_logprobs.shape),
    )

    # ---------------------------------------------------------
    # Cached PPO quantities
    # ---------------------------------------------------------

    width = new_logprobs.shape[1]

    old_logprobs = pad_tensor(
        rows,
        "old_logprobs",
        width,
    )

    ref_logprobs = pad_tensor(
        rows,
        "ref_logprobs",
        width,
    )

    values = pad_tensor(
        rows,
        "values",
        width,
    )

    terminal_rewards = torch.tensor(
        [
            float(row["effective_terminal_reward"])
            for row in rows
        ],
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # Construct KL-shaped rewards
    # ---------------------------------------------------------

    rewards = (
        -float(cfg["kl_beta"])
        * (old_logprobs - ref_logprobs)
        * response_mask
    )

    for i, row in enumerate(rows):
        n = min(
            int(row["response_tokens"]),
            width,
        )

        if n > 0:
            rewards[i, n - 1] += terminal_rewards[i]

    # ---------------------------------------------------------
    # GAE / returns
    # ---------------------------------------------------------

    advantages, returns = compute_gae(
        rewards,
        values,
        response_mask,
        gamma=float(cfg["gamma"]),
        lam=float(cfg["gae_lambda"]),
    )

    advantages = normalize_advantages(
        advantages,
        response_mask,
    ).float()

    # ---------------------------------------------------------
    # PPO clipping analysis
    # ---------------------------------------------------------

    log_ratio = new_logprobs - old_logprobs

    # Numerical safety.
    log_ratio = torch.clamp(
        log_ratio,
        min=-20.0,
        max=20.0,
    )

    ratio = torch.exp(log_ratio)

    results = {}

    print("\n========== PPO CLIPPING ANALYSIS ==========")

    for epsilon in cfg["clip_values"]:

        epsilon = float(epsilon)

        surr1 = ratio * advantages

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - epsilon,
            1.0 + epsilon,
        )

        surr2 = clipped_ratio * advantages

        clipped_objective = torch.minimum(
            surr1,
            surr2,
        )

        valid = response_mask.bool()

        mean_surrogate = (
            clipped_objective[valid].mean().item()
        )

        affected = (
            (ratio < 1.0 - epsilon)
            |
            (ratio > 1.0 + epsilon)
        )

        affected_fraction = (
            affected[valid].float().mean().item()
        )

        raw_surrogate = (
            surr1[valid].mean().item()
        )

        results[str(epsilon)] = {
            "epsilon": epsilon,
            "cached_clipped_surrogate": mean_surrogate,
            "unclipped_surrogate": raw_surrogate,
            "affected_token_fraction": affected_fraction,
        }

        print(
            f"\nepsilon = {epsilon:.2f}"
        )

        print(
            f"  unclipped surrogate       : "
            f"{raw_surrogate:.6f}"
        )

        print(
            f"  clipped surrogate         : "
            f"{mean_surrogate:.6f}"
        )

        print(
            f"  affected-token fraction   : "
            f"{affected_fraction:.6f}"
        )

    # ---------------------------------------------------------
    # Additional diagnostics
    # ---------------------------------------------------------

    valid = response_mask.bool()

    ratio_valid = ratio[valid]

    diagnostics = {
        "ratio_mean": ratio_valid.mean().item(),
        "ratio_std": ratio_valid.std(unbiased=False).item(),
        "ratio_min": ratio_valid.min().item(),
        "ratio_max": ratio_valid.max().item(),
        "mean_advantage": advantages[valid].mean().item(),
        "std_advantage": advantages[valid].std(unbiased=False).item(),
        "num_rollouts": len(rows),
        "num_response_tokens": int(valid.sum().item()),
    }

    print("\n========== DIAGNOSTICS ==========")

    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path = repo_path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "epsilons": results,
        "diagnostics": diagnostics,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nSaved clipping analysis to: {output_path}"
    )


if __name__ == "__main__":
    main()