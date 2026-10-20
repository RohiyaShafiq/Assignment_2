from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed

from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    reference_mode,
    token_values,
)

from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)

from task2_ppo.ppo import (
    compute_gae,
    shaped_rewards,
    ppo_policy_loss,
    value_mse_loss,
    normalize_advantages,
)

from common.metrics import (
    sampled_kl,
    sample_entropy,
    mean_response_length,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])

    # Load supplied PPO midpoint policy.
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )

    # Load supplied PPO midpoint value model.
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )

    # Frozen reward model.
    reward_model, reward_tokenizer = load_reward_model(cfg)

    # Fixed RL training prompt pool.
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # Policy optimizer.
    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )

    # Value optimizer.
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def _grad_norm(parameters):
    """
    Compute global gradient norm without modifying gradients.
    """
    params = [p for p in parameters if p.grad is not None]

    if not params:
        return 0.0

    total = torch.zeros(
        1,
        device=params[0].grad.device,
        dtype=torch.float32,
    )

    for p in params:
        grad = p.grad.detach().float()
        total += torch.sum(grad * grad)

    return float(torch.sqrt(total).item())


def _select_prompt_batch(prompt_rows, start, batch_size):
    """
    Deterministically select prompts from the supplied prompt pool.

    The assignment uses fixed prompt IDs and prompts_per_update=1.
    """
    n = len(prompt_rows)

    if n == 0:
        raise ValueError("The RL training prompt pool is empty.")

    selected = []

    for i in range(batch_size):
        idx = (start + i) % n
        selected.append(prompt_rows[idx])

    return selected


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):

    start_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


    bundle = prepare_ppo_continuation(config_path)

    cfg = bundle["cfg"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]

    if updates is not None:
        cfg["updates"] = int(updates)

    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)

    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)

    num_updates = int(cfg["updates"])
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))

    gamma = float(cfg.get("gamma", 1.0))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    clip_eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])

    value_coef = float(cfg.get("value_coef", 0.5))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    max_prompt_length = int(cfg["max_prompt_length"])
    max_response_length = int(cfg["max_response_length"])
    reward_max_length = int(cfg["reward_max_length"])

    missing_eos_penalty = float(
        cfg.get("missing_eos_penalty", 0.0)
    )

    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "metrics.jsonl"

    # Start a fresh metrics file for this run.
    if metrics_path.exists():
        metrics_path.unlink()

    print("=" * 70)
    print("PPO STANDARD CONTINUATION")
    print("=" * 70)
    print(f"Updates:             {num_updates}")
    print(f"Prompts/update:      {prompts_per_update}")
    print(f"PPO epochs:          {ppo_epochs}")
    print(f"Clip epsilon:        {clip_eps}")
    print(f"KL beta:             {beta_kl}")
    print(f"Gamma:               {gamma}")
    print(f"GAE lambda:          {gae_lambda}")
    print(f"Policy LR:            {cfg['policy_learning_rate']}")
    print(f"Value LoRA LR:        {cfg['value_lora_learning_rate']}")
    print(f"Value head LR:        {cfg['value_head_learning_rate']}")
    print("=" * 70)

    total_start = time.perf_counter()

    for update_idx in range(num_updates):

        update_start = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ------------------------------------------------------------
        # 1. Select fixed/deterministic prompt batch
        # ------------------------------------------------------------

        selected_rows = _select_prompt_batch(
            prompt_rows,
            update_idx * prompts_per_update,
            prompts_per_update,
        )

        prompts = [
            prompt_messages(row)
            for row in selected_rows
        ]

        # ------------------------------------------------------------
        # 2. Generate on-policy responses
        # ------------------------------------------------------------

        generation_cfg = cfg["generation"]

        rollout = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_response_length,
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            do_sample=bool(generation_cfg["do_sample"]),
        )

        sequences = rollout["sequences"]
        attention_mask = rollout["attention_mask"]
        response_ids = rollout["response_ids"]
        response_mask = rollout["response_mask"]

        responses = rollout["responses"]

        # ------------------------------------------------------------
        # 3. Old-policy log probabilities
        #
        # These are frozen after rollout and become pi_old.
        # ------------------------------------------------------------

        policy.train()

        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy,
                sequences,
                attention_mask,
                rollout["prompt_width"],
                response_ids,
            )

            old_logp = old_logp.detach()

        # ------------------------------------------------------------
        # 4. Reference-policy log probabilities
        #
        # Disable the PPO LoRA adapter.
        # This gives the frozen reference model.
        # ------------------------------------------------------------

        with torch.no_grad():
            with reference_mode(policy):

                ref_logp, _ = response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    rollout["prompt_width"],
                    response_ids,
                )

            ref_logp = ref_logp.detach()

        # ------------------------------------------------------------
        # 5. Learned reward
        # ------------------------------------------------------------

        with torch.no_grad():
            task_reward = score_reward_pairs(
                rm_model=reward_model,
                rm_tokenizer=reward_tokenizer,
                prompts=prompts,
                responses=responses,
                max_length=reward_max_length,
            )

        # ------------------------------------------------------------
        # 6. Penalize responses that do not terminate with EOS.
        # ------------------------------------------------------------

        if missing_eos_penalty > 0.0:
            terminated = torch.tensor(
                rollout["terminated_with_eos"],
                device=task_reward.device,
                dtype=torch.bool,
            )

            task_reward = task_reward.clone()

            task_reward[~terminated] -= missing_eos_penalty

        # ------------------------------------------------------------
        # 7. KL-shaped token rewards
        #
        # r_t =
        #   - beta * (log pi(a_t) - log pi_ref(a_t))
        #
        # plus learned reward on final valid response token.
        # ------------------------------------------------------------

        rewards = shaped_rewards(
            task_reward=task_reward,
            policy_logp=old_logp,
            ref_logp=ref_logp,
            response_mask=response_mask,
            beta_kl=beta_kl,
        )

        # ------------------------------------------------------------
        # 8. Critic values at response-token states
        # ------------------------------------------------------------

        with torch.no_grad():

            all_values = token_values(
                value_model,
                sequences,
                attention_mask,
            )

            prompt_width = int(rollout["prompt_width"])
            response_steps = response_ids.shape[1]

            # The value corresponding to response token t is the
            # state immediately before response token t.
            value_start = prompt_width - 1
            value_end = value_start + response_steps

            values = all_values[
                :,
                value_start:value_end,
            ]

            values = values.detach()

            print(
                "DEBUG values:",
                "finite=", torch.isfinite(values).all().item(),
                "min=", values.nan_to_num().min().item(),
                "max=", values.nan_to_num().max().item(),
                "mean=", values.nan_to_num().mean().item(),
            )

            print(
                "DEBUG rewards:",
                "finite=", torch.isfinite(rewards).all().item(),
                "min=", rewards.nan_to_num().min().item(),
                "max=", rewards.nan_to_num().max().item(),
            )

        if values.shape != rewards.shape:
            raise RuntimeError(
                f"Value/reward shape mismatch: "
                f"values={values.shape}, rewards={rewards.shape}"
            )

        # ------------------------------------------------------------
        # 9. Compute GAE and returns
        # ------------------------------------------------------------

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            mask=response_mask,
            gamma=gamma,
            lam=gae_lambda,
        )

        print(
            "DEBUG advantages:",
            "finite=", torch.isfinite(advantages).all().item(),
            "min=", advantages.nan_to_num().min().item(),
            "max=", advantages.nan_to_num().max().item(),
        )

        print(
            "DEBUG returns:",
            "finite=", torch.isfinite(returns).all().item(),
            "min=", returns.nan_to_num().min().item(),
            "max=", returns.nan_to_num().max().item(),
        )

        advantages = normalize_advantages(
            advantages,
            response_mask,
        ).detach()

        returns = returns.detach()

        # ------------------------------------------------------------
        # 10. PPO optimization
        # ------------------------------------------------------------

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_clip_fractions = []
        epoch_entropies = []

        epoch_policy_grad_norms = []
        epoch_value_grad_norms = []

        for epoch in range(ppo_epochs):

            # --------------------------------------------------------
            # Policy update
            # --------------------------------------------------------

            policy_optimizer.zero_grad(set_to_none=True)

            new_logp, _ = response_token_logprobs(
                policy,
                sequences,
                attention_mask,
                rollout["prompt_width"],
                response_ids,
            )

            policy_loss, ratio, clip_fraction = ppo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantage=advantages,
                mask=response_mask,
                eps=clip_eps,
            )

            policy_loss.backward()

            policy_grad_norm = _grad_norm(
                trainable_parameters(policy)
            )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters(policy),
                max_grad_norm,
            )

            policy_optimizer.step()

            # --------------------------------------------------------
            # Value / critic update
            # --------------------------------------------------------

            value_optimizer.zero_grad(set_to_none=True)

            current_values_all = token_values(
                value_model,
                sequences,
                attention_mask,
            )

            current_values = current_values_all[
                :,
                value_start:value_end,
            ]

            print(
                "DEBUG current_values:",
                "finite=", torch.isfinite(current_values).all().item(),
                "min=", current_values.nan_to_num().min().item(),
                "max=", current_values.nan_to_num().max().item(),
            )

            value_loss = value_mse_loss(
                predicted_values=current_values,
                returns=returns,
                mask=response_mask,
            )

            value_loss.backward()

            value_grad_norm = _grad_norm(
                value_model.parameters()
            )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters(value_model),
                max_grad_norm,
            )

            value_optimizer.step()

            # --------------------------------------------------------
            # Diagnostics
            # --------------------------------------------------------

            with torch.no_grad():
                entropy = sample_entropy(
                    new_logp.detach(),
                    response_mask,
                )

            epoch_policy_losses.append(
                float(policy_loss.detach().item())
            )

            epoch_value_losses.append(
                float(value_loss.detach().item())
            )

            epoch_clip_fractions.append(
                float(clip_fraction.detach().item())
            )

            epoch_entropies.append(
                float(entropy.detach().item())
            )

            epoch_policy_grad_norms.append(
                policy_grad_norm
            )

            epoch_value_grad_norms.append(
                value_grad_norm
            )

        # ------------------------------------------------------------
        # 11. Rollout-level diagnostics
        # ------------------------------------------------------------

        with torch.no_grad():

            mean_reward = float(
                task_reward.float().mean().item()
            )

            mean_kl = float(
                sampled_kl(
                    old_logp,
                    ref_logp,
                    response_mask,
                ).item()
            )

            response_length = mean_response_length(
                response_mask
            )

        update_time = time.perf_counter() - update_start
        total_time = time.perf_counter() - total_start

        if torch.cuda.is_available():
            peak_vram_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )
        else:
            peak_vram_mb = 0.0

        mean_policy_loss = sum(epoch_policy_losses) / len(
            epoch_policy_losses
        )

        mean_value_loss = sum(epoch_value_losses) / len(
            epoch_value_losses
        )

        mean_clip_fraction = sum(epoch_clip_fractions) / len(
            epoch_clip_fractions
        )

        mean_entropy = sum(epoch_entropies) / len(
            epoch_entropies
        )

        mean_policy_grad_norm = (
            sum(epoch_policy_grad_norms)
            / len(epoch_policy_grad_norms)
        )

        mean_value_grad_norm = (
            sum(epoch_value_grad_norms)
            / len(epoch_value_grad_norms)
        )

        # Combined gradient norm diagnostic.
        gradient_norm = (
            mean_policy_grad_norm ** 2
            + mean_value_grad_norm ** 2
        ) ** 0.5

        # ------------------------------------------------------------
        # 12. Machine-readable metrics
        # ------------------------------------------------------------

        record = {
            "run_name": run_name,
            "update": update_idx + 1,

            "learned_reward": mean_reward,
            "kl_from_reference": mean_kl,

            "policy_loss": mean_policy_loss,
            "value_loss": mean_value_loss,

            "entropy": mean_entropy,

            "gradient_norm": gradient_norm,
            "policy_gradient_norm": mean_policy_grad_norm,
            "value_gradient_norm": mean_value_grad_norm,

            "clip_fraction": mean_clip_fraction,

            "response_length": response_length,

            "ppo_epochs": ppo_epochs,
            "clip_epsilon": clip_eps,
            "kl_beta": beta_kl,

            "terminated_fraction": float(
                sum(rollout["terminated_with_eos"])
                / len(rollout["terminated_with_eos"])
            ),

            "truncated_fraction": float(
                sum(rollout["truncated"])
                / len(rollout["truncated"])
            ),

            "update_time_seconds": update_time,
            "elapsed_time_seconds": total_time,

            "peak_vram_mb": peak_vram_mb,
        }

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(record)
                + "\n"
            )

        # ------------------------------------------------------------
        # 13. Print update summary
        # ------------------------------------------------------------

        print(
            f"[Update {update_idx + 1:02d}/{num_updates}] "
            f"reward={mean_reward:.4f} | "
            f"KL={mean_kl:.4f} | "
            f"policy_loss={mean_policy_loss:.4f} | "
            f"value_loss={mean_value_loss:.4f} | "
            f"entropy={mean_entropy:.4f} | "
            f"clip={mean_clip_fraction:.4f} | "
            f"length={response_length:.1f} | "
            f"grad={gradient_norm:.4f} | "
            f"time={update_time:.1f}s"
        )

    wall_clock_seconds = time.perf_counter() - start_time

    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_vram_gb = 0.0

    print("=" * 70)
    print("CONTINUATION RESOURCE USAGE")
    print("=" * 70)
    print(f"Peak VRAM:       {peak_vram_gb:.2f} GB")
    print(f"Wall-clock time: {wall_clock_seconds:.2f} seconds")
    print(f"Wall-clock time: {wall_clock_seconds / 60:.2f} minutes")

    # ------------------------------------------------------------
    # 14. Save final policy
    # ------------------------------------------------------------

    policy_dir = out / "policy"

    print("\nSaving final PPO policy...")
    policy.save_pretrained(policy_dir)
    tokenizer.save_pretrained(policy_dir)

    # ------------------------------------------------------------
    # 15. Save final value model
    # ------------------------------------------------------------

    value_dir = out / "value_model"

    print("Saving final PPO value model...")
    value_model.save_pretrained(value_dir)

    # ------------------------------------------------------------
    # 16. Save run configuration
    # ------------------------------------------------------------

    summary = {
        "run_name": run_name,
        "updates": num_updates,
        "prompts_per_update": prompts_per_update,
        "ppo_epochs": ppo_epochs,

        "clip_epsilon": clip_eps,
        "kl_beta": beta_kl,

        "gamma": gamma,
        "gae_lambda": gae_lambda,

        "policy_learning_rate": float(
            cfg["policy_learning_rate"]
        ),

        "value_lora_learning_rate": float(
            cfg["value_lora_learning_rate"]
        ),

        "value_head_learning_rate": float(
            cfg["value_head_learning_rate"]
        ),

        "value_coef": value_coef,

        "seed": int(cfg["seed"]),

        "policy_checkpoint": cfg["paths"][
            "ppo_midpoint_policy"
        ],

        "value_checkpoint": cfg["paths"][
            "ppo_midpoint_value"
        ],

        "final_policy": str(policy_dir),
        "final_value_model": str(value_dir),

        "metrics_file": str(metrics_path),

        "total_time_seconds": (
            time.perf_counter() - total_start
        ),
    }

    with (out / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("PPO CONTINUATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {out}")
    print(f"Metrics:          {metrics_path}")
    print(f"Policy:           {policy_dir}")
    print(f"Value model:      {value_dir}")
    print("=" * 70)

    return {
        "output": str(out),
        "metrics": str(metrics_path),
        "policy": str(policy_dir),
        "value_model": str(value_dir),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    ap.add_argument(
        "--output",
    )

    ap.add_argument(
        "--updates",
        type=int,
    )

    ap.add_argument(
        "--clip-epsilon",
        type=float,
    )

    ap.add_argument(
        "--kl-beta",
        type=float,
    )

    ap.add_argument(
        "--run-name",
        default="standard",
    )

    args = ap.parse_args()
    
    
    run_ppo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()