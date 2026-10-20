from __future__ import annotations
import argparse, json, random, time
import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare(config_path):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    rm, rmtok = load_reward_model(cfg)
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    if not rows:
        raise ValueError("No training prompts found.")
    opt = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return cfg, tok, policy, rm, rmtok, rows, opt


def length_gradient_stats(advantages, token_mask, lengths, loss_type, max_length):
    """
    Absolute policy-term derivative w.r.t. each sampled token log-probability
    at ratio=1. Canonical GRPO divides by realized valid length; Dr. GRPO
    divides by the fixed completion budget. This excludes the KL term and is
    not the full model-parameter gradient norm.
    """
    valid_len = token_mask.sum(-1)
    usable = valid_len > 0
    denom = valid_len.clamp_min(1.0) if loss_type == "grpo" else torch.full_like(
        valid_len, float(max_length)
    )
    per_token = advantages.abs() / denom.clamp_min(1.0)
    per_sequence = per_token * valid_len
    short = usable & (lengths <= max(1, max_length // 2))
    long = usable & (lengths > max(1, max_length // 2))

    def mean(x, mask):
        return float(x[mask].mean().detach().cpu()) if bool(mask.any()) else float("nan")

    return {
        "short_n": int(short.sum().item()),
        "long_n": int(long.sum().item()),
        "short_mean_abs_policy_grad_per_token": mean(per_token, short),
        "long_mean_abs_policy_grad_per_token": mean(per_token, long),
        "short_mean_abs_policy_grad_per_sequence": mean(per_sequence, short),
        "long_mean_abs_policy_grad_per_sequence": mean(per_sequence, long),
        "short_mean_abs_advantage": mean(advantages.abs(), short),
        "long_mean_abs_advantage": mean(advantages.abs(), long),
    }


@torch.inference_mode()
def evaluate_heldout(policy, tok, rm, rmtok, rows, cfg, run_name, results_dir, max_eval=None):
    if not rows:
        raise ValueError("Held-out evaluation set is empty.")
    if max_eval:
        rows = rows[:max_eval]
    prompts = [prompt_messages(row) for row in rows]
    gen = cfg.get("generation", {})
    max_prompt = int(cfg["max_prompt_length"])
    max_new = int(cfg["max_completion_length"])
    batch_size = max(1, int(cfg.get("eval_batch_size", 4)))
    set_seed(int(cfg["seed"]) + 9001)

    rewards_all, lengths_all, kls_all, truncated_all, examples = [], [], [], [], []
    policy.eval()
    for start in range(0, len(prompts), batch_size):
        pb = prompts[start:start + batch_size]
        ro = batch_generate(
            model=policy, tokenizer=tok, prompts=pb,
            max_prompt_length=max_prompt, max_new_tokens=max_new,
            temperature=float(gen.get("temperature", 0.7)),
            top_p=float(gen.get("top_p", 0.9)),
            do_sample=bool(gen.get("do_sample", True)),
        )
        ids, mask = ro["response_ids"], ro["response_mask"].float()
        rewards = score_reward_pairs(rm, rmtok, pb, ro["responses"]).float().cpu()
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(
                policy, ro["sequences"], ro["attention_mask"], ro["prompt_width"], ids
            )
        pol_logp, _ = response_token_logprobs(
            policy, ro["sequences"], ro["attention_mask"], ro["prompt_width"], ids
        )
        lr = ref_logp - pol_logp
        per_tok_kl = torch.exp(lr) - lr - 1.0
        md = mask.to(per_tok_kl.device)
        seq_kl = ((per_tok_kl * md).sum(-1) / md.sum(-1).clamp_min(1.0)).float().cpu()
        lengths = mask.sum(-1).float().cpu()

        rewards_all.extend(rewards.tolist())
        lengths_all.extend(lengths.tolist())
        kls_all.extend(seq_kl.tolist())
        truncated_all.extend([bool(x) for x in ro["truncated"]])
        for j, response in enumerate(ro["responses"]):
            if len(examples) < 12:
                examples.append({
                    "prompt_index": start + j,
                    "prompt": pb[j],
                    "response": response,
                    "reward": float(rewards[j]),
                    "response_length_tokens": int(lengths[j]),
                    "truncated": bool(ro["truncated"][j]),
                })
        del ro, ids, mask, ref_logp, pol_logp, lr, per_tok_kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "run_name": run_name,
        "heldout_prompt_count": len(prompts),
        "heldout_mean_reward": sum(rewards_all) / max(1, len(rewards_all)),
        "heldout_mean_sampled_kl": sum(kls_all) / max(1, len(kls_all)),
        "heldout_mean_response_length_tokens": sum(lengths_all) / max(1, len(lengths_all)),
        "heldout_truncated_fraction": sum(truncated_all) / max(1, len(truncated_all)),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run_name}_heldout_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (results_dir / f"{run_name}_heldout_examples.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(
        f"Held-out reward={summary['heldout_mean_reward']:.4f} | "
        f"KL={summary['heldout_mean_sampled_kl']:.6f} | "
        f"length={summary['heldout_mean_response_length_tokens']:.1f} | "
        f"n={summary['heldout_prompt_count']}"
    )
    return summary


def run_grpo(config_path, output=None, updates=None, loss_type="grpo",
             run_name="standard", max_eval_prompts=None):
    if loss_type not in ("grpo", "dr_grpo"):
        raise ValueError(f"Unsupported loss type: {loss_type}")
    cfg, tok, policy, rm, rmtok, train_rows, optimizer = prepare(config_path)
    if updates is not None:
        cfg["updates"] = int(updates)
    n_updates = int(cfg["updates"])
    k = int(cfg["num_generations"])
    ppu = int(cfg.get("prompts_per_update", 1))
    max_prompt = int(cfg["max_prompt_length"])
    max_completion = int(cfg["max_completion_length"])
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    clip_eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    if n_updates < 1 or k < 2 or ppu < 1:
        raise ValueError("updates and prompts_per_update must be positive; K must be >= 2.")

    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{run_name}.jsonl"
    gen = cfg.get("generation", {})
    temperature, top_p = float(gen.get("temperature", 0.7)), float(gen.get("top_p", 0.9))
    do_sample = bool(gen.get("do_sample", True))
    device = next(policy.parameters()).device
    logs = []

    print(f"Run={run_name} | loss={loss_type} | midpoint={cfg['paths']['grpo_midpoint_policy']}")
    for step in range(1, n_updates + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        selected = random.choices(train_rows, k=ppu)
        prompt_groups = [prompt_messages(r) for r in selected]
        repeated = [p for group in prompt_groups for p in [group] * k]
        ro = batch_generate(
            model=policy, tokenizer=tok, prompts=repeated,
            max_prompt_length=max_prompt, max_new_tokens=max_completion,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
        )
        response_ids = ro["response_ids"]
        response_mask = ro["response_mask"].float()
        truncated = ro["truncated"]
        token_mask = mask_truncated_sequences(response_mask, truncated) if mask_truncated else response_mask
        valid_lengths = token_mask.sum(-1).float()

        with torch.no_grad():
            rewards = score_reward_pairs(rm, rmtok, repeated, ro["responses"]).to(device).float()
        group_ids = torch.arange(ppu, device=device).repeat_interleave(k)
        advantages = group_relative_advantages(rewards, group_ids).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, ro["sequences"], ro["attention_mask"], ro["prompt_width"], response_ids
                )
        policy.eval()
        new_logp, _ = response_token_logprobs(
            policy, ro["sequences"], ro["attention_mask"], ro["prompt_width"], response_ids
        )
        old_logp = new_logp.detach()
        loss, stats = grpo_policy_loss(
            new_logp=new_logp, old_logp=old_logp, seq_adv=advantages,
            token_mask=token_mask, ref_logp=ref_logp, eps=clip_eps, beta=beta,
            loss_type=loss_type,
            max_completion_length=max_completion if loss_type == "dr_grpo" else None,
        )
        grad_stats = length_gradient_stats(
            advantages.detach(), token_mask.detach(), valid_lengths.detach(),
            loss_type, max_completion,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
        grad_norm_value = float(grad_norm.detach().float().cpu())
        optimizer.step()
        policy.train()

        group_stds = torch.stack([
            rewards[group_ids == gid].std(unbiased=False) for gid in torch.unique(group_ids)
        ])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_vram = 0.0
        elapsed = time.perf_counter() - start
        row = {
            "run_name": run_name, "update": step, "loss_type": loss_type,
            "mean_reward": float(rewards.mean().detach().cpu()),
            "mean_group_reward_std": float(group_stds.mean().detach().cpu()),
            "frac_uninformative_groups": float((group_stds <= 1e-8).float().mean().cpu()),
            "policy_loss": float(stats["policy_term"].cpu()),
            "total_loss": float(loss.detach().cpu()),
            "kl": float(stats["sampled_kl"].cpu()),
            "clip_fraction": float(stats["clip_fraction"].cpu()),
            "ratio_mean": float(stats["ratio_mean"].cpu()),
            "sample_entropy": float(stats["sample_entropy"].cpu()),
            "mean_response_length": float(torch.tensor(ro["response_lengths"]).float().mean()),
            "mean_valid_policy_length": float(valid_lengths.mean().detach().cpu()),
            "truncated_fraction": float(torch.tensor(truncated).float().mean()),
            "gradient_norm_preclip": grad_norm_value,
            "peak_vram_gb": peak_vram, "wall_clock_seconds": elapsed,
            **grad_stats,
        }
        logs.append(row)
        write_jsonl(metrics_path, logs)
        print(
            f"{step:02d}/{n_updates} reward={row['mean_reward']:.4f} "
            f"KL={row['kl']:.6f} length={row['mean_response_length']:.1f} "
            f"policy-grad/token short={row['short_mean_abs_policy_grad_per_token']:.6g} "
            f"long={row['long_mean_abs_policy_grad_per_token']:.6g} "
            f"grad_norm={grad_norm_value:.4f}"
        )
        del ro, response_ids, response_mask, token_mask, valid_lengths
        del rewards, group_ids, advantages, ref_logp, new_logp, old_logp, loss, stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    metadata = {
        "run_name": run_name, "loss_type": loss_type, "updates_completed": len(logs),
        "num_generations": k, "prompts_per_update": ppu,
        "max_prompt_length": max_prompt, "max_completion_length": max_completion,
        "mask_truncated_completions": mask_truncated,
        "learning_rate": float(cfg["learning_rate"]), "clip_epsilon": clip_eps,
        "kl_beta": beta, "seed": int(cfg["seed"]),
        "starting_adapter": str(repo_path(cfg["paths"]["grpo_midpoint_policy"])),
        "output_adapter": str(out), "metrics_file": str(metrics_path),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved adapter: {out}\nSaved metrics: {metrics_path}")

    eval_path = cfg.get("paths", {}).get("rl_prompt_eval")
    if eval_path:
        eval_rows = read_jsonl(eval_path)
        evaluate_heldout(
            policy, tok, rm, rmtok, eval_rows, cfg, run_name, results_dir, max_eval_prompts
        )
    else:
        print("WARNING: paths.rl_prompt_eval is missing; held-out evaluation skipped.")
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--output")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    parser.add_argument("--run-name", default="standard")
    parser.add_argument("--max-eval-prompts", type=int, default=None)
    args = parser.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.max_eval_prompts)


if __name__ == "__main__":
    main()
