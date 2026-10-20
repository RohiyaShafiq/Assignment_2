from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


# ============================================================
# Data collation
# ============================================================

def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen = []
        rejected = []

        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)

            chosen.append(
                encode_prompt_response(
                    tokenizer,
                    prompt,
                    yc,
                    max_length,
                )
            )

            rejected.append(
                encode_prompt_response(
                    tokenizer,
                    prompt,
                    yr,
                    max_length,
                )
            )

        return (
            pad_batch(tokenizer, chosen),
            pad_batch(tokenizer, rejected),
        )

    return collate


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

# ============================================================
# Prepare model, tokenizer, data and optimizer
# ============================================================

def prepare_dpo_run(
    config_path: str,
    dataset_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    cfg = load_yaml(config_path)

    # Reproducibility
    set_seed(int(cfg["seed"]))

    # Fixed course-provided standard DPO dataset
    path = dataset_path or cfg["paths"]["dpo_standard_train"]

    rows = read_jsonl(path)

    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    # Filter examples that cannot fit under the released
    # max_sequence_length constraint.
    rows = filter_long_prompts(
        tokenizer,
        rows,
        int(cfg["max_sequence_length"]),
    )

    # Fresh LoRA initialization
    model = load_policy(
        cfg,
        trainable=True,
        fresh_lora=True,
    )

    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(
            tokenizer,
            int(cfg["max_sequence_length"]),
        ),
    )

    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(
            cfg.get("weight_decay", 0.0)
        ),
    )

    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(
            cfg["beta"] if beta is None else beta
        ),
    }


# ============================================================
# Sequence log probability
# ============================================================

def sequence_logprob(model, batch):
    """
    Compute the summed log probability of the RESPONSE tokens.

    Prompt tokens are excluded using response_mask.

    Because this is a causal LM:
        logits[:, t] predicts input_ids[:, t+1]
    """

    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits

    # Shift for causal language modeling
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    shift_attention = attention_mask[:, 1:]
    shift_response_mask = response_mask[:, 1:]

    log_probs = torch.log_softmax(
        shift_logits,
        dim=-1,
    )

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    # Keep only response-token log probabilities
    token_log_probs = (
        token_log_probs
        * shift_attention
        * shift_response_mask
    )

    return token_log_probs.sum(dim=-1)


# ============================================================
# Training
# ============================================================

def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    bundle = prepare_dpo_run(
        config_path,
        dataset_path,
        beta,
        max_examples,
    )

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]

    beta_value = bundle["beta"]

    output = repo_path(
        output_path
        or cfg["standard_output"]
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = next(
        model.parameters()
    ).device

    epochs = int(cfg["epochs"])
    grad_accum_steps = int(
        cfg.get("grad_accum_steps", 1)
    )

    max_grad_norm = float(
        cfg.get("max_grad_norm", 1.0)
    )

    print("=" * 70)
    print("STANDARD DPO TRAINING")
    print("=" * 70)

    print(f"Run name:              {run_name}")
    print(f"Base model:            {cfg['base_model']}")
    print(f"Dataset:               {cfg['paths']['dpo_standard_train']}")
    print(f"Training examples used:    {len(rows)}")
    print(f"Epochs:                {epochs}")
    print(f"Batch size:            {cfg['batch_size']}")
    print(f"Gradient accumulation: {grad_accum_steps}")
    print(f"Effective batch size:  {int(cfg['batch_size']) * grad_accum_steps}")
    print(f"Learning rate:         {cfg['learning_rate']}")
    print(f"Beta:                  {beta_value}")
    print(f"Max sequence length:   {cfg['max_sequence_length']}")
    print(f"Device:                {device}")
    print("=" * 70)

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    global_step = 0

    total_examples = 0
    total_loss = 0.0

    history = []

    for epoch in range(epochs):

        epoch_loss = 0.0
        epoch_examples = 0

        epoch_correct = 0.0

        for batch_idx, (
            chosen,
            rejected,
        ) in enumerate(loader):

            # ------------------------------------------------
            # Move batches to device
            # ------------------------------------------------

            chosen = {
                k: v.to(device)
                for k, v in chosen.items()
            }

            rejected = {
                k: v.to(device)
                for k, v in rejected.items()
            }

            batch_size = chosen[
                "input_ids"
            ].size(0)

            # ------------------------------------------------
            # Policy log probabilities
            # ------------------------------------------------

            policy_chosen_logp = sequence_logprob(
                model,
                chosen,
            )

            policy_rejected_logp = sequence_logprob(
                model,
                rejected,
            )

            # ------------------------------------------------
            # Reference policy log probabilities
            #
            # The reference policy is the original base model
            # with the trainable LoRA adapter disabled.
            # ------------------------------------------------

            with torch.no_grad():

                with reference_mode(model):

                    ref_chosen_logp = sequence_logprob(
                        model,
                        chosen,
                    )

                    ref_rejected_logp = sequence_logprob(
                        model,
                        rejected,
                    )

            # ------------------------------------------------
            # Standard DPO objective
            # ------------------------------------------------

            loss, diagnostics = dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=beta_value,
            )

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            loss_for_backward = (
                loss / grad_accum_steps
            )

            loss_for_backward.backward()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            batch_loss = loss.item()

            batch_accuracy = diagnostics[
                "preference_accuracy"
            ].item()

            epoch_loss += (
                batch_loss * batch_size
            )

            epoch_examples += batch_size

            epoch_correct += (
                batch_accuracy * batch_size
            )

            total_loss += (
                batch_loss * batch_size
            )

            total_examples += batch_size

            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            should_step = (
                (batch_idx + 1)
                % grad_accum_steps
                == 0
                or
                (batch_idx + 1)
                == len(loader)
            )

            if should_step:

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model),
                    max_grad_norm,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

                # ------------------------------------------------
                # Logging
                # ------------------------------------------------

                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "batch": batch_idx + 1,
                    "loss": batch_loss,
                    "preference_accuracy": batch_accuracy,
                    "policy_margin": diagnostics[
                        "policy_margin_mean"
                    ].item(),
                    "logit_mean": diagnostics[
                        "logit_mean"
                    ].item(),
                }

                history.append(record)

                if (
                    global_step % 10 == 0
                    or global_step == 1
                ):
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Step {global_step:4d} | "
                        f"Loss {batch_loss:.4f} | "
                        f"Pref Acc {batch_accuracy:.4f} | "
                        f"Policy Margin "
                        f"{record['policy_margin']:.4f}"
                    )

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        avg_epoch_loss = (
            epoch_loss
            / max(epoch_examples, 1)
        )

        epoch_accuracy = (
            epoch_correct
            / max(epoch_examples, 1)
        )

        print()
        print(
            f"Epoch {epoch + 1}/{epochs} completed"
        )
        print(
            f"Average loss:        {avg_epoch_loss:.6f}"
        )
        print(
            f"Preference accuracy: {epoch_accuracy:.6f}"
        )
        print()

    # ========================================================
    # Save LoRA adapter
    # ========================================================

    print("=" * 70)
    print("Saving trained DPO adapter")
    print("=" * 70)

    model.save_pretrained(
        output
    )

    tokenizer.save_pretrained(
        output
    )

    # ========================================================
    # Save training history
    # ========================================================

    average_loss = (
        total_loss
        / max(total_examples, 1)
    )

    training_summary = {
        "run_name": run_name,
        "base_model": cfg["base_model"],
        "dataset": cfg["paths"]["dpo_standard_train"],
        "dataset_size": len(rows),
        "epochs": epochs,
        "batch_size": int(
            cfg["batch_size"]
        ),
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": (
            int(cfg["batch_size"])
            * grad_accum_steps
        ),
        "learning_rate": float(
            cfg["learning_rate"]
        ),
        "beta": beta_value,
        "max_sequence_length": int(
            cfg["max_sequence_length"]
        ),
        "average_training_loss": average_loss,
        "global_steps": global_step,
    }

    with open(
        output / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            training_summary,
            f,
            indent=2,
        )

    with open(
        output / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    print(
        f"Adapter saved to: {output}"
    )

    print(
        f"Training summary saved to: "
        f"{output / 'training_summary.json'}"
    )

    print(
        f"Training history saved to: "
        f"{output / 'training_history.json'}"
    )

    print("=" * 70)
    print("DPO TRAINING COMPLETE")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--output", default=None)

    ap.add_argument(
        "--beta",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--max-examples",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    run_training(
        config_path=args.config,
        run_name=args.run_name,
        dataset_path=args.dataset,
        output_path=args.output,
        beta=args.beta,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()