from __future__ import annotations

import torch
import torch.nn.functional as F


def _response_mask(response_ids: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    mask = torch.ones_like(response_ids, dtype=torch.float32)
    if eos_id is None:
        return mask
    for i in range(response_ids.shape[0]):
        eos_pos = (response_ids[i] == eos_id).nonzero(as_tuple=False)
        if len(eos_pos):
            first = int(eos_pos[0].item())
            if first + 1 < response_ids.shape[1]:
                mask[i, first + 1:] = 0.0
    return mask


def batch_generate(
    model,
    tokenizer,
    prompts: list[list[dict]],
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    rendered = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    was_training = model.training
    model.eval()
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        seq = model.generate(**enc, **kwargs)

    seq = seq.clone()
    
    if was_training:
        model.train()

    prompt_width = enc["input_ids"].shape[1]
    response_ids = seq[:, prompt_width:]
    rmask = _response_mask(response_ids, tokenizer.eos_token_id).to(seq.device)
    terminated, truncated, lengths, texts = [], [], [], []
    for row, mask in zip(response_ids, rmask):
        n = int(mask.sum().item())
        has_eos = bool((row[:n] == tokenizer.eos_token_id).any().item()) if tokenizer.eos_token_id is not None else False
        terminated.append(has_eos)
        truncated.append(bool(n >= max_new_tokens and not has_eos))
        lengths.append(n)
        texts.append(tokenizer.decode(row[:n], skip_special_tokens=True))

    attn = torch.ones_like(seq, dtype=torch.long)
    attn[:, :prompt_width] = enc["attention_mask"]
    return {
        "sequences": seq,
        "attention_mask": attn,
        "prompt_width": prompt_width,
        "response_ids": response_ids,
        "response_mask": rmask,
        "responses": texts,
        "terminated_with_eos": terminated,
        "truncated": truncated,
        "response_lengths": lengths,
    }




def response_token_logprobs(
    model,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    response_ids: torch.Tensor,
    chunk_size: int = 32,
):
    """
    Return chosen-token log probabilities and response-position logits.

    Computes log probabilities in chunks of sequence positions to reduce
    the size of temporary float32 log-softmax tensors.
    """
    outputs = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    start = prompt_width - 1
    end = start + response_ids.shape[1]
    logits = outputs.logits[:, start:end, :]

    # Compute chosen-token log probabilities in chunks.
    chosen_chunks = []
    for pos in range(0, logits.shape[1], chunk_size):
        logits_chunk = logits[:, pos:pos + chunk_size, :].float()
        targets = response_ids[:, pos:pos + chunk_size]

        log_probs = torch.log_softmax(logits_chunk, dim=-1)
        chosen = torch.gather(
            log_probs,
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)

        chosen_chunks.append(chosen)

    chosen_log_probs = torch.cat(chosen_chunks, dim=1)

    return chosen_log_probs, logits



def response_sequence_logprobs(model, batch: dict):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    tok = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1), tok, mask


@torch.no_grad()
def score_reward_pairs(rm_model, rm_tokenizer, prompts, responses, max_length=1024):
    texts = []
    for prompt, response in zip(prompts, responses):
        messages = list(prompt) + [{"role": "assistant", "content": response}]
        texts.append(
            rm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    device = next(rm_model.parameters()).device
    enc = rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    return rm_model(**enc).logits[:, 0].float()
