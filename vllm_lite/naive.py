import torch
from .model import step

@torch.inference_mode()
def generate_no_cache(model, input_ids, attention_mask, max_new_tokens, eos, on_token=None):
    ids = input_ids
    mask = attention_mask
    new_token_ids = []

    for _ in range(max_new_tokens):
        next_logits, _ = step(model, ids, mask, use_cache=False)
        next_token_id = next_logits.argmax(dim=1, keepdim=True)
        tid = next_token_id.item()
        new_token_ids.append(tid)
        if on_token is not None:
            on_token(tid)
        if tid in eos:
            break
        ids = torch.cat((ids, next_token_id), dim=1)
        mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)

    return new_token_ids

@torch.inference_mode()
def generate_with_cache(model, input_ids, attention_mask, max_new_tokens, eos, on_token=None):
    cur = input_ids
    mask = attention_mask
    cache = None
    new_token_ids = []

    for _ in range(max_new_tokens):
        next_logits, cache = step(model, cur, mask, past_key_values=cache, use_cache=True)
        next_token_id = next_logits.argmax(dim=1, keepdim=True)
        tid = next_token_id.item()
        new_token_ids.append(tid)
        if on_token is not None:
            on_token(tid)
        if tid in eos:
            break
        cur = next_token_id
        mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)

    return new_token_ids
