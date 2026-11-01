import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .model import step

@torch.inference_mode()
def generate_no_cache(model, input_ids, attention_mask, max_new_tokens, eos):
    ids = input_ids
    mask = attention_mask
    new_token_ids = []

    for _ in range(max_new_tokens):
        next_logits, _ = step(model, ids, mask, use_cache=False)
        next_token_id = next_logits.argmax(dim=1, keepdim=True)
        tid = next_token_id.item()
        new_token_ids.append(tid)
        ids = torch.cat((ids,next_token_id), dim=1)
        mask = torch.cat((mask,torch.tensor([[1]]).to("cuda")),dim=1)
        if tid in eos:
            break

    return new_token_ids

@torch.inference_mode()
def generate_with_cache(model, input_ids, attention_mask, max_new_tokens, eos):
    ids = input_ids 
    mask = attention_mask
    new_token_ids = []

    next_logits, cache = step(model, ids, mask, use_cache=True)
    next_token_id = next_logits.argmax(dim=1, keepdim=True)
    tid = next_token_id.item()
    new_token_ids.append(tid)
    ids = torch.cat((ids,next_token_id), dim=1)
    mask = torch.cat((mask,torch.tensor([[1]]).to("cuda")), dim=1)
    if(tid in eos): 
        return new_token_ids

    for i in range(max_new_tokens-1):
        next_logits, cache = step(model, next_token_id, mask, past_key_values=cache, use_cache=True)
        next_token_id = next_logits.argmax(dim=1, keepdim=True)
        tid = next_token_id.item()
        new_token_ids.append(tid)
        mask = torch.cat((mask,torch.tensor([[1]]).to("cuda")), dim=1)
        if(tid in eos):
            break

    return new_token_ids
        

    