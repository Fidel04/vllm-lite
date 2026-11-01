import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def load():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    return tok, model

def build_inputs(tok, prompt):
    messages = [{"role":"user", "content": prompt}]
    return tok.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True
    ).to("cuda")

def eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set();
    return set(eos) if isinstance(eos, (list,tuple)) else {eos}

@torch.inference_mode()
def step(model, input_ids, attention_mask, past_key_values = None, use_cache = True):
    out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values = past_key_values, use_cache = use_cache, logits_to_keep=1)
    cache = out.past_key_values
    next_logits = out.logits[:,-1,:]

    return next_logits, cache
    
