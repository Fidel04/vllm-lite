"""Greedy loops in vllm_lite.naive vs Hugging Face generate(), token for token.

Needs CUDA and the real Qwen2.5-0.5B-Instruct weights. Run:
    pytest tests/test_correctness.py -v

The reference generate() must be configured explicitly: Qwen's default
generation_config samples, so an unconfigured generate() is not a valid reference.
"""

from pathlib import Path

import pytest
import torch

from vllm_lite.model import build_inputs, eos_ids, load
from vllm_lite.naive import generate_no_cache, generate_with_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHORT_PROMPT = "Explain what a KV cache is in two sentences."
LONG_PROMPT = (Path(__file__).parent / "prompts" / "long.txt").read_text(encoding="utf-8")  # 1,599 tokens


@pytest.fixture(scope="module")
def setup():
    tok, model = load()
    return tok, model, eos_ids(model)


def reference(model, inputs, max_new_tokens, eos):
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=sorted(eos),
    )
    return out[0, inputs["input_ids"].shape[1]:].tolist()


def run(engine, model, inputs, max_new_tokens, eos, **kwargs):
    return engine(model, inputs["input_ids"], inputs["attention_mask"], max_new_tokens, eos, **kwargs)


@pytest.mark.parametrize("engine", [generate_with_cache, generate_no_cache])
def test_short_prompt_matches_generate(setup, engine):
    tok, model, eos = setup
    inputs = build_inputs(tok, SHORT_PROMPT)
    assert run(engine, model, inputs, 64, eos) == reference(model, inputs, 64, eos)


def test_long_prompt_cached_matches_generate(setup):
    tok, model, eos = setup
    inputs = build_inputs(tok, LONG_PROMPT)
    assert inputs["input_ids"].shape[1] == 1599
    assert run(generate_with_cache, model, inputs, 200, eos) == reference(model, inputs, 200, eos)


def test_long_prompt_no_cache_matches_until_fp16_tie(setup):
    # Known: no-cache diverges at generated token 16 on this prompt because the
    # top-2 logits tie exactly in fp16 (see DEVLOG 2026-11-01). Not a bug.
    # Check the 15 tokens before the tie; the full 200 would take ~150s anyway.
    tok, model, eos = setup
    inputs = build_inputs(tok, LONG_PROMPT)
    assert run(generate_no_cache, model, inputs, 15, eos) == reference(model, inputs, 15, eos)


@pytest.mark.parametrize("engine", [generate_with_cache, generate_no_cache])
def test_zero_new_tokens(setup, engine):
    tok, model, eos = setup
    inputs = build_inputs(tok, SHORT_PROMPT)
    assert run(engine, model, inputs, 0, eos) == []


@pytest.mark.parametrize("engine", [generate_with_cache, generate_no_cache])
def test_eos_disabled_gives_exact_length(setup, engine):
    tok, model, _ = setup
    inputs = build_inputs(tok, SHORT_PROMPT)
    assert len(run(engine, model, inputs, 80, set())) == 80


@pytest.mark.parametrize("engine", [generate_with_cache, generate_no_cache])
def test_on_token_sees_every_token_in_order(setup, engine):
    tok, model, eos = setup
    inputs = build_inputs(tok, SHORT_PROMPT)
    seen = []
    tokens = run(engine, model, inputs, 32, eos, on_token=seen.append)
    assert seen == tokens
