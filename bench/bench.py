"""Offline benchmark harness.

Generates a fixed, seeded workload of random-token prompts, submits every
request at t=0, serves them with the chosen engine, and records per-request
timings plus throughput and p50/p99 summaries to bench/results/<label>.json.

All requests arriving at t=0 means latency includes queueing: request i waits
for requests 0..i-1 under the naive engine. That is the cost continuous
batching (phase 3) attacks, so keep the workload flags identical across runs.

Engine contract (per-request engines, phase 2):
    fn(model, input_ids, attention_mask, max_new_tokens, eos, on_token=None) -> list[int]

Usage:
    python bench/bench.py --engine vllm_lite.naive:generate_with_cache --label naive_cached
    python bench/bench.py --engine vllm_lite.naive:generate_with_cache --label naive_cached --n 100

Restart the process and confirm Sysmem Fallback is off before any run you quote.
"""

import argparse
import importlib
import json
import platform
import random
import time
from pathlib import Path

import torch
import transformers

from vllm_lite.model import load


def load_engine(spec):
    module_name, fn_name = spec.split(":")
    return getattr(importlib.import_module(module_name), fn_name)
    # importlib.import_module dynamically imports module module_name


def make_workload(n, prompt_min, prompt_max, out_min, out_max, vocab_size, seed):
    """Seeded list of (prompt_ids [1, L] on CPU, out_len). Same args -> same workload."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed) # random no generator object
    workload = []
    for _ in range(n):
        prompt_len = rng.randint(prompt_min, prompt_max)
        out_len = rng.randint(out_min, out_max)
        ids = torch.randint(0, vocab_size, (1, prompt_len), generator=gen)
        workload.append((ids, out_len))
    return workload


def percentile(xs, p):
    """Linear interpolation between closest ranks (same as numpy's default)."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values):
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def run_request(engine, model, prompt_ids, out_len, device):
    """Serve one request. Returns (start, first_token_time, finish, n_tokens)."""
    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    token_times = []

    # .item() inside the engine syncs the GPU each token, so these timestamps
    # mark when each token actually exists.
    def on_token(_tid):
        token_times.append(time.perf_counter())

    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = engine(model, input_ids, attention_mask, out_len, set(), on_token=on_token)
    torch.cuda.synchronize()
    finish = time.perf_counter()

    assert len(tokens) == out_len, f"expected {out_len} tokens, got {len(tokens)} (EOS should be off)"
    return start, token_times, finish


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="module:function, e.g. vllm_lite.naive:generate_with_cache")
    ap.add_argument("--label", required=True, help="results file name")
    ap.add_argument("--n", type=int, default=20, help="number of requests")
    ap.add_argument("--prompt-min", type=int, default=64)
    ap.add_argument("--prompt-max", type=int, default=512)
    ap.add_argument("--out-min", type=int, default=32)
    ap.add_argument("--out-max", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2, help="untimed warmup requests")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    engine = load_engine(args.engine)
    tok, model = load()
    device = model.device
    vocab_size = tok.vocab_size  # base vocab, excludes special tokens

    workload = make_workload(args.n, args.prompt_min, args.prompt_max,
                             args.out_min, args.out_max, vocab_size, args.seed)
    warmup = make_workload(args.warmup, 32, 32, 8, 8, vocab_size, args.seed + 1)

    for prompt_ids, out_len in warmup:
        run_request(engine, model, prompt_ids, out_len, device)
    torch.cuda.reset_peak_memory_stats()

    # Every request arrives at t0; naive serving handles them in order.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    requests = []
    for i, (prompt_ids, out_len) in enumerate(workload):
        start, token_times, finish = run_request(engine, model, prompt_ids, out_len, device)
        tpot = (token_times[-1] - token_times[0]) / (out_len - 1) if out_len > 1 else 0.0  # time per output token
        requests.append({
            "id": i,
            "prompt_len": prompt_ids.shape[1],
            "out_len": out_len,
            "queue_wait": start - t0,
            "ttft": token_times[0] - t0, # time to first token
            "latency": finish - t0,
            "service_time": finish - start,
            "tpot": tpot,
        })
        print(f"[{i + 1}/{args.n}] prompt={prompt_ids.shape[1]:4d} out={out_len:4d} "
              f"service={finish - start:6.2f}s latency={finish - t0:7.2f}s", flush=True)
    wall = time.perf_counter() - t0

    total_out = sum(r["out_len"] for r in requests)
    props = torch.cuda.get_device_properties(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    summary = {
        "wall_time_s": wall,
        "throughput_tok_s": total_out / wall,
        "requests_per_s": args.n / wall,
        "total_output_tokens": total_out,
        "latency_s": summarize([r["latency"] for r in requests]),
        "ttft_s": summarize([r["ttft"] for r in requests]),
        "tpot_s": summarize([r["tpot"] for r in requests]),
        "service_time_s": summarize([r["service_time"] for r in requests]),
        "peak_reserved_mb": peak_reserved / 2**20,
    }

    result = {
        "label": args.label,
        "engine": args.engine,
        "workload": {k: getattr(args, k) for k in
                     ("n", "prompt_min", "prompt_max", "out_min", "out_max", "seed")},
        "env": {
            "gpu": props.name,
            "vram_total_mb": props.total_memory / 2**20,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "summary": summary,
        "requests": requests,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n== {args.label} ({args.engine}) ==")
    print(f"wall {wall:.1f}s | {summary['throughput_tok_s']:.1f} tok/s | {summary['requests_per_s']:.3f} req/s")
    for name in ("latency_s", "ttft_s", "tpot_s"):
        s = summary[name]
        print(f"{name:10s} p50 {s['p50']:8.3f}  p99 {s['p99']:8.3f}  max {s['max']:8.3f}")
    print(f"peak reserved {summary['peak_reserved_mb']:.0f} MB / {result['env']['vram_total_mb']:.0f} MB")
    if args.n < 100:
        print(f"note: n={args.n}; p99 is basically the max at this size. Use --n 100 before quoting p99.")
    if peak_reserved > props.total_memory:
        print("WARNING: peak reserved exceeds VRAM. Sysmem fallback is spilling; timings are invalid.")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
