"""Time the standalone MLX Talkie reference with the same knobs as ollama-bench.

One discarded warmup, then RUNS runs of MAX_TOKENS at a fixed seed. Stop tokens are
disabled so every run generates the full count, like ollama's num_predict cap. The
decode rate excludes prefill, matching ollama's eval_duration.

The reference package is not part of this repo, so run this with its venv:

    cd ~/Documents/AI/talkie
    .venv/bin/python ~/Documents/dev/ollama/x/models/talkie/bench_reference.py

Env overrides: TALKIE_MODEL_DIR, PROMPT, MAX_TOKENS (200), RUNS (3), SEED (42).
Unload talkie from the fork first (keep_alive 0): two 26 GB copies do not fit in 64 GB.
"""

import os
import time

import mlx.core as mx

from talkie.chat import format_prompt
from talkie.mlx.generate import MLXGenerationConfig, MLXTalkie

MODEL_DIR = os.environ.get(
    "TALKIE_MODEL_DIR", os.path.expanduser("~/models/talkie-1930-13b-it-mlx")
)
PROMPT = os.environ.get(
    "PROMPT", "Write a long and detailed essay upon the future of the railways."
)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 200))
RUNS = int(os.environ.get("RUNS", 3))
SEED = int(os.environ.get("SEED", 42))

t0 = time.perf_counter()
talkie = MLXTalkie(MODEL_DIR)
talkie.stop_token_ids = set()
prompt = format_prompt(PROMPT)
print(f"load {time.perf_counter() - t0:.1f}s | mlx {mx.__version__}", flush=True)

cfg = MLXGenerationConfig(max_tokens=MAX_TOKENS, seed=SEED)
n_prompt = len(talkie.tokenizer.encode(prompt, allowed_special="all"))


def one_run():
    start = time.perf_counter()
    first = last = None
    n = 0
    for _ in talkie._generate_ids(prompt, cfg):
        now = time.perf_counter()
        if first is None:
            first = now  # prefill + first sample done
        last = now
        n += 1
    return n, first - start, (n - 1) / (last - first)


one_run()  # warmup, discarded
rates = []
for r in range(1, RUNS + 1):
    n, ttft, rate = one_run()
    rates.append(rate)
    print(
        f"run {r}: {n} tok | prefill+first {ttft * 1000:.0f} ms ({n_prompt} prompt tok)"
        f" | gen {rate:.2f} tok/s",
        flush=True,
    )
print(f"AVG gen {sum(rates) / len(rates):.2f} tok/s | peak mem {mx.get_peak_memory() / 1e9:.1f} GB")
