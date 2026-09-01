---
name: ollama-bench
description: Fairly benchmark local ollama model throughput (tokens/sec) and A/B compare two models or two Modelfile parameter sets. Use when asked to benchmark, measure speed, compare tok/s, check if a model/param change is faster or slower, or produce before/after perf numbers for an ollama model — especially for runner/MTP/quant work. Not for remote/cloud models.
---

# ollama-bench

Repeatable, apples-to-apples throughput benchmarks for local ollama models. Exists because ad-hoc `curl`/`ollama run` loops drift: they forget the warmup, run unbounded and blow timeouts, and don't fix the seed — so two "benchmarks" aren't comparable. This standardizes the method.

## When to use

- "Is this faster/slower?" for a Modelfile param change (e.g. `draft_num_predict`, `num_ctx`, quant).
- Before/after numbers for MLX runner / MTP / speculative-decoding work.
- A/B two models or two variants of the same base.

## The method (why each knob matters)

- **Warmup run, discarded** — the first request pays cold model-load (seconds). Never measure it.
- **`num_predict` cap** — bounds each run so it finishes predictably. An uncapped "write 300 words" run can generate 1700 tokens and blow a 2-minute timeout (learned the hard way).
- **Fixed seed** — makes token counts comparable run-to-run and model-to-model.
- **N runs, averaged** — one run is an anecdote; report the average and eyeball the spread for noise.
- **Report gen AND prefill rate** — generation tok/s is the headline, but prompt-eval (prefill) tok/s matters for long-context work.
- **HTTP API, not `ollama run`** — clean JSON stats, exact control, no TTY escape junk.

## How to run

```bash
# single model
.claude/skills/ollama-bench/scripts/bench.sh qwen3.6:27b-fast

# A/B two models/variants (both benchmarked with identical settings)
RUNS=3 NUM_PREDICT=256 .claude/skills/ollama-bench/scripts/bench.sh qwen3.6:27b-fast qwen3.6:27b
```

Env overrides: `PROMPT`, `NUM_PREDICT` (default 200), `RUNS` (3), `SEED` (42), `THINK` (`false`), `HOST` (`http://localhost:11434`).

The script preflights that ollama is reachable and prints a per-run + average table, then the settings line so results are reproducible.

## Reporting results

- Lead with the average gen tok/s per model and the delta (e.g. "19.1 → 12.0 tok/s, ~37% slower").
- Call out variance if runs disagree by more than a few percent.
- When a change underperforms, explain _why_ if known (e.g. "no MTP head on this GGUF, so speculative decoding is all overhead") — a number without a mechanism invites a re-run.
- Note the exact settings (num_predict/seed/think) so the comparison is reproducible.

## Gotchas

- Only one model is resident at a time; benchmarking model B evicts A. That's fine (load time is excluded via warmup), but don't interleave A/B runs — do all of A, then all of B (the script already does this).
- Thinking models: keep `THINK=false` for throughput comparisons unless you're specifically measuring thinking-mode cost, or the reasoning preamble dominates the token budget.
- `num_predict` caps generation, so a model that would naturally stop early may report fewer tokens — that's fine for rate, but keep the cap ≥ what all models will produce for a clean rate.
