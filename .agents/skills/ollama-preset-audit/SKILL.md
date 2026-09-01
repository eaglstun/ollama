---
name: ollama-preset-audit
description: Audit a local ollama model's shipped sampling defaults (temperature, top_p, presence_penalty, etc.) against the model author's upstream recommended presets on HuggingFace, and flag mismatches — especially thinking-vs-instruct preset crossing. Use when a model "feels off" or underwhelming, after pulling a new model, when checking whether ollama's default params are right, or when investigating quality/repetition/speed complaints that might be sampling-related. Not a benchmark (that's ollama-bench).
---

# ollama-preset-audit

Checks whether an ollama model's **shipped default sampling parameters** actually match what the model's authors recommend. Ollama's official models inherit params from upstream, but the mapping is hand-maintained and can be wrong — this skill found a real one: `qwen3.6:27b` ships `presence_penalty 1.5` (the upstream _instruct_ value) while defaulting to _thinking_ mode, for which the author recommends `0.0`. That mismatch degrades output. (Filed as ollama/ollama#17197.)

## When to use

- A model feels underwhelming, rambly, repetitive, or oddly worded — before blaming the weights, check the sampling.
- Right after pulling/building a new model, as a sanity check.
- Any "are these defaults right?" question.

## The core insight

Reasoning models publish **two+ presets**: one for **thinking mode**, one for **non-thinking / instruct**. They differ (often in `temperature`, `top_p`, and especially `presence_penalty`). The common packaging bug is **crossing them** — shipping thinking-mode `temp`/`top_p` alongside the instruct-mode `presence_penalty`, or vice versa. Since the model usually **defaults to thinking**, judge the shipped params against the **thinking** preset.

## Procedure

1. **Dump local params + identity:**

   ```bash
   .claude/skills/ollama-preset-audit/scripts/local-params.sh <model>
   ```

   This prints the architecture, quant, capabilities (note whether `thinking` is present), the shipped `PARAMETER` lines, and next steps.

2. **Find the upstream source.** Usually `Qwen/<name>`, `meta-llama/<name>`, `google/<name>`, `mistralai/<name>`. If unsure, web-search `"<model> huggingface generation_config best practices"`.

3. **Fetch the ground truth (two sources):**
   - Raw `generation_config.json`: `https://huggingface.co/<org>/<repo>/raw/main/generation_config.json` — the literal sampling defaults (often minimal: just temp/top_p/top_k).
   - The model card's **Best Practices / recommended sampling** section — this is where the per-mode presets (thinking vs instruct) and any `presence_penalty` guidance live. The card usually holds the presets the raw JSON omits.

4. **Build the comparison table.** Rows = each upstream preset (thinking-general, thinking-coding, instruct) + the ollama-shipped row. Columns = temperature, top_p, top_k, min_p, presence_penalty, repeat_penalty.

5. **Flag mismatches**, judging the shipped row against the mode the model **defaults to** (thinking, if advertised). Call out specifically:
   - a `presence_penalty` from the wrong preset (the classic bug),
   - `temperature`/`top_p` that match neither preset,
   - params present locally but absent upstream (added by the packager) or vice versa.

6. **Report** with a clear verdict: is the default correct, a defensible choice, or a genuine cross-preset bug? Give the corrected params for the user's actual mode, and note that a fixed local variant can be built with a Modelfile (`FROM <model>` + corrected `PARAMETER` lines, rebuilt via `ollama create <model>-fast -f ...`).

## Honesty guardrails

- The `generation_config.json` → ollama mapping is an inference about ollama's packaging practice — say so; don't assert ollama "did X" without the evidence in front of you.
- A divergence may be **intentional** (e.g. a deliberate anti-repetition default). Frame findings as "the default appears to cross presets," and let the evidence — not a hunch — carry the verdict.
- Verify against the _current_ upstream card; model authors revise recommended settings between releases.

## Related

- Speed complaints are often thinking-mode overhead, not sampling — quantify with **ollama-bench** and check whether `--think=false` is the real fix.
- Filing an upstream issue from a finding: keep it factual, link the three sources (card, generation_config, ollama model page), and respect draft-only / low-profile norms.
