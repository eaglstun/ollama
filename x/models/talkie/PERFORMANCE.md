# talkie-1930: where the decode time goes, and what would make it faster

Research notes, 2026-09-12, on the M4 Max (40-core GPU, 546 GB/s, 64 GB). Items 1 and 2
are applied on branch `talkie-decode-speedups`; see
[Applied: items 1 and 2](#applied-items-1-and-2) for the measured result and one
correction to what this file predicted. Everything else here is a measured map of the
options, not a change. Every number is
labelled **measured** (a command in this file produced it) or **inferred** (arithmetic
from measured numbers or from reading code). All measurements were taken with both
servers idle (`/api/ps` empty on 11434 and 11435 before each run) and with only one
copy of the model resident at a time.

## Summary

The fork decodes at **60.6 ms/token (16.5 tok/s)**. The real floor for this model on
this GPU is not the 49 ms the paper bandwidth suggests but **55.4 ms** (measured: the
bare GEMVs over the 25.9 GB a token has to read run at 467 GB/s, 85% of the 546
nominal). So the port carries about **5 ms/token of overhead**, and all of it is
accounted for by three things in `talkie.go`.

Ranked by what they buy, divided by what they cost:

1. **Fold `lm_head_gain` into `lm_head` once at load** instead of multiplying the whole
   671 MB table every token (`talkie.go:288`). −2.8 ms/token, bit-identical output,
   three lines. Talkie-specific.
2. **Use `mlx.fast.rms_norm` with no weight** for the weightless RMSNorm
   (`talkie.go:234-240`, called 162 times per token). −2.1 ms/token, bit-identical to
   the reference's `_rms_norm` on a 60-token greedy test, one line
   (`mlx.RMSNormFn(x, nil, rmsEps)`). Talkie-specific. It is not bit-identical to the
   port's own previous formula, which had drifted from the reference; see Applied.
   Together, 1 and 2 measure at −4.9 ms on the reference, which would put the fork at
   about 55.7 ms/token, roughly **18 tok/s, at the floor** (inferred).
3. **int8 quantization** is the only way past the floor at this parameter count. It
   measured **33.5 ms/token (29.9 tok/s)** on the reference, mean KL 0.017 nats against
   bf16, top-1 agreement 95%, and the prose stays in period. The fork's create path
   would quantize the right tensors for talkie without changes. Uniform 4-bit is not
   acceptable for this model (KL 0.85, degenerate repetition); a mixed int4 with K/V
   and the MLP down projection at int8 lands in between (25.3 ms, KL 0.10) but the
   create policy would need talkie's tensor names added to get it.
4. **`mlx.fast.rope` with `scale = -1`** replaces the 10-op hand RoPE and buys another
   −1.1 ms, but it is not bit-identical (max logit difference 0.25, greedy diverges
   after 41 tokens), so it costs the seeded exact-match test. Same for compiling the
   residual/gain arithmetic (−1.4 ms, diverges after 7 tokens).

Everything else checked (fusing Q/K/V and gate/up, hoisting the RoPE table gathers,
the sampler, KV-cache growth, prefill, the array-lifetime rework, `ClearCache`) is
either already fine or worth under 0.5 ms and is listed under "Not worth doing".

Speculative decoding has no cheap path here: talkie ships no draft head, no draft model
exists for its 65540-token vocabulary, and the runner has no n-gram drafter. Details
below on what it would take and why the earlier −37% result is expected to repeat.

## Applied: items 1 and 2

Both are in `talkie.go` on branch `talkie-decode-speedups`. Fork, `ollama-bench` on port
11435, 200 tokens, seed 42, warmup plus 3 runs, both servers idle, all in one session:

| build              | gen tok/s | ms/token | run range   |
| ------------------ | --------- | -------- | ----------- |
| before             | 16.46     | 60.8     | 16.42–16.52 |
| item 1 (fold) only | 17.30     | 57.8     | 17.27–17.32 |
| items 1 and 2      | **17.80** | **56.2** | 17.76–17.82 |

−4.6 ms/token against the −4.9 predicted, and 0.8 ms above the 55.4 ms floor.

**Item 1 is exact, as predicted.** With only the fold applied, the README's seeded smoke
test (temperature 0.75, seed 1930) reproduced the pre-change output byte for byte.

**Item 2 is not exact against the old port, and the old port was the one that was wrong.**
With both applied, the seeded answer changed from the first token. At position 0 of the
greedy run on the smoke-test prompt, the port's top two tokens moved from
`It` −0.68 / `The` −0.93 to `The` −0.37 / `It` −1.49. The "bit-identical" result above
was measured against the reference's `_rms_norm`, which computes
`x * rsqrt(mean(x^2) + eps)`. The port's hand-rolled norm computed
`x / sqrt(mean(x^2) + eps)` instead, and that rounding difference compounds across 162 norms
per token. `norm_truth.py` (below) settles which side is correct by running the reference
model with each formula swapped in:

| norm inside the reference model | first-token top two         | greedy continuation starts                                               |
| ------------------------------- | --------------------------- | ------------------------------------------------------------------------ |
| reference `_rms_norm` (rsqrt)   | `The` −0.4027, `It` −1.4027 | "The wireless telephone may become a means of personal communication..." |
| `mx.fast.rms_norm`              | identical (max Δ 0.0000)    | identical                                                                |
| the port's old sqrt/divide      | `It` −0.6825, `The` −0.9325 | "It might become a means of communication between moving trains..."      |

The fused kernel is the reference. The old formula, dropped into the reference, reproduces
the old port's first-token logprobs to within 0.002 and its greedy text, and the fixed
port's greedy text matches the reference's. So item 2 is a correctness fix as well as a
speed fix: the port had been off-reference since it was written, and the smoke-test answer
recorded on 2026-09-02 came from that drifted forward pass. The README now carries the new
seeded text, and a comment on `rmsNorm` says not to go back.

A smaller gap remains between the fixed port and the reference (for example `The` −0.37
in the port vs −0.40 in the reference at position 0). It was not chased. The two graphs
still differ in places, for one the RoPE tables, which the port precomputes in float64
(`buildRoPE`) and the reference in float32, both then cast to bf16. The operation order
is the same in both (RoPE, then Q/K norm, then head gain). The gap does not change the
greedy text on this prompt: the fixed port's greedy output begins with the reference's
40-token greedy continuation word for word.

## Time budget for one decode token

Fork, short context (23-token prompt), temperature 0, 200 tokens: **60.6 ms** (measured,
`bench.sh` gave 16.50 tok/s; `fork_sampling.py` greedy runs gave 60.2–60.7 ms).

| component                                                                                                       | ms/token          | how known                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Weight streaming: 281 GEMVs over 25.89 GB at the achieved 467 GB/s                                              | 55.4              | measured (`floor.py`, MODE=separate)                                                                                                                    |
| of which the nominal-bandwidth figure (25.89 GB / 546 GB/s)                                                     | 47.4              | inferred; the other 8.0 ms is bandwidth this GPU does not deliver to GEMV, not overhead                                                                 |
| `lm_head * lm_head_gain` materialised every token (1.34 GB traffic)                                             | 2.8               | measured (`lmhead.py`: 4.35 vs 1.51 ms; in-model V3b: −2.7)                                                                                             |
| Hand-rolled RMSNorm, 7 kernels × 162 calls instead of 1 × 162                                                   | 2.1               | measured on the reference (V0→V1), inferred to transfer                                                                                                 |
| Hand-rolled RoPE, 2 × 10 kernels × 40 layers                                                                    | 1.1               | measured on the reference (V1→V2)                                                                                                                       |
| Residual gains, SwiGLU from primitives, KV slice-update, SDPA, sampling, Go graph build, cgo, scope bookkeeping | ≤ 1               | inferred: the items above already sum to 61.4 ms, so the rest is inside run-to-run drift; the pipelined decoder hides host work (`pipeline.go:373-470`) |
| Context: KV read grows 0.82 GB per 1000 tokens of context                                                       | +2.2 per 1000 ctx | measured (`fork_ctx.py`: 60.6 → 62.2 → 63.7 ms at 23 / 550 / 1211 tokens); 1.75 ms is the pure read at 467 GB/s                                         |

The reference model with GPU argmax (no numpy sampling) decodes at 62.3–62.7 ms; the
fork is already 2 ms faster than the reference's bare forward because it overlaps host
work with the GPU.

Kernel count, inferred from `talkie.go` per layer at L=1: 4 rmsNorm × 7 ops = 28,
2 rope × 10 = 20, 7 matmuls, 1 head-gain multiply, 2 KV slice-updates, 1 SDPA, 6 gain
and residual ops, 3 for silu·linear: about 68 kernels per layer, roughly 2700 per token.
`llama.go` does the same work in about 17 per layer (fused norm, fused rope, compiled
SwiGLU). The 972 kernels the fast norm removes cost 2.1 ms, so a tiny kernel costs about
2.2 µs here, dispatch-bound, and that is the whole story of the non-bandwidth overhead.

## Opportunities

### 1. Fold the lm_head gain at load

**What.** `Unembed` (`talkie.go:287-290`) does `mlx.Mul(m.LMHead, m.LMHeadGain)` every
token, which reads and writes the full `[65540, 5120]` bf16 table (671 MB each way)
before the matmul reads it a third time. The reference does the same
(`~/Documents/AI/talkie/src/talkie/mlx/model.py`, last lines of `__call__`), so the port
inherited it. Multiply once in `LoadWeights` (after `m.LMHead`/`m.LMHeadGain` are read,
`talkie.go:172-177`), `mlx.Eval` it so `base.Weights` pins the product, and have
`Unembed` use it. Scaling the logits after the matmul instead is equally cheap (1.54 vs
1.51 ms measured) but rounds differently; folding into the table reproduced the bf16
logits exactly here.

**Evidence.** `lmhead.py`: matmul with pre-folded gain 1.51 ms, with the per-token Mul
4.35 ms. `variants.py` V3b (fold alone on the reference): 62.7 → 60.0 ms, max logit
difference 0.0000, 60/60 greedy tokens identical.

**Gain** −2.8 ms/token (measured). **Effort** trivial. **Risk** none measured;
memory unchanged (the product replaces the raw table if the raw handle is dropped).
**Scope** talkie only; no other port in `x/models` carries a scalar head gain.

### 2. Weightless `fast.rms_norm`

**What.** `rmsNorm` (`talkie.go:234-240`) upcasts to fp32, squares, means, adds eps,
sqrts, divides, and casts back: 7 kernels, called on the embedding, twice per layer on
the residual, twice per layer on q and k, and once at the end, 162 times per token.
`mlx.RMSNormFn(x, nil, rmsEps)` (`x/mlxrunner/mlx/ops_extra.go:537`) already accepts a
nil weight, and MLX-C passes that through as `std::nullopt`
(`build/_deps/mlx-c-src/mlx/c/fast.cpp:542-556`), which is exactly the weightless form.
The kernel accumulates in fp32 internally, so the fp32 upcast the port does by hand is
preserved. The `nn.RMSNorm` type in `x/models/nn/nn.go:147-162` is not usable as-is
because it requires a weight, so call `mlx.RMSNormFn` directly.

**Evidence.** `variants.py` V0→V1: 62.7 → 60.7 ms, max logit difference 0.0000, 60/60
greedy tokens identical. V13 (this plus the fold): 57.5–57.8 ms in three runs.

**Gain** −2.1 ms/token (measured on the reference; inferred to transfer to the fork,
which builds the same graph). **Effort** one line. **Risk** the equality is empirical
(rsqrt-multiply vs sqrt-divide could differ in the last fp32 ulp before the bf16 cast;
it did not on 60 tokens × 65540 logits, but rerun the seeded smoke test in the README
after the change). **Scope** talkie only; the other ports already use the fused norm.

### 3. int8 quantization

**What.** Halve the bytes per token. This is the only lever that moves the floor, since
the floor is bandwidth.

**Evidence.** `quant.py` on the reference, with items 1 and 2 applied so only the
quantization differs (bf16 baseline in that harness: 58.3 ms):

| format                                          | linear bytes | ms/token | tok/s | mean KL vs bf16 (max) | top-1 | greedy text                               |
| ----------------------------------------------- | ------------ | -------- | ----- | --------------------- | ----- | ----------------------------------------- |
| bf16                                            | 25.89 GB     | 58.3     | 17.2  | 0                     | 100%  | reference                                 |
| int8, group 64, affine                          | 13.75 GB     | 33.5     | 29.9  | 0.017 (0.25)          | 95.0% | in period, coherent                       |
| int4 with attn_key/attn_value/mlp_resid at int8 | 9.73 GB      | 25.3     | 39.5  | 0.099 (0.80)          | 86.7% | in period, coherent                       |
| int4, group 64, affine                          | 7.28 GB      | 20.3     | 49.3  | 0.85 (8.5)            | 73.3% | loops ("the greatest step forward..." ×4) |
| mxfp4, group 32                                 | 6.88 GB      | 19.5     | 51.3  | 0.92 (9.9)            | 64.2% | loops                                     |

KL is teacher-forced over the 120-token bf16 greedy continuation of the README prompt
(`What might the wireless telephone one day become?`), so it measures how much the
quantized model disagrees with bf16 on bf16's own path. The mean log-prob bf16 assigns
its own tokens is −0.221; int8 assigns them −0.247, int4 −1.14.

**Through the fork's create path.** `x/create/create.go:171-199` (`ShouldQuantize`)
and `:263-323` (`GetTensorQuantization`) would treat talkie as follows, by reading the
rules against the tensor names in the checkpoint:

- every `blocks.N.*.weight` is 2-D with last dimension 5120 or 13696, both divisible by
  64, 32 and 16, so all five types (`int8`, `int4`, `mxfp8`, `mxfp4`, `nvfp4`) align;
- `embed.weight` and `blocks.N.embed_skip.a_g` are skipped by the `embed` substring
  rule; `attn_gain.a_g`, `mlp_gain.a_g`, `head_gain.head_g` are skipped because they
  lack the `.weight` suffix;
- `lm_head` also lacks the suffix, so it stays bf16 (671 MB, 1.4 ms/token). That is
  consistent with `talkie.go:172` reading `tensors["lm_head"]` raw, which would break
  if it were quantized; folding the gain (item 1) keeps working;
- the int4 rule that promotes `k_proj`/`v_proj`/`down_proj` to 8-bit (`create.go:314-320`)
  does not match `attn_key`/`attn_value`/`mlp_resid`, so `--quantize int4` would be the
  uniform 4-bit row of the table above, not the mixed one. Getting the mixed result
  needs a talkie policy (the pattern is `x/create/gemma4.go:61` and siblings: an
  arch-specific `quantizePolicy` that falls through to `GetTensorQuantization`).

The runtime side already loads it: `LoadWeights` goes through `model.NewLinearFactory`
(`talkie.go:166`), which returns `nn.QuantizedLinear` whenever `<name>.weight_scale`
exists (`x/mlxrunner/model/linear.go:62-95`), and `newModel` reads `root.QuantType()`
(`talkie.go:130-138`). Not verified by running, because `ollama create` was off limits.

Command, unverified: a Modelfile with `FROM <the safetensors directory used for the
original import>` (it must contain the `tokenizer.json` that `convert_tokenizer.py`
produces; `~/models/talkie-1930-13b-it-mlx` alone has only `vocab.txt`), then

```sh
OLLAMA_HOST=127.0.0.1:11435 ~/Documents/dev/ollama/ollama create talkie-1930-int8 \
  --quantize int8 -f Modelfile.int8
```

`int8`, `q8` and `fp8` are aliases (`x/quant/quant.go`, `Canonical`). Whether `FROM
talkie-1930` (re-quantizing a model already in the store) works on the MLX path was not
checked; the safetensors directory route is the one the code clearly supports
(`x/create/classify.go:53-55`).

**Gain** ~+80% tok/s (measured on the reference; the fork should land near 33 ms too,
since the non-GEMV overhead of items 1–2 is the same in either precision). Resident
size drops to about 15 GB, so bf16 and int8 can be loaded side by side for A/B
listening. **Effort** an import and a Modelfile. **Risk** quality: 0.017 nats mean KL is
small, but this is a voice model and the divergences are in word choice, so listen to it
(the int8 greedy text went to trains and a general's field telephone where bf16 went to
trains and ships; both are 1930). The README's seeded exact-match test becomes
meaningless across precisions; see "Verification" below. **Scope** all MLX-runner models;
the policy gap for K/V/down is talkie-specific.

### 4. `fast.rope` with `scale = -1` (not bit-exact)

**What.** talkie's rotation is `y1 = x1 cos + x2 sin; y2 = -x1 sin + x2 cos`, which is
standard NeoX RoPE at angle −θ. MLX's kernel computes positions as
`(arange + offset) * scale` (`build/_deps/mlx-src/mlx/fast.cpp:551-571`), so
`mlx.RoPEWithBase(q, int(m.HeadDim), false, m.RopeBase, -1.0, positions)` on the
`[B, H, L, D]` layout (`x/mlxrunner/mlx/ops_extra.go:429`) produces the same rotation,
replacing the 2 gathers, 4 multiplies, negate, 2 adds and concatenate in
`talkie.go:244-255`, twice per layer. It needs q and k transposed before RoPE rather
than after (`talkie.go:321-327`), as `llama.go:288-298` does. Positions per row are
`b.SeqOffsets`, which the kernel extends by 0..L−1 itself, so the `posIdx` array
(`talkie.go:262-269`) and the bf16 tables (`talkie.go:213-230`) go away.

**Evidence.** `variants.py` V1→V2: 60.7 → 59.5 ms. Max logit difference 0.25, greedy
diverges after 41 tokens. The difference is rounding, not sign: the kernel computes the
angles in fp32 while the port and reference multiply by bf16-rounded cos/sin tables, so
the kernel is the more accurate of the two.

**Gain** −1.1 ms/token (measured). **Effort** small, but touches the layout order.
**Risk** the seeded smoke test would stop matching and would need re-baselining, and
"different from the PyTorch reference by bf16 rounding" is a harder thing to verify
than "identical". **Scope** talkie only.

### 5. Compile the elementwise tails (not bit-exact)

**What.** The residual chain `x + attn·g_a; x + mlp·g_m; x + eX·g_e` (`talkie.go:308-311`)
and `silu(gate)·linear` (`talkie.go:343-345`, deliberately kept as primitives) could be
`mlx.Compile` kernels like `mlx.SwiGLU` (`x/mlxrunner/mlx/act.go`). `mlx.EnableCompile()`
is already on (`runner.go:146`).

**Evidence.** `variants.py` V4→V5: −1.4 ms, but max logit difference 0.5 and greedy
diverges after 7 tokens, because the compiled kernel keeps intermediates in registers
instead of rounding each to bf16 as the reference does.

**Gain** −1.4 ms. **Risk** same class as item 4 but a larger numeric change. Not
recommended unless the port stops promising to mirror the reference.

### 6. Speculative decoding: what it would take

The runner speculates only with a `base.DraftModel` (`x/mlxrunner/model/base/base.go:37`):
either a separate draft under `root.Draft` in the manifest (`base.NewDraft`,
`base.go:136`) or a `SelfDraft` head inline with the target. talkie has neither; there is
no n-gram or prompt-lookup drafter in `x/mlxrunner`. So the options are training an MTP
head or a small DFlash-style block drafter (`x/mlxrunner/dflash.go`) on 1930 text with
talkie's tokenizer, or a small talkie-vocab draft model, and writing the draft port.

The economics on this machine: a verify step over k+1 tokens costs about the same as one
token (weights stream once) until the GEMMs become compute-bound, so the gain is
acceptance-rate-bound, and the earlier measurement of −37% with `draft_num_predict` on
another model shows what a low-acceptance drafter costs when it also has to stream its
own weights. For open-ended period prose there is no obvious source of high-acceptance
drafts. Rate this as a research project, not a tuning knob, and expect int8 to deliver
more for far less.

## Not worth doing

- **Fusing Q/K/V into one `[15360, 5120]` and gate/linear into `[27392, 5120]`.** The
  bare GEMV floor does improve, 55.4 → 53.9 ms (`floor.py` MODE=fused, 480 GB/s), but
  in the real forward the split of the fused output costs it back: V4b measured 62.69 vs
  62.65 ms (`variants.py`), no gain, and it doubles peak memory during load (44.8 GB
  peak in the harness). Leave the weights as they are.
- **Hoisting the cos/sin gathers out of the layer loop** (`talkie.go:246-247` gather
  the tables 4 times per layer). V13h measured −0.08 ms; the gathers are 64-element
  rows. If RoPE is touched at all, item 4 is the change.
- **The sampler.** Fork measurements (`fork_sampling.py`): greedy 60.2–60.7 ms,
  ollama defaults (temperature 0.8, top_k 40, top_p 0.9) 61.3–61.6, plus
  repeat_penalty 1.1 61.2–61.4, greedy again 61.2. The differences are inside the
  run-to-run drift. Sampling is fused onto the forward's chain and evaluated
  asynchronously (`pipeline.go:465-470`, `sample.go:850-857`), so it never stalls the
  GPU. Note the default `repeat_penalty` is 1.0 (`api/types.go:1147`), so the history
  ring is off unless a request turns it on.
- **KV cache layout.** `KVCache.appendKV` (`x/mlxrunner/cache/kvcache.go:51-92`) grows
  in 256-slot steps and writes with `SliceUpdate` into a buffer whose Go handle is
  replaced by `Set`, which leaves MLX free to donate the buffer in place. The context
  test (`fork_ctx.py`) shows decode growing 2.2 ms per 1000 tokens of context, against
  1.75 ms for the pure KV read at 467 GB/s; a whole-buffer copy per layer would have
  doubled that (inferred). At L=1 SDPA drops the causal flag and takes the no-mask
  fast path (`x/models/nn/sdpa.go:180-184`). Nothing to do here.
- **Prefill.** Cache-busting prompts measured 494–514 tok/s at 550 tokens and 444–509
  at 1211 tokens, about 13.7 TFLOPS achieved, a reasonable fraction of what this GPU
  does in bf16 matmul. The hand-rolled norms and RoPE are a smaller fraction of a
  prefill step than of a decode step. Fixed cost for a 23-token prompt was 126–326 ms.
  Chunk size is 2048 (`pipeline.go:22-24`), equal to `max_seq_len`, so a prompt is one
  chunk. Not a lever for a chat-length prompt.
- **The 2026-09-12 array-lifetime rework** (commits 13037ecb, aeb8f711, ba064c36,
  b68b112b). `New` appends to a scope slice and `remove` scans from the end
  (`x/mlxrunner/mlx/scope.go:174-203`); around 3000 arrays per token is microseconds of
  Go, off the GPU's critical path thanks to the pipelined decoder. `mlxCheck` after
  every binding call (ba064c36) is a buffer read per op, same order. The fork's number
  today (16.50 tok/s) is slightly above the 16.28 recorded before those commits landed,
  so nothing regressed.
- **`mlx.ClearCache()` every 256 tokens** (`pipeline.go:282, 351`). One allocator
  refill every 256 tokens; invisible in the 200-token runs and would be under 0.1% in
  longer ones.
- **Uniform int4 or mxfp4.** Fast (49–51 tok/s) and wrong: KL 0.85–0.92 with degenerate
  repetition in the greedy sample. This model has no learned norm gains and a 65540
  vocabulary; do not ship it at 4 bits without the K/V/down promotion, and even then
  (KL 0.10) it needs a listening test.
- **Chasing the last 8 ms to 546 GB/s.** 467 GB/s on GEMV is what this GPU delivers;
  the gap is not software.

## Verification after each kind of change

- Item 1 (exact): the README's seeded test (`temperature 0.75, seed 1930`,
  "What might the wireless telephone one day become?") must reproduce the text from before
  the change word for word; if it does not, treat it as a bug.
- Item 2 (changes the port's numerics to match the reference): this re-baselined the
  seeded text on 2026-09-12, after `norm_truth.py` showed the new output is the reference's.
  The README carries the new text. Any later change to the norm must reproduce it.
- Items 4 and 5 (rounding changes): re-baseline the seeded test after confirming the
  new logits sit within bf16 rounding of the old (the `variants.py` check: max |Δlogit|
  ≤ 0.5 on the first token, and the divergence point on a greedy run), and re-read the
  output for period voice.
- Item 3 (quantized): the seeded exact-match test cannot apply across precisions.
  Replace it with the `quant.py` method: teacher-forced mean KL and top-1 agreement
  against bf16 over a fixed continuation (int8 here: 0.017 nats and 95%; flag anything
  above about 0.05 or below 90%), plus the period-voice reading, plus a seeded
  same-precision determinism check (the quantized model against itself across rebuilds,
  which is the exact-match test re-baselined per precision).

## Commands and scripts

All scratch scripts lived in the session scratchpad and are reproduced here in full.
The reference-side ones run in the talkie venv, which has `mlx` and the `talkie`
package; unload the fork's copy first and confirm both `/api/ps` are empty.

```sh
# preflight for every measurement
curl -s http://127.0.0.1:11434/api/ps; curl -s http://127.0.0.1:11435/api/ps
# unload the fork's copy before any reference-side script
curl -s http://127.0.0.1:11435/api/generate -d '{"model":"talkie-1930","keep_alive":0}'

# fork decode baseline (16.50 tok/s today)
HOST=http://127.0.0.1:11435 PROMPT="Write a long and detailed essay upon the future of the railways." \
  .agents/skills/ollama-bench/scripts/bench.sh talkie-1930

# fork prefill and context scaling, fork sampler paths (fork must be up; it reloads the model)
python3 fork_ctx.py
python3 fork_sampling.py

# reference side
cd ~/Documents/AI/talkie
MODE=separate .venv/bin/python floor.py
MODE=fused    .venv/bin/python floor.py
.venv/bin/python lmhead.py
.venv/bin/python variants.py                       # all variants, ~45 GB peak because of the fused ones
.venv/bin/python variants.py "V0 reference" "V13 rms_norm+fold (exact)"   # ~27 GB peak
BITS=16 .venv/bin/python quant.py                  # writes ref16.npz next to the script
BITS=8  .venv/bin/python quant.py
BITS=4  .venv/bin/python quant.py
BITS=4 PROMOTE=attn_key,attn_value,mlp_resid .venv/bin/python quant.py
BITS=4 GROUP=32 QMODE=mxfp4 .venv/bin/python quant.py
```

Raw output from today's runs, for the record:

```
bench.sh:            16.50 / 16.50 / 16.51 tok/s (200 tok, seed 42)
fork_ctx.py:         short  23 tok: prefill 326/126 ms, gen 60.6/60.7 ms/tok
                     mid   550 tok: prefill 1071/1118 ms (514/494 tok/s), gen 62.0/62.4 ms/tok
                     long 1211 tok: prefill 2377/2730 ms (509/444 tok/s), gen 63.9/63.5 ms/tok
fork_sampling.py:    greedy 60.15/60.65; defaults 61.31/61.55; +penalty 61.44/61.24; greedy 61.22/61.18 ms/tok
floor.py separate:   25.89 GB, 281 gemv, 55.41 ms/step, 18.05 tok/s, 467 GB/s
floor.py fused:      25.89 GB, 161 gemv, 53.90 ms/step, 18.55 tok/s, 480 GB/s
lmhead.py:           folded 1.51 ms, per-token Mul 4.35 ms (+2.84), matmul-then-scale 1.54 ms
variants.py:         V0 62.74/62.65/62.32 | V1 60.65 exact | V2 59.52 (0.25, 41/60) | V3 56.70
                     V4 57.14 | V5 55.78 (0.5, 7/60) | V3b 60.04 exact | V4b 62.69 exact
                     V13 57.77/57.45 exact | V13h 57.37 exact
quant.py:            bf16 58.28 | int8 33.46-33.59, KL 0.0169 (0.246), top1 95.0%
                     int4 20.27, KL 0.8496 (8.498), top1 73.3% | mxfp4 19.48, KL 0.9152 (9.855), top1 64.2%
                     int4+k/v/down int8 25.32, KL 0.0991 (0.801), top1 86.7%
```

### norm_truth.py

Settles which RMSNorm formula is correct by running the reference model three ways on the
smoke-test prompt. The 12 token ids are the reference's `format_prompt` encoding, which
matches the fork's template token for token. Run it from the talkie venv:

```sh
cd ~/Documents/AI/talkie && .venv/bin/python norm_truth.py
```

```python
"""Which RMSNorm formula matches the reference? Run the reference MLX model on the
smoke-test prompt with three norms swapped in and print first-token top-5 and a
40-token greedy continuation for each."""

import mlx.core as mx
import talkie.mlx.model as tm
from talkie.mlx.generate import MLXTalkie

IDS = [65537, 4323, 1227, 260, 29313, 9462, 494, 850, 2134, 63, 65536, 65538]
EPS = mx.finfo(mx.float32).eps
t = MLXTalkie("/Users/eeaglstun/models/talkie-1930-13b-it-mlx")


def fast(x):
    return mx.fast.rms_norm(x, None, EPS)


def fork_old(x):  # the fork's pre-fix formula: sqrt then divide, cast to bf16
    xf = x.astype(mx.float32)
    ms = mx.mean(xf * xf, axis=-1, keepdims=True)
    return (xf / mx.sqrt(ms + EPS)).astype(mx.bfloat16)


VARIANTS = {"reference native (rsqrt*)": tm._rms_norm, "mx.fast.rms_norm": fast, "fork old (sqrt/)": fork_old}
first = {}
for name, fn in VARIANTS.items():
    tm._rms_norm = fn
    logits, cache = t.model(mx.array([IDS], dtype=mx.int32))
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp, cache)
    first[name] = lp
    top = mx.argsort(-lp[0])[:5].tolist()
    print(f"\n== {name}")
    print("   " + "  ".join(f"{t.tokenizer.decode([i])!r} {lp[0, i].item():.4f}" for i in top))
    out, tok = [], top[0]
    for _ in range(40):
        out.append(tok)
        logits, cache = t.model(mx.array([[tok]], dtype=mx.int32), cache)
        tok = int(mx.argmax(logits[0]).item())
    print("   greedy:", t.tokenizer.decode(out))

names = list(VARIANTS)
for a in names:
    for b in names:
        if a < b:
            print(f"max |Δ first-token logprob| {a} vs {b}: {mx.abs(first[a] - first[b]).max().item():.4f}")
```

Output on 2026-09-12:

```
== reference native (rsqrt*)
   'The' -0.4027  'It' -1.4027  'By' -3.8402  'A' -4.7777  'What' -4.8402
   greedy: The wireless telephone may become a means of personal communication between moving trains and fixed stations, and between ships at sea. It may enable a passenger in an express train to speak to his friends at the place

== mx.fast.rms_norm
   'The' -0.4027  'It' -1.4027  'By' -3.8402  'A' -4.7777  'What' -4.8402
   greedy: The wireless telephone may become a means of personal communication between moving trains and fixed stations, and between ships at sea. It may enable a passenger in an express train to speak to his friends at the place

== fork old (sqrt/)
   'It' -0.6825  'The' -0.9325  'By' -3.4950  'A' -4.1825  'In' -5.3075
   greedy: It might become a means of communication between moving trains and fixed stations. It might enable a traveller to speak from a train to a friend at home. It might enable a general to superintend the operations of
max |Δ first-token logprob| mx.fast.rms_norm vs reference native (rsqrt*): 0.0000
max |Δ first-token logprob| fork old (sqrt/) vs reference native (rsqrt*): 1.5327
max |Δ first-token logprob| fork old (sqrt/) vs mx.fast.rms_norm: 1.5327
```

### fork_ctx.py

```python
"""Prefill and context-length scaling on the fork (port 11435).

Each request carries a fresh nonce so the prefix cache cannot serve it, which
is the only way to see prefill through bench.sh's blind spot. Reports
prompt_eval (prefill) tok/s and eval (decode) tok/s at short and long context.
"""
import json, sys, time, urllib.request, uuid

HOST = "http://127.0.0.1:11435"
MODEL = "talkie-1930"
PARA = ("The railway, which in the last century transformed the commerce of "
        "nations and the habits of the people, stands to-day upon the threshold "
        "of a further change. Electrification, the motor omnibus, and the "
        "aeroplane each press upon it from a different quarter, and the wise "
        "director will consider not whether his line shall alter, but how. ")

def gen(prompt, num_predict=200):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"num_predict": num_predict, "seed": 42}}).encode()
    req = urllib.request.Request(HOST + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    wall = time.perf_counter() - t0
    pc, pd = d["prompt_eval_count"], d["prompt_eval_duration"] / 1e9
    ec, ed = d["eval_count"], d["eval_duration"] / 1e9
    return pc, pd, ec, ed, wall

def run(label, reps, num_predict=200):
    for r in range(reps):
        prompt = f"Reference {uuid.uuid4().hex[:8]}. " + PARA * reps_para + \
                 "Write a long and detailed essay upon the future of the railways."
        pc, pd, ec, ed, wall = gen(prompt, num_predict)
        print(f"{label:14s} run {r+1}: prompt {pc:5d} tok in {pd*1000:7.0f} ms "
              f"({pc/pd if pd else 0:7.1f} tok/s) | gen {ec} tok "
              f"{ec/ed:6.2f} tok/s ({ed/ec*1000:5.1f} ms/tok) | wall {wall:.1f}s",
              flush=True)

for label, reps_para in (("short", 0), ("mid ~700tok", 8), ("long ~1500tok", 18)):
    run(label, 2)
```

### fork_sampling.py

```python
"""Cost of the fork's sampler paths, measured on port 11435.

Same prompt and 200 tokens each; a warmup request first so load and prefill
are out of the way. Three option sets: greedy (argmax only), ollama defaults
(temperature 0.8, top_k 40, top_p 0.9, no penalty), and defaults plus
repeat_penalty 1.1 (turns on the history ring and the penalty gather/scatter).
"""
import json, time, urllib.request

HOST = "http://127.0.0.1:11435"
MODEL = "talkie-1930"
PROMPT = "Write a long and detailed essay upon the future of the railways."

def gen(options):
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "stream": False,
                       "options": {"num_predict": 200, "seed": 42, **options}}).encode()
    req = urllib.request.Request(HOST + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["eval_count"], d["eval_duration"] / 1e9

gen({})
for label, opts in (("greedy temp=0", {"temperature": 0}),
                    ("defaults", {}),
                    ("defaults+repeat_penalty 1.1", {"repeat_penalty": 1.1}),
                    ("greedy temp=0 (again)", {"temperature": 0})):
    for r in range(2):
        ec, ed = gen(opts)
        print(f"{label:30s} run {r+1}: {ec} tok {ec/ed:6.2f} tok/s ({ed/ec*1000:5.2f} ms/tok)", flush=True)
```

### floor.py

```python
"""Weight-streaming floor for talkie on this machine.

Runs only the matvecs a decode token needs (every linear weight in every
layer plus lm_head), nothing else, and times them. That is the real
bandwidth floor for this model on this GPU, as opposed to the paper figure
546 GB/s / 26.6 GB. MODE=fused concatenates Q/K/V and gate/linear so the
GEMV count drops from 7 to 4 per layer.

    cd ~/Documents/AI/talkie && MODE=separate .venv/bin/python floor.py
    cd ~/Documents/AI/talkie && MODE=fused    .venv/bin/python floor.py
"""
import glob, os, time
import mlx.core as mx

MODEL_DIR = os.path.expanduser("~/models/talkie-1930-13b-it-mlx")
MODE = os.environ.get("MODE", "separate")
STEPS = int(os.environ.get("STEPS", 30))

w = {}
for f in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
    w.update(mx.load(f))
mx.eval(*w.values())
L = 40
lin = []
for i in range(L):
    p = f"blocks.{i}"
    if MODE == "fused":
        qkv = mx.concatenate([w[f"{p}.attn.attn_query.weight"], w[f"{p}.attn.attn_key.weight"], w[f"{p}.attn.attn_value.weight"]], axis=0)
        gu = mx.concatenate([w[f"{p}.mlp.mlp_gate.weight"], w[f"{p}.mlp.mlp_linear.weight"]], axis=0)
        mx.eval(qkv, gu)
        for k in ("attn.attn_query", "attn.attn_key", "attn.attn_value", "mlp.mlp_gate", "mlp.mlp_linear"):
            del w[f"{p}.{k}.weight"]
        lin.append([qkv, w[f"{p}.attn.attn_resid.weight"], gu, w[f"{p}.mlp.mlp_resid.weight"]])
    else:
        lin.append([w[f"{p}.attn.attn_query.weight"], w[f"{p}.attn.attn_key.weight"], w[f"{p}.attn.attn_value.weight"],
                    w[f"{p}.attn.attn_resid.weight"], w[f"{p}.mlp.mlp_gate.weight"], w[f"{p}.mlp.mlp_linear.weight"],
                    w[f"{p}.mlp.mlp_resid.weight"]])
lm = w["lm_head"]
total_bytes = sum(a.nbytes for layer in lin for a in layer) + lm.nbytes
print(f"mode={MODE} streamed bytes/token = {total_bytes/1e9:.2f} GB, gemv count/token = {sum(len(l) for l in lin)+1}")

x5120 = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
x13696 = mx.random.normal((1, 1, 13696)).astype(mx.bfloat16)

def step():
    acc = None
    for layer in lin:
        for wt in layer:
            x = x5120 if wt.shape[1] == 5120 else x13696
            y = mx.matmul(x, wt.T)
            s = y.sum()
            acc = s if acc is None else acc + s
    acc = acc + mx.matmul(x5120, lm.T).sum()
    return acc

for _ in range(3):
    mx.eval(step())
t0 = time.perf_counter()
for _ in range(STEPS):
    mx.eval(step())
dt = (time.perf_counter() - t0) / STEPS
print(f"{dt*1000:.2f} ms/step -> {1/dt:.2f} tok/s ceiling, effective {total_bytes/dt/1e9:.0f} GB/s")

# Same thing with the per-token lm_head gain multiply the port and the
# reference both do (Mul over the full [65540, 5120] table every token).
gain = mx.array([1.0], dtype=mx.bfloat16)
def step_gain():
    return step() + mx.matmul(x5120, (lm * gain).T).sum()
for _ in range(2):
    mx.eval(step_gain())
t0 = time.perf_counter()
for _ in range(STEPS):
    mx.eval(step_gain())
dt2 = (time.perf_counter() - t0) / STEPS
print(f"with per-token lm_head*gain: {dt2*1000:.2f} ms/step (+{(dt2-dt)*1000:.2f} ms)")
print(f"peak mem {mx.get_peak_memory()/1e9:.1f} GB")
```

(The `step_gain` tail of this script gave +26.1 ms in the separate run and +5.8 ms in
the fused run; it double-counts the lm_head matmul and was noisy, so `lmhead.py` below
is the number used.)

### lmhead.py

```python
"""Isolated cost of the per-token `lm_head * gain` the port does in Unembed.

    cd ~/Documents/AI/talkie && .venv/bin/python lmhead.py
"""
import os, time
import mlx.core as mx

MODEL_DIR = os.path.expanduser("~/models/talkie-1930-13b-it-mlx")
w = mx.load(os.path.join(MODEL_DIR, "model-00007-of-00007.safetensors"))
lm = w.get("lm_head")
if lm is None:
    import glob
    for f in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
        w = mx.load(f)
        if "lm_head" in w:
            lm = w["lm_head"]; break
gain = w.get("lm_head_gain.w_g", mx.array([1.0], dtype=mx.bfloat16))
mx.eval(lm, gain)
x = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
folded = lm * gain
mx.eval(folded)

def bench(fn, n=100):
    for _ in range(10): mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1000

a = bench(lambda: x @ folded.T)
b = bench(lambda: x @ (lm * gain).T)
c = bench(lambda: (x @ lm.T) * gain)
print(f"lm_head {lm.shape} {lm.dtype} = {lm.nbytes/1e6:.0f} MB")
print(f"matmul with pre-folded gain : {a:.2f} ms")
print(f"matmul with per-token Mul   : {b:.2f} ms  (+{b-a:.2f} ms/token, the port's Unembed)")
print(f"matmul then scale logits    : {c:.2f} ms")
```

### variants.py

```python
"""Decode-step cost of talkie forward variants, on the reference weights.

The Go port mirrors the MLX reference op for op, so each variant here is a
stand-in for the same change in x/models/talkie/talkie.go. Every variant is
timed on a 60-token decode after a short prompt (KV cache live), GPU argmax
feeding back the next token so sampling is out of the picture. Numerics are
checked against V0 on the same prompt: max |logit diff| and how many greedy
tokens agree before the first divergence.

    cd ~/Documents/AI/talkie && .venv/bin/python variants.py [names...]
"""
import math, os, sys, time
import mlx.core as mx
from talkie.chat import format_prompt
from talkie.mlx.generate import MLXTalkie
from talkie.mlx import model as ref

MODEL_DIR = os.path.expanduser("~/models/talkie-1930-13b-it-mlx")
PROMPT = format_prompt("What might the wireless telephone one day become?")
STEPS = int(os.environ.get("STEPS", 60))
EPS = float(mx.finfo(mx.float32).eps)

talkie = MLXTalkie(MODEL_DIR)
m = talkie.model
c = m.config
W = m.weights
H, D, E = c.n_head, c.head_dim, c.n_embd
prompt_ids = talkie.tokenizer.encode(PROMPT, allowed_special="all")

# --- building blocks --------------------------------------------------------

def rms_ref(x):
    return ref._rms_norm(x)

def rms_fast(x):
    return mx.fast.rms_norm(x, None, EPS)

def rope_ref(x, offset):
    return m._apply_rope(x, offset)

def rope_fast(x, offset):
    # x: [B, T, H, D] -> kernel wants [B, H, T, D]. scale=-1 flips the sign
    # of every angle, which is exactly talkie's y2 = -x1 sin + x2 cos.
    y = mx.fast.rope(x.transpose(0, 2, 1, 3), D, traditional=False,
                     base=c.rope_base, scale=-1.0, offset=offset)
    return y.transpose(0, 2, 1, 3)

@mx.compile
def swiglu(gate, lin):
    return gate * mx.sigmoid(gate) * lin

@mx.compile
def resid_gain(x, y, g):
    return x + y * g

@mx.compile
def resid_gain2(x, mlp, mg, ex, eg):
    return x + mlp * mg + ex * eg

def rope_hoisted(cos, sin):
    # Same arithmetic as the reference, but cos/sin were gathered once per
    # forward (talkie.go gathers them per layer for q and again for k).
    def f(x, offset):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return mx.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1).astype(x.dtype)
    return f

def build(name, rms, rope, fold_head, fused, compiled):
    hoist = rope == "hoist"
    lm = W["lm_head"] * W["lm_head_gain.w_g"] if fold_head else None
    if fold_head:
        mx.eval(lm)
    layers = []
    for i in range(c.n_layer):
        p = f"blocks.{i}"
        d = dict(
            hg=W[f"{p}.attn.head_gain.head_g"].reshape(1, 1, H, 1),
            ag=W[f"{p}.attn_gain.a_g"], mg=W[f"{p}.mlp_gain.a_g"], eg=W[f"{p}.embed_skip.a_g"],
            o=W[f"{p}.attn.attn_resid.weight"], down=W[f"{p}.mlp.mlp_resid.weight"],
        )
        if fused:
            d["qkv"] = mx.concatenate([W[f"{p}.attn.attn_query.weight"], W[f"{p}.attn.attn_key.weight"], W[f"{p}.attn.attn_value.weight"]], axis=0)
            d["gu"] = mx.concatenate([W[f"{p}.mlp.mlp_gate.weight"], W[f"{p}.mlp.mlp_linear.weight"]], axis=0)
            mx.eval(d["qkv"], d["gu"])
        else:
            d["q"], d["k"], d["v"] = (W[f"{p}.attn.attn_{n}.weight"] for n in ("query", "key", "value"))
            d["gate"], d["lin"] = W[f"{p}.mlp.mlp_gate.weight"], W[f"{p}.mlp.mlp_linear.weight"]
        layers.append(d)

    def attention(d, x, cache, offset):
        B, T, _ = x.shape
        if fused:
            qkv = x @ d["qkv"].T
            q, k, v = mx.split(qkv, 3, axis=-1)
        else:
            q, k, v = x @ d["q"].T, x @ d["k"].T, x @ d["v"].T
        q = q.reshape(B, T, H, D); k = k.reshape(B, T, H, D); v = v.reshape(B, T, H, D)
        q = rms(rope(q, offset)); k = rms(rope(k, offset))
        q = q * d["hg"]
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2); v = mx.concatenate([cache[1], v], axis=2)
            mask = None if T == 1 else "causal"
        else:
            mask = "causal"
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(D), mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, E)
        return y @ d["o"].T, (k, v)

    def mlp(d, x):
        if fused:
            gu = x @ d["gu"].T
            gate, lin = mx.split(gu, 2, axis=-1)
        else:
            gate, lin = x @ d["gate"].T, x @ d["lin"].T
        h = swiglu(gate, lin) if compiled else (gate * mx.sigmoid(gate)) * lin
        return h @ d["down"].T

    def forward(ids, cache):
        nonlocal rope
        offset = 0 if cache is None else cache[0][0].shape[2]
        if hoist:
            T = ids.shape[1]
            rope = rope_hoisted(m._cos[:, offset:offset + T], m._sin[:, offset:offset + T])
        x = rms(W["embed.weight"][ids]); ex = x
        new = []
        for i, d in enumerate(layers):
            a, kv = attention(d, rms(x), None if cache is None else cache[i], offset)
            if compiled:
                x = resid_gain(x, a, d["ag"])
                x = resid_gain2(x, mlp(d, rms(x)), d["mg"], ex, d["eg"])
            else:
                x = x + a * d["ag"]
                x = x + mlp(d, rms(x)) * d["mg"]
                x = x + ex * d["eg"]
            new.append(kv)
        x = rms(x)
        head = lm if fold_head else W["lm_head"] * W["lm_head_gain.w_g"]
        return (x[:, -1, :] @ head.T).astype(mx.float32), new
    return forward

VARIANTS = {
    "V0 reference":                (rms_ref,  rope_ref,  False, False, False),
    "V1 +fast.rms_norm":           (rms_fast, rope_ref,  False, False, False),
    "V2 +fast.rope(scale=-1)":     (rms_fast, rope_fast, False, False, False),
    "V3 +fold lm_head gain":       (rms_fast, rope_fast, True,  False, False),
    "V4 +fused qkv/gate-up":       (rms_fast, rope_fast, True,  True,  False),
    "V5 +compiled elementwise":    (rms_fast, rope_fast, True,  True,  True),
    "V13 rms_norm+fold (exact)":   (rms_fast, rope_ref,  True,  False, False),
    "V13h +hoisted cos/sin (exact)": (rms_fast, "hoist",  True,  False, False),
    "V3b fold only (on V0)":       (rms_ref,  rope_ref,  True,  False, False),
    "V4b fused only (on V0)":      (rms_ref,  rope_ref,  False, True,  False),
}
want = sys.argv[1:] or list(VARIANTS)

def decode(forward, steps):
    ids = mx.array([prompt_ids], dtype=mx.int32)
    logits, cache = forward(ids, None)
    mx.eval(logits, cache)
    first = logits
    tok = mx.argmax(logits, axis=-1).astype(mx.int32)
    out = []
    mx.eval(tok)
    t0 = time.perf_counter()
    for _ in range(steps):
        logits, cache = forward(tok.reshape(1, 1), cache)
        tok = mx.argmax(logits, axis=-1).astype(mx.int32)
        mx.eval(tok, cache)
        out.append(tok.item())
    dt = (time.perf_counter() - t0) / steps
    return dt, first, out

base_logits = base_out = None
for name in want:
    fwd = build(name, *VARIANTS[name])
    decode(fwd, 5)  # warmup (compiles, allocator)
    dt, first, out = decode(fwd, STEPS)
    if base_logits is None:
        base_logits, base_out = first, out
        agree = "-"; diff = 0.0
    else:
        diff = mx.abs(first - base_logits).max().item()
        n = next((i for i, (a, b) in enumerate(zip(out, base_out)) if a != b), len(out))
        agree = f"{n}/{len(out)} greedy tokens match before divergence"
    print(f"{name:28s} {dt*1000:6.2f} ms/tok {1/dt:6.2f} tok/s | max|dlogit| vs V0 {diff:.4f} | {agree}", flush=True)
    del fwd
    mx.clear_cache()
print(f"peak mem {mx.get_peak_memory()/1e9:.1f} GB")
```

### quant.py

```python
"""Quantized talkie: speed and a quality spot-check, without touching ollama.

BITS=16 runs bf16 and writes the greedy continuation of the README prompt plus
its per-position log-probs to ref16.npz. BITS=8/4 (GROUP=64 default, MODE
affine) quantizes every linear weight and lm_head with mx.quantize, keeps
embed/gains in bf16, then reports decode tok/s, teacher-forced KL(bf16 ||
quant) and top-1 agreement over the bf16 continuation, and its own greedy text
for a read. bf16 is freed layer by layer so peak memory stays near 27 GB.

    cd ~/Documents/AI/talkie && BITS=16 .venv/bin/python quant.py
    cd ~/Documents/AI/talkie && BITS=8  .venv/bin/python quant.py
    cd ~/Documents/AI/talkie && BITS=4  .venv/bin/python quant.py
"""
import math, os, time
import numpy as np
import mlx.core as mx
from talkie.chat import format_prompt
from talkie.mlx.generate import MLXTalkie
from talkie.mlx import model as ref

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.expanduser("~/models/talkie-1930-13b-it-mlx")
BITS = int(os.environ.get("BITS", 16))
GROUP = int(os.environ.get("GROUP", 64))
QMODE = os.environ.get("QMODE", "affine")
N_GEN = int(os.environ.get("N_GEN", 120))
PROMPT = format_prompt("What might the wireless telephone one day become?")

talkie = MLXTalkie(MODEL_DIR)
m = talkie.model; c = m.config; W = m.weights
H, D, E = c.n_head, c.head_dim, c.n_embd
EPS = float(mx.finfo(mx.float32).eps)
prompt_ids = talkie.tokenizer.encode(PROMPT, allowed_special="all")

LINEAR_KEYS = [k for k in W if k.endswith(".weight") and W[k].ndim == 2 and k != "embed.weight"] + ["lm_head"]

# PROMOTE=attn_key,attn_value,mlp_resid keeps the named projections at 8-bit
# affine/g64 inside a 4-bit run, mirroring what create.go does for
# k_proj/v_proj/down_proj on stock archs (talkie's names miss that rule).
PROMOTE = [p for p in os.environ.get("PROMOTE", "").split(",") if p]

class Lin:
    def __init__(self, w, bits=BITS, group=GROUP, mode=QMODE):
        self.bits, self.group, self.mode = bits, group, mode
        if bits == 16:
            self.w = w; self.q = None
        else:
            self.q = mx.quantize(w, group, bits, mode=mode)
            mx.eval(*self.q)
    def __call__(self, x):
        if self.q is None:
            return x @ self.w.T
        wq, s = self.q[0], self.q[1]
        b = self.q[2] if len(self.q) > 2 else None  # mxfp4/mxfp8 have no biases
        return mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=self.group, bits=self.bits, mode=self.mode)

lin = {}
for k in LINEAR_KEYS:
    src = W[k] * W["lm_head_gain.w_g"] if k == "lm_head" else W[k]
    if BITS != 16 and any(p in k for p in PROMOTE):
        lin[k] = Lin(src, bits=8, group=64, mode="affine")
    else:
        lin[k] = Lin(src)
    if BITS != 16:
        del W[k]
mx.clear_cache()
qbytes = sum((sum(a.nbytes for a in l.q) if l.q else l.w.nbytes) for l in lin.values())
print(f"bits={BITS} group={GROUP} mode={QMODE}: linear+lm_head bytes {qbytes/1e9:.2f} GB, active mem {mx.get_active_memory()/1e9:.1f} GB", flush=True)

def rms(x): return mx.fast.rms_norm(x, None, EPS)

def forward(ids, cache):
    offset = 0 if cache is None else cache[0][0].shape[2]
    x = rms(W["embed.weight"][ids]); ex = x
    B, T, _ = x.shape
    new = []
    for i in range(c.n_layer):
        p = f"blocks.{i}"
        h = rms(x)
        q = lin[f"{p}.attn.attn_query.weight"](h).reshape(B, T, H, D)
        k = lin[f"{p}.attn.attn_key.weight"](h).reshape(B, T, H, D)
        v = lin[f"{p}.attn.attn_value.weight"](h).reshape(B, T, H, D)
        q = rms(m._apply_rope(q, offset)); k = rms(m._apply_rope(k, offset))
        q = q * W[f"{p}.attn.head_gain.head_g"].reshape(1, 1, H, 1)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        if cache is not None:
            k = mx.concatenate([cache[i][0], k], axis=2); v = mx.concatenate([cache[i][1], v], axis=2)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(D), mask=None if T == 1 else "causal")
        y = lin[f"{p}.attn.attn_resid.weight"](y.transpose(0, 2, 1, 3).reshape(B, T, E))
        x = x + y * W[f"{p}.attn_gain.a_g"]
        h = rms(x)
        g = lin[f"{p}.mlp.mlp_gate.weight"](h); u = lin[f"{p}.mlp.mlp_linear.weight"](h)
        x = x + lin[f"{p}.mlp.mlp_resid.weight"](g * mx.sigmoid(g) * u) * W[f"{p}.mlp_gain.a_g"]
        x = x + ex * W[f"{p}.embed_skip.a_g"]
        new.append((k, v))
    x = rms(x)
    return lin["lm_head"](x).astype(mx.float32), new

# --- speed: greedy decode after the prompt -----------------------------------
def greedy(n):
    ids = mx.array([prompt_ids], dtype=mx.int32)
    logits, cache = forward(ids, None); mx.eval(logits, cache)
    tok = mx.argmax(logits[:, -1], axis=-1).astype(mx.int32); mx.eval(tok)
    out = [tok.item()]
    t0 = time.perf_counter()
    for _ in range(n - 1):
        logits, cache = forward(tok.reshape(1, 1), cache)
        tok = mx.argmax(logits[:, -1], axis=-1).astype(mx.int32)
        mx.eval(tok, cache)
        out.append(tok.item())
    return out, (time.perf_counter() - t0) / (n - 1)

greedy(6)
out, dt = greedy(N_GEN)
print(f"decode {dt*1000:.2f} ms/tok = {1/dt:.2f} tok/s (greedy, GPU argmax)", flush=True)
text = talkie.tokenizer.decode(out)
print("greedy text:", repr(text[:600]), flush=True)

# --- quality: teacher-forced against the bf16 continuation -------------------
ref_path = os.path.join(HERE, "ref16.npz")
def teacher_logprobs(tokens):
    ids = mx.array([prompt_ids + tokens], dtype=mx.int32)
    logits, _ = forward(ids, None)
    lp = logits[0, len(prompt_ids) - 1: len(prompt_ids) - 1 + len(tokens)]
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    return np.array(lp, dtype=np.float32)

if BITS == 16:
    lp = teacher_logprobs(out)
    np.savez(ref_path, tokens=np.array(out, dtype=np.int32), logprobs=lp)
    print(f"saved {ref_path}")
else:
    r = np.load(ref_path)
    rt, rlp = [int(t) for t in r["tokens"]], r["logprobs"]
    qlp = teacher_logprobs(rt)
    kl = np.sum(np.exp(rlp) * (rlp - qlp), axis=-1)
    top1 = np.mean(np.argmax(rlp, -1) == np.argmax(qlp, -1))
    chosen = qlp[np.arange(len(rt)), rt]
    print(f"vs bf16 over {len(rt)} teacher-forced tokens: mean KL {kl.mean():.4f} nats (max {kl.max():.3f}), "
          f"top-1 agreement {top1*100:.1f}%, mean logprob of bf16 tokens {chosen.mean():.3f} vs bf16 {rlp[np.arange(len(rt)), rt].mean():.3f}")
    n = next((i for i, (a, b) in enumerate(zip(out, rt)) if a != b), len(rt))
    print(f"greedy continuation matches bf16 for {n}/{len(rt)} tokens")
print(f"peak mem {mx.get_peak_memory()/1e9:.1f} GB")
```
