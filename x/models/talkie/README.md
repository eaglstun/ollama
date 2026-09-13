# talkie-1930 on this fork: build and install runbook

Point an agent at this file to get `ollama run talkie-1930` working again on this Mac.

## Current state: built and working

**As of 2026-09-12 this is done.** The fork is built, a fork server is running, and both
`talkie-1930` and `talkie-1930-sys` generate correctly through it. It was first brought up on
2026-09-02 and re-verified after the 2026-09-12 upstream sync, which bumped MLX, MLX-C, and
llama.cpp. The same day, two decode fixes took it from 16.5 to 17.8 tok/s, and one of them
brought the port's RMSNorm back in line with the reference, which changed the smoke-test
answer (see [Verify](#verify)). If you only need to _use_ talkie, skip to [Serve](#serve). The build sections
below are for rebuilding after an upstream sync or on a fresh machine. Read
[the stale-checkout trap](#rebuilding-after-an-upstream-sync-the-stale-checkout-trap) before
any rebuild that follows a sync.

The one thing that is **not** done, deliberately, is putting the fork on `PATH`. So this
still fails, and will keep failing:

```
$ ollama run talkie-1930
Error: mlx runner failed: Error: unsupported architecture: TalkieForCausalLM (exit: exit status 1)
```

Bare `ollama` is the stock Homebrew build (`/opt/homebrew/bin/ollama`, v0.32.7) talking to
the Homebrew server on port 11434, and stock ollama has never heard of `TalkieForCausalLM`.
Nothing is wrong with the model or the weights. You are simply talking to the wrong binary.
Every command in this file that touches talkie is therefore prefixed with an explicit
`OLLAMA_HOST` and an explicit path to the fork's binary. **Do not drop those prefixes.**

## What is already done (do not redo it)

- `x/models/talkie/talkie.go` implements the architecture for the MLX runner and is
  **tracked on `main`**. It self-registers via `base.Register("TalkieForCausalLM", newModel)`.
  Note: the 2026-09-01 upstream sync changed the `base.Model` interface, so `Forward` now
  returns `(hidden, auxHidden *mlx.Array)` instead of a single array, and the port was
  fixed on 2026-09-02 to match. Like the other dense archs it returns the final hidden
  state twice, since talkie has no draft/multi-token-prediction head. **If a future
  upstream sync breaks the build again, this interface is the first place to look.**
  The 2026-09-12 sync reworked MLX array lifetimes (scoped instead of pinned and swept)
  and needed no change to `talkie.go`.
- Two decode fixes landed on 2026-09-12 (branch `talkie-decode-speedups`; the research
  behind them is in [`PERFORMANCE.md`](PERFORMANCE.md)). `lm_head_gain` is folded into
  `lm_head` once at load instead of rescaling the whole 671 MB table every token, and
  `rmsNorm` calls the fused `mlx.RMSNormFn(x, nil, rmsEps)`. The fused norm computes
  `x * rsqrt(mean(x^2) + eps)`, which matches the MLX reference bit for bit. The port's
  earlier hand-rolled `x / sqrt(mean(x^2) + eps)` did not, and the difference was large
  enough to change the smoke-test answer. **Do not put the hand-rolled norm back.**
- `x/mlxrunner/imports.go` already has `_ "github.com/ollama/ollama/x/models/talkie"`, so
  the registration is linked into the runner. No wiring step is needed.
- The model is **already imported** into the shared store at `~/.ollama/models` as
  `talkie-1930:latest` (26 GB, safetensors-format layers carrying talkie's original tensor
  names such as `blocks.0.attn.attn_query.weight`). Do not re-import or re-download it.
- The `talkie-arch` branch is stale and far behind `main`. Ignore it. Build `main`.

## Prerequisites

Verified present on this machine as of 2026-09-12:

| Requirement     | State                                            |
| --------------- | ------------------------------------------------ |
| CMake 3.24+     | yes, 4.4.2                                       |
| Ninja           | yes, `/opt/homebrew/bin/ninja`                   |
| Xcode           | yes, 26.6                                        |
| Metal toolchain | yes, `xcrun -sdk macosx metal --version` answers |
| Disk headroom   | yes, ~858 GB free                                |
| **Go 1.26+**    | yes, 1.27.1                                      |

If `xcrun -sdk macosx metal --version` ever stops answering, run
`xcodebuild -downloadComponent MetalToolchain` and retry.

## Build

From the repo root (`~/Documents/dev/ollama`), on branch `main`:

```sh
cmake -B build .
cmake --build build --parallel 8
```

This produces the `ollama` binary at the repo root and the native runtime payload under
`build/lib/ollama`. On macOS arm64 the MLX backends are **on by default**
(`cmake/local.cmake` picks `metal_v4` or `metal_v3` from the SDK version and adds
`ollama-mlx-backends` to `ALL`), so no preset is required. The `MLX Metal` preset in
`CMakePresets.json` only exists to build the MLX backends _alone_, which is not what you want
here.

Expect this to take a while on a cold cache. It compiles llama.cpp and MLX from source.

The build's exit status is the only reliable signal. If you pipe it through `tail` or
background it behind an `echo`, check the log for `make: *** [all] Error` rather than
trusting a zero exit.

### Rebuilding after an upstream sync: the stale-checkout trap

When a sync changes `LLAMA_CPP_VERSION`, `MLX_VERSION`, or `MLX_C_VERSION`, the next build
can die a few minutes in with:

```
CMake Error at .../build/ollama-llama-cpp-source-prefix/tmp/ollama-llama-cpp-source-gitupdate.cmake:301 (message):
  Failed to unstash changes in:
  '/Users/eeaglstun/Documents/dev/ollama/build/_deps/llama_cpp-src'.
```

The dependencies are git checkouts under `build/_deps/`, and CMake's update step stashes any
local edits, checks out the new pin, and tries to pop the stash back on top. The edits in
there are almost always the build's own patches for the _old_ pin, so they no longer fit, the
pop fails, and CMake rolls the checkout back to the old commit and stops. It fails on one
dependency per run, so a sync that bumps two of them fails twice in a row.

Check all three checkouts before rebuilding:

```sh
cd ~/Documents/dev/ollama
for d in build/_deps/*-src; do echo "== $d"; git -C "$d" status --short; done
```

Do not blindly reset a dirty checkout. Identify the edits first:

- **`llama_cpp-src`** is patched by the build, from `llama/compat/001-llama-cpp-hooks.patch`
  and `llama/compat/models/*.patch`. Confirm the dirt is exactly those patches as they stood
  _before_ the sync (take them from the pre-sync commit with `git show <old-sha>:<path>`):

  ```sh
  git -C build/_deps/llama_cpp-src apply --reverse --check /path/to/old.patch
  ```

  If every dirty file is accounted for, reset those files with `git checkout -- <files>` and
  rebuild. The build re-applies the current patches to the new pin by itself.

- **`mlx-c-src`** has no patch step at all, so any dirt there was put in by hand. On
  2026-09-12 it was a shim that let the old MLX-C bindings (`fba4470`) compile against a
  newer MLX: a `force_fused` argument on `mlx_fast_scaled_dot_product_attention` and a
  cache argument on the `compile_clear_cache`/`compile_erase` calls. The new pin
  (`c74db53`) ships both APIs properly, so the shim was dropped. For any future dirt, check
  whether the new pin already covers it (`git -C build/_deps/mlx-c-src show <new-pin>:<file>`)
  before throwing it away, and ask the human if it does not.

- **`mlx-src`** was clean on 2026-09-12.

These are gitignored build checkouts, so nothing here touches the repo itself.

## Serve

Two servers coexist on this machine on purpose. They share the model store at
`~/.ollama/models` but listen on different ports:

| Port  | Server                                           | Knows talkie? |
| ----- | ------------------------------------------------ | ------------- |
| 11434 | Homebrew ollama 0.32.7, running as a LaunchAgent | no            |
| 11435 | **this fork**                                    | **yes**       |

Check whether the fork is already up before starting another one. A second `serve` on a
taken port dies instantly with `bind: address already in use`, and the failure is easy to
miss if you background it:

```sh
lsof -nP -iTCP:11435 -sTCP:LISTEN
```

If nothing is listening, start it:

```sh
cd ~/Documents/dev/ollama
OLLAMA_HOST=127.0.0.1:11435 ./ollama serve
```

Then talk to it, always with both the host and the fork's own binary:

```sh
OLLAMA_HOST=127.0.0.1:11435 ~/Documents/dev/ollama/ollama run talkie-1930 \
  "What might the wireless telephone one day become?"
```

To stop the fork's server, kill the PID that `lsof` reported. Leave the Homebrew LaunchAgent
alone: every other model on this box is served by it.

**Ask the human before doing any of these.** All three change the state of the machine beyond
the task, and the split-port arrangement above exists specifically so none of them is needed:

- `brew services stop ollama`, which takes every other model on the box offline
- `brew uninstall ollama` plus putting the fork on `PATH`, which makes an unreleased build the
  permanent system ollama
- `cmake --install build --prefix ...` into a shared prefix

## Verify

Success is a period-voiced completion, not a modern one. This is the smoke test:

```sh
OLLAMA_HOST=127.0.0.1:11435 ~/Documents/dev/ollama/ollama run talkie-1930 \
  "What might the wireless telephone one day become?"
```

A correct answer sounds like 1930: besieged garrisons, moving trains, skilled operators,
lovers parted by sea. Expected output since 2026-09-12, at temperature 0.75 and seed 1930
(217 tokens):

> The wireless telephone might become a means of communication between moving trains and
> fixed stations, or between ships at sea. It might enable a passenger in an express train
> to speak to his friends at the place from which he started, or a mariner to hold converse
> with the shore... The voice might be made audible from London to Paris, or from New York
> to San Francisco.

It ends: "...who shall venture to set bounds to the possible achievements of science, in an
age which has seen railways and telegraphs established, and has heard of flying
machines?--Chambers."

`ollama run` cannot set a seed from the command line, so for an exact-match regression check
after a rebuild, use the API with the same options:

```sh
curl -s http://127.0.0.1:11435/api/generate -d '{
  "model": "talkie-1930",
  "prompt": "What might the wireless telephone one day become?",
  "stream": false,
  "options": {"temperature": 0.75, "seed": 1930}
}'
```

This text dates from the 2026-09-12 RMSNorm fix. Before it, the port gave a different answer
at this seed ("It might become an important means of communication between moving trains
and fixed stations... a commander could issue orders to his troops while on the march",
ending on a Bradshaw), because its hand-rolled norm had drifted from the reference. The
fixed port matches the MLX reference: on this prompt the reference with its own norm and
with the fused kernel gives identical first-token logprobs and the same greedy
continuation, and swapping the old formula into the reference reproduces the old answer
(details and script in [`PERFORMANCE.md`](PERFORMANCE.md#applied-items-1-and-2)).

Same seed, same tokens means the forward pass is unchanged. If a rebuild changes the wording
at this seed, find out why before trusting the build.

For load and speed numbers, see [Throughput](#throughput). If the answer sounds like a 2020s
assistant instead, something is loading the wrong weights. Compare against the known-good MLX
reference, which does not involve ollama at all:

```sh
cd ~/Documents/AI/talkie
.venv/bin/talkie-mlx --model-dir ~/models/talkie-1930-13b-it-mlx --max-tokens 70 "..."
```

## Throughput

Measured 2026-09-12 on the M4 Max (40-core GPU, 546 GB/s nominal, 64 GB), all bf16, 200
generated tokens, seed 42, one discarded warmup, 3 runs averaged:

| engine                                          | gen tok/s | run range   |
| ----------------------------------------------- | --------- | ----------- |
| this fork, both decode fixes                    | **17.80** | 17.76–17.82 |
| this fork, `lm_head` fold only                  | 17.30     | 17.27–17.32 |
| this fork, before the fixes                     | 16.46     | 16.42–16.52 |
| standalone MLX reference (`bench_reference.py`) | 14.82     | 14.80–14.84 |
| measured floor (25.9 GB at 467 GB/s, 55.4 ms)   | ~18.0     |             |

The fixes took decode from 60.8 to 56.2 ms per token, within about 1 ms of the floor. Bare
matrix-vector products over this model's weights reach 467 GB/s on this GPU, not the 546
nominal, so 55.4 ms per token is as fast as bf16 goes here. Going meaningfully faster means
quantizing: int8 measured about 30 tok/s on the reference with small quality loss (item 3 in
[`PERFORMANCE.md`](PERFORMANCE.md)). The reference loses time sampling on the CPU in numpy
every token. Load was 4.1 s, and 1.1 s with the weights already in the page cache.

To re-run the ollama side, point the repo's bench script at the fork. It defaults to the
Homebrew port, where talkie does not exist:

```sh
HOST=http://127.0.0.1:11435 PROMPT="Write a long and detailed essay upon the future of the railways." \
  .agents/skills/ollama-bench/scripts/bench.sh talkie-1930
```

It reports prefill as 0 for talkie because the warmup caches the prompt, so the measured runs
skip prefill entirely. The reference side has no timer of its own. It is timed with
[`bench_reference.py`](bench_reference.py), a small harness around `MLXTalkie._generate_ids`
with stop tokens disabled so every run reaches 200. Run it from the reference venv:

```sh
cd ~/Documents/AI/talkie
.venv/bin/python ~/Documents/dev/ollama/x/models/talkie/bench_reference.py
```

**Benchmark only on an idle machine.** Check that neither server is busy first:

```sh
curl -s http://127.0.0.1:11434/api/ps; curl -s http://127.0.0.1:11435/api/ps
```

A first measurement on 2026-09-12 ran while the Homebrew server was in use and came back at
9.2 tok/s with an 11 s load. Both numbers were contention, not the model, and they briefly
made it into this file as fact.

## The system-prompt slot (done, kept here as reference)

`~/Documents/AI/talkie/Modelfile` fixes a real bug in the stock template: it has no
`<|system|>` slot, so system prompts are silently discarded before the model ever sees them.
The patched variant was **created on 2026-09-02** and verified working. To recreate it:

```sh
cd ~/Documents/AI/talkie
OLLAMA_HOST=127.0.0.1:11435 ~/Documents/dev/ollama/ollama create talkie-1930-sys -f Modelfile
```

It reuses the existing weight layers, so it costs no extra disk.

**The fork's `ollama run` has no `--system` flag** (it was removed upstream; the old command
in earlier versions of this file fails with `Error: unknown flag: --system`). Pass the system
prompt through the API instead, or use `/set system` inside an interactive session:

```sh
curl -s http://127.0.0.1:11435/api/generate -d '{
  "model": "talkie-1930-sys",
  "system": "Thou art a minister of the gospel.",
  "prompt": "What might the wireless telephone one day become?",
  "stream": false
}'
```

To confirm the system slot is actually wired up rather than just producing plausible prose,
send the same prompt with a fixed `seed` twice, once with `system` and once without. The
completions diverge if the template works, and are identical if the prompt is being dropped.

Re-checked 2026-09-12 on the fixed build, at temperature 0.75 and seed 1930. With no system
prompt, `talkie-1930-sys` produced exactly the `talkie-1930` answer from [Verify](#verify)
(217 tokens, trains and ships at sea). With `"Thou art a stern Presbyterian minister of the
gospel."` it diverged to ships of the same fleet and war-time use (143 tokens). The slot
still works.

A persona only steers this model if 1930 contained one. Verified 2026-09-02 through the
fork on port 11435, same question and seed, only the system prompt changing. These runs
predate the RMSNorm fix, so the exact replies would differ today; the pattern is the point:

| system prompt                 | reply length | what came back                          |
| ----------------------------- | ------------ | --------------------------------------- |
| none                          | 140 tokens   | ships at sea, an 1854 Commons committee |
| blunt Silicon Valley engineer | 39 tokens    | lovers conversing, nothing technical    |
| sentimental poet              | 140 tokens   | tender vows from midnight to morning    |
| stern Presbyterian minister   | 67 tokens    | sweet nothings from the top of Snowdon  |

Read that carefully, because it is two results, not one. "Be terse" landed hard: the engineer
run came back at a quarter the length of the control. "Silicon Valley engineer" landed
nowhere at all, and the model spent its new brevity on lovers. The system prompt is a real
lever, but it only catches on words the corpus actually holds, and 1930 held brevity and
ministers while holding no startup engineers whatsoever.

## Reference

- Architecture notes and the two traps (weightless fp32 RMSNorm, inverted RoPE sign) are in
  the package doc comment at the top of `talkie.go`.
- Reference implementations this port mirrors: `~/Documents/AI/talkie/src/talkie/model.py`
  (PyTorch) and `~/Documents/AI/talkie/src/talkie/mlx/model.py` (MLX).
- Build prerequisites and platform notes: `docs/development.md`.
