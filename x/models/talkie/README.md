# talkie-1930 on this fork: build and install runbook

Point an agent at this file to get `ollama run talkie-1930` working again on this Mac.

## Current state: built and working

**As of 2026-09-02 this is done.** The fork is built, a fork server is running, and both
`talkie-1930` and `talkie-1930-sys` generate correctly through it. If you only need to _use_
talkie, skip to [Serve](#serve). The build sections below are for rebuilding after an
upstream sync or on a fresh machine.

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
- `x/mlxrunner/imports.go` already has `_ "github.com/ollama/ollama/x/models/talkie"`, so
  the registration is linked into the runner. No wiring step is needed.
- The model is **already imported** into the shared store at `~/.ollama/models` as
  `talkie-1930:latest` (26 GB, safetensors-format layers carrying talkie's original tensor
  names such as `blocks.0.attn.attn_query.weight`). Do not re-import or re-download it.
- The `talkie-arch` branch is stale and far behind `main`. Ignore it. Build `main`.

## Prerequisites

Verified present on this machine as of 2026-09-02:

| Requirement     | State                                            |
| --------------- | ------------------------------------------------ |
| CMake 3.24+     | yes, 4.4.2                                       |
| Ninja           | yes, `/opt/homebrew/bin/ninja`                   |
| Xcode           | yes, 26.6                                        |
| Metal toolchain | yes, `xcrun -sdk macosx metal --version` answers |
| Disk headroom   | yes, ~960 GB free                                |
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
lovers parted by sea. Verified output from 2026-09-02, at temperature 0.75 and seed 1930:

> It might become an important means of communication between moving trains and fixed
> stations... a commander could issue orders to his troops while on the march. In cases of
> accident, aid could be speedily summoned; and, in times of war, intelligence could be
> rapidly transmitted between various parts of a field of battle.

Cold load was about 6 seconds. If the answer sounds like a 2020s assistant instead, something
is loading the wrong weights. Compare against the known-good MLX reference, which does not
involve ollama at all:

```sh
cd ~/Documents/AI/talkie
.venv/bin/talkie-mlx --model-dir ~/models/talkie-1930-13b-it-mlx --max-tokens 70 "..."
```

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

A persona only steers this model if 1930 contained one. Verified 2026-09-02 through the
fork on port 11435, same question and seed, only the system prompt changing:

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
