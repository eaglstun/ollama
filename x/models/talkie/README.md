# talkie-1930 on this fork: build and install runbook

Point an agent at this file to get `ollama run talkie-1930` working again on this Mac.

## The problem

`ollama run talkie-1930` currently fails:

```
Error: mlx runner failed: Error: unsupported architecture: TalkieForCausalLM (exit: exit status 1)
```

Nothing is wrong with the model, and the weights are fine. The `ollama` on `PATH` is the stock
Homebrew build (`/opt/homebrew/bin/ollama`, v0.32.7), and stock ollama has never heard of
`TalkieForCausalLM`. The architecture only exists in **this fork**, and this fork has no
built binary. The fix is to build it and serve the model from the result.

## What is already done (do not redo it)

- `x/models/talkie/talkie.go` implements the architecture for the MLX runner and is
  **tracked on `main`**. It self-registers via `base.Register("TalkieForCausalLM", newModel)`.
  Note: the 2026-09-01 upstream sync changed the `base.Model` interface — `Forward` now
  returns `(hidden, auxHidden *mlx.Array)` instead of a single array — and the port was
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

**A conflict has to be resolved first.** Homebrew's ollama runs as a LaunchAgent
(`brew services list` shows it started) and holds `127.0.0.1:11434`. Two servers cannot
share that port, and both read the same model store at `~/.ollama/models`.

The least destructive option, and the default unless the human says otherwise:

```sh
brew services stop ollama
cd ~/Documents/dev/ollama
./ollama serve
```

Then in another shell, from the same repo so you use the fork's client:

```sh
./ollama run talkie-1930 "What might the wireless telephone one day become?"
```

The fork's server defaults to the same `~/.ollama/models`, so `talkie-1930` is already there
and needs no re-import.

**Ask the human before doing either of these**, since both change the state of the machine
beyond this task:

- `brew uninstall ollama` and installing the fork binary onto `PATH` (makes the fork the
  permanent system ollama, and every other model on this box then runs on an unreleased build)
- `cmake --install build --prefix ...` into a shared prefix

To avoid touching brew at all, serve the fork on another port instead:
`OLLAMA_HOST=127.0.0.1:11435 ./ollama serve`, then
`OLLAMA_HOST=127.0.0.1:11435 ./ollama run talkie-1930 "..."`.

## Verify

Success is a period-voiced completion, not a modern one. This is the smoke test:

```sh
./ollama run talkie-1930 "What might the wireless telephone one day become?"
```

A correct answer sounds like 1930: besieged garrisons, moving trains, skilled operators,
lovers parted by sea. If it sounds like a 2020s assistant, something is loading the wrong
weights. Compare against the known-good MLX reference, which does not involve ollama at all:

```sh
cd ~/Documents/AI/talkie
.venv/bin/talkie-mlx --model-dir ~/models/talkie-1930-13b-it-mlx --max-tokens 70 "..."
```

## Follow-up worth doing while you are here

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
send the same prompt with a fixed `seed` twice — once with `system` and once without. The
completions diverge if the template works, and are identical if the prompt is being dropped.

A persona only steers this model if 1930 contained one. "Poet" and "minister" work.
"Silicon Valley engineer" does not, because the corpus has no such creature.

## Reference

- Architecture notes and the two traps (weightless fp32 RMSNorm, inverted RoPE sign) are in
  the package doc comment at the top of `talkie.go`.
- Reference implementations this port mirrors: `~/Documents/AI/talkie/src/talkie/model.py`
  (PyTorch) and `~/Documents/AI/talkie/src/talkie/mlx/model.py` (MLX).
- Build prerequisites and platform notes: `docs/development.md`.
