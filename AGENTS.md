# AGENTS.md

## Building

For a full build from the repository root:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

For quick Go-only iteration against an existing native payload:

```sh
go build .
go run . serve
```

See `docs/development.md` for prerequisites, platform notes, GPU backends, and
the full development workflow.

## Fork-local: the talkie-1930 architecture

This fork carries a custom architecture, `TalkieForCausalLM`, in `x/models/talkie/`, which
stock ollama does not have. If the task involves talkie, or `ollama run talkie-1930` fails
with `unsupported architecture: TalkieForCausalLM`, read
[`x/models/talkie/README.md`](x/models/talkie/README.md) first. It is the build, serve, and
verify runbook.
