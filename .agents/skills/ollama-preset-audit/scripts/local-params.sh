#!/usr/bin/env bash
# local-params.sh — dump a local ollama model's sampling params + identity, and
# suggest where to find the upstream recommended presets to compare against.
#
# This is the LOCAL half of the audit. The agent does the WEB half (fetch the
# HF generation_config.json + model-card best-practices) and the comparison.
#
# Usage: local-params.sh <model>
set -euo pipefail

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
  echo "usage: local-params.sh <model>" >&2
  exit 2
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not on PATH" >&2; exit 1
fi
if ! ollama show "$MODEL" >/dev/null 2>&1; then
  echo "ERROR: model '$MODEL' not found locally (ollama list to see options)" >&2; exit 1
fi

echo "=== IDENTITY: $MODEL ==="
ollama show "$MODEL" 2>/dev/null | grep -iE "architecture|parameters|quantization|context length|capabilities|completion|vision|tools|thinking" || true

echo ""
echo "=== SHIPPED SAMPLING PARAMS (ollama default for this model) ==="
ollama show "$MODEL" --modelfile 2>/dev/null | grep -E "^PARAMETER|^RENDERER|^PARSER" || echo "(none set)"

echo ""
echo "=== THINKING DEFAULT? ==="
# Capture first, then match with pure-bash (no pipe). Piping into `grep -q`
# under `set -o pipefail` is a trap: grep -q closes the pipe on first match, the
# producer gets SIGPIPE and exits non-zero, and pipefail reports the whole
# pipeline as failed even though the match succeeded. A bash [[ == ]] test has
# no pipe and no subprocess, so it sidesteps the problem entirely.
SHOW_OUT="$(ollama show "$MODEL" 2>/dev/null)"
if [[ "$SHOW_OUT" == *[Tt]hinking* ]]; then
  echo "This model advertises 'thinking' capability. Qwen-family + many reasoning"
  echo "models default to THINKING MODE ON — so the shipped params should be judged"
  echo "against the upstream THINKING preset, not the instruct/non-thinking one."
else
  echo "No 'thinking' capability advertised; judge against the standard preset."
fi

echo ""
echo "=== NEXT: fetch upstream recommendation and compare ==="
echo "1. Identify the upstream repo (usually Qwen/<name>, meta-llama/<name>, google/<name>, etc.)."
echo "   If unknown, web-search: \"<model-name> huggingface generation_config best practices\""
echo "2. Fetch the raw generation_config.json, e.g.:"
echo "     https://huggingface.co/<org>/<repo>/raw/main/generation_config.json"
echo "3. Fetch the model card 'Best Practices' / recommended sampling section."
echo "4. Build the comparison table and flag mismatches (see SKILL.md)."
