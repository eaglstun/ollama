#!/usr/bin/env bash
# bench.sh — fair, repeatable tok/s benchmark for one or more local ollama models.
#
# Bakes in the gotchas that bite ad-hoc benchmarks:
#   - a warmup run per model (discarded) so cold model-load never skews eval rate
#   - num_predict cap so runs finish in bounded time (no more blown timeouts)
#   - fixed seed so token counts are comparable across models
#   - N runs per model, averaged, with per-run detail
#   - reports BOTH eval rate (generation) and prompt-eval rate (prefill)
#
# Uses the HTTP API (not `ollama run`) so we get clean JSON stats and exact control.
#
# Usage:
#   bench.sh <model> [<model2> ...]
#
# Env overrides:
#   PROMPT        prompt text (default: a medium technical prompt)
#   NUM_PREDICT   max tokens to generate per run (default: 200)
#   RUNS          measured runs per model (default: 3)
#   SEED          sampling seed (default: 42)
#   THINK         "true" to allow thinking, "false" to force direct (default: false)
#   HOST          ollama host (default: http://localhost:11434)
#
# Example A/B:
#   RUNS=3 NUM_PREDICT=256 bench.sh qwen3.6:27b-fast qwen3.6:27b
set -euo pipefail

HOST="${HOST:-http://localhost:11434}"
NUM_PREDICT="${NUM_PREDICT:-200}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
THINK="${THINK:-false}"
PROMPT="${PROMPT:-Explain how TCP congestion control works, covering slow start and congestion avoidance. Be technical.}"

if [ "$#" -lt 1 ]; then
  echo "usage: bench.sh <model> [<model2> ...]" >&2
  exit 2
fi

# Preflight: is ollama reachable?
if ! curl -sf "$HOST/api/tags" >/dev/null 2>&1; then
  echo "ERROR: ollama not reachable at $HOST (is 'ollama serve' running?)" >&2
  exit 1
fi

# One request -> prints "eval_count eval_dur_s prompt_count prompt_dur_s" or "ERR ..."
_one () {
  local model="$1"
  curl -s "$HOST/api/generate" -d "$(python3 - "$model" "$PROMPT" "$THINK" "$NUM_PREDICT" "$SEED" <<'PY'
import json,sys
model,prompt,think,npred,seed=sys.argv[1:6]
print(json.dumps({
  "model":model,"prompt":prompt,"think":think=="true","stream":False,
  "options":{"num_predict":int(npred),"seed":int(seed)}
}))
PY
)" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
except Exception as e:
    print('ERR bad-json'); sys.exit()
if 'error' in d:
    print('ERR '+str(d['error'])[:80]); sys.exit()
ec=d.get('eval_count',0); ed=d.get('eval_duration',0)/1e9
pc=d.get('prompt_eval_count',0); pd=d.get('prompt_eval_duration',0)/1e9
print(f'{ec} {ed:.4f} {pc} {pd:.4f}')
"
}

printf '%-34s %10s %12s %14s\n' "MODEL" "gen_tok/s" "prefill_tok/s" "gen_tok(count)"
printf '%s\n' "----------------------------------------------------------------------------"

for model in "$@"; do
  # warmup (discarded)
  _one "$model" >/dev/null 2>&1 || true

  tot_gen=0; tot_pre=0; n=0
  for r in $(seq 1 "$RUNS"); do
    read -r ec ed pc pd < <(_one "$model")
    if [ "$ec" = "ERR" ]; then
      printf '%-34s  run %s FAILED: %s\n' "$model" "$r" "$ed $pc $pd"
      continue
    fi
    gen=$(python3 -c "print(f'{$ec/$ed:.2f}' if $ed>0 else '0')")
    pre=$(python3 -c "print(f'{$pc/$pd:.2f}' if $pd>0 else '0')")
    printf '%-34s %10s %12s %14s   (run %s)\n' "$model" "$gen" "$pre" "$ec" "$r"
    tot_gen=$(python3 -c "print($tot_gen+$gen)")
    tot_pre=$(python3 -c "print($tot_pre+$pre)")
    n=$((n+1))
  done
  if [ "$n" -gt 0 ]; then
    avg_gen=$(python3 -c "print(f'{$tot_gen/$n:.2f}')")
    avg_pre=$(python3 -c "print(f'{$tot_pre/$n:.2f}')")
    printf '%-34s %10s %12s %14s   <== AVG of %s\n' "$model" "$avg_gen" "$avg_pre" "-" "$n"
  fi
  echo ""
done

echo "settings: num_predict=$NUM_PREDICT seed=$SEED think=$THINK runs=$RUNS host=$HOST"
