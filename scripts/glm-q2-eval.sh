#!/usr/bin/env bash
# glm-q2-eval.sh MODEL LABEL
#
# Evaluate one GLM Q2 candidate GGUF on notible:
#   1. perplexity (primary, objective, deterministic) via ds4 --perplexity-file
#   2. a small greedy/sampled battery (qualitative: does it loop / decohere?)
#
# Drives a full-model load, so it STOPS prod (port 8085) first and ALWAYS restarts
# it via trap EXIT (survives an SSH/agent disconnect).  Detached: launch with a
# nohup subshell.  Results land in /tmp/glm-q2-eval/$LABEL/.
set -u
MODEL="${1:?usage: glm-q2-eval.sh MODEL LABEL}"
LABEL="${2:?usage: glm-q2-eval.sh MODEL LABEL}"
DS4DIR=/Users/notible/Documents/ds4-glm
LAUNCH=/Users/notible/Documents/unsaltedbutter/scripts/setup-launchagents-tts.sh
PPLTXT="$DS4DIR/scripts/glm-ppl-heldout.txt"
OUT="/tmp/glm-q2-eval/$LABEL"

restart_prod() {
    bash -lc "$LAUNCH --start jumbo_server" >>"$OUT/prod.log" 2>&1
    echo "[trap] prod restart issued at $(date)" >>"$OUT/prod.log"
}
trap restart_prod EXIT

rm -rf "$OUT"; mkdir -p "$OUT"
{ echo "eval label=$LABEL model=$MODEL start $(date)"; ls -lh "$MODEL" 2>&1; } >"$OUT/run.log"
[ -f "$MODEL" ] || { echo "ABORT: no model $MODEL" >>"$OUT/run.log"; echo ABORTED >"$OUT/DONE"; exit 1; }

# --- stop prod, wait for the launchd instance to actually exit ---
bash -lc "$LAUNCH --stop jumbo_server" >"$OUT/stop.log" 2>&1
for i in $(seq 1 90); do pgrep -f ds4-server >/dev/null || break; sleep 2; done
if pgrep -f ds4-server >/dev/null; then
    echo "ABORT: ds4-server still up after 180s (prod left running)" >>"$OUT/run.log"
    echo ABORTED >"$OUT/DONE"; exit 1
fi
echo "prod down $(date)" >>"$OUT/run.log"

cd "$DS4DIR" || { echo "ABORT: no $DS4DIR" >>"$OUT/run.log"; echo ABORTED >"$OUT/DONE"; exit 1; }
[ -x ./ds4 ] || { echo "ABORT: no ./ds4" >>"$OUT/run.log"; echo ABORTED >"$OUT/DONE"; exit 1; }

# DS4_EXTRA passes extra ds4 flags (e.g. --ssd-streaming for a >256 GB candidate).
# NOPPL=1 skips perplexity (it is uninformative for GLM, and far too slow when streaming).
# --- 1. perplexity: 2000 scored tokens, ctx 2048 (same for every candidate) ---
if [ "${NOPPL:-0}" != "1" ]; then
    echo "=== perplexity @ $(date) ===" >>"$OUT/run.log"
    ./ds4 -m "$MODEL" -c 2048 -n 2000 ${DS4_EXTRA:-} --perplexity-file "$PPLTXT" >"$OUT/ppl.out" 2>"$OUT/ppl.err"
    echo "  $(grep -E 'ppl=' "$OUT/ppl.out" 2>/dev/null | tail -1)" >>"$OUT/run.log"
fi

# --- 2. battery: greedy is the deterministic signal; one sampled long run too ---
# Set PPLONLY=1 to skip the battery (fast iteration: perplexity is the primary metric).
if [ "${PPLONLY:-0}" = "1" ]; then
    echo "PPLONLY=1, skipping battery" >>"$OUT/run.log"
    echo "all done $(date)" >>"$OUT/run.log"; echo DONE >"$OUT/DONE"; exit 0
fi
SHORT="List the first five planets from the Sun."
LONG="List the first five planets from the Sun and give one interesting fact about each one."
gen() { # name prompt nmax temp topp
    echo "=== $1 temp=$4 top_p=$5 @ $(date) ===" >>"$OUT/run.log"
    ./ds4 -m "$MODEL" -c 2048 -n "$3" --temp "$4" --top-p "$5" ${DS4_EXTRA:-} -p "$2" >"$OUT/$1.out" 2>"$OUT/$1.err"
}
gen short_t0  "$SHORT" 100 0   1.0
gen long_t0   "$LONG"  240 0   1.0
gen long_t06  "$LONG"  240 0.6 0.95

echo "all done $(date)" >>"$OUT/run.log"
echo DONE >"$OUT/DONE"
# trap restarts prod on exit
