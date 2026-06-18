#!/usr/bin/env bash
# glm-q2-characterize.sh
#
# Characterize the CURRENT GLM-5.2 Q2 GGUF: run short/long prompts across
# temperatures and capture verbatim output + tok/s, to document what works and
# what doesn't.  Runs on notible.  Drives a full-model load, so it STOPS the
# prod DeepSeek server (port 8085) first and ALWAYS restarts it via trap EXIT
# (survives an SSH/agent disconnect).  Detached: launch with setsid/nohup.
set -u

MODEL=/Volumes/4TB-1/glm-5.2-q2.gguf
DS4DIR=/Users/notible/Documents/ds4-glm
LAUNCH=/Users/notible/Documents/unsaltedbutter/scripts/setup-launchagents-tts.sh
OUT=/tmp/glm-q2-char

restart_prod() {
    bash -lc "$LAUNCH --start jumbo_server" >>"$OUT/prod.log" 2>&1
    echo "[trap] prod restart issued at $(date)" >>"$OUT/prod.log"
}
trap restart_prod EXIT

rm -rf "$OUT"; mkdir -p "$OUT"
echo "start $(date)" >"$OUT/run.log"

# --- stop prod and wait for the launchd instance to actually exit ---
echo "stopping prod jumbo_server..." >>"$OUT/run.log"
bash -lc "$LAUNCH --stop jumbo_server" >"$OUT/stop.log" 2>&1
for i in $(seq 1 90); do pgrep -f ds4-server >/dev/null || break; sleep 2; done
if pgrep -f ds4-server >/dev/null; then
    echo "ERROR: ds4-server still up after 180s; aborting (prod left running)" >>"$OUT/run.log"
    echo "ABORTED" >"$OUT/DONE"; exit 1
fi
echo "prod down at $(date)" >>"$OUT/run.log"

cd "$DS4DIR" || { echo "no $DS4DIR" >>"$OUT/run.log"; echo "ABORTED" >"$OUT/DONE"; exit 1; }
[ -x ./ds4 ] || { echo "no ./ds4 binary" >>"$OUT/run.log"; echo "ABORTED" >"$OUT/DONE"; exit 1; }

run() { # name temp topp nmax prompt
    local name="$1" temp="$2" topp="$3" nmax="$4" prompt="$5"
    echo "=== $name temp=$temp top_p=$topp n=$nmax @ $(date) ===" >>"$OUT/run.log"
    ./ds4 -m "$MODEL" -c 2048 -n "$nmax" --temp "$temp" --top-p "$topp" -p "$prompt" \
        >"$OUT/$name.out" 2>"$OUT/$name.err"
    echo "  $name exit=$?" >>"$OUT/run.log"
}

SHORT="List the first five planets from the Sun."
LONG="List the first five planets from the Sun and give one interesting fact about each one."

# greedy is the reliable baseline; 1.0 is GLM's nominal temp (suspected decoherent);
# 0.6 is our adopted default.  top_p 0.95 for all sampled runs (greedy ignores it).
run short_t0    0    1.0  100  "$SHORT"
run short_t10   1.0  0.95 100  "$SHORT"
run long_t0     0    1.0  240  "$LONG"
run long_t06    0.6  0.95 240  "$LONG"
run long_t10    1.0  0.95 240  "$LONG"

echo "all runs done $(date)" >>"$OUT/run.log"
echo "DONE" >"$OUT/DONE"
# trap restarts prod on exit
