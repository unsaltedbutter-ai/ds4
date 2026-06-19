#!/usr/bin/env bash
# glm-bench-round2.sh — fairness round: DeepSeek V4 Flash vs GLM-5.2 Q2 at temp 0.6 +
# reasoning=high (GLM's recommended serving mode, vs round 1's matched greedy/reasoning-off).
# Only the two fast contenders (GLM Q4 at ~0.6 t/s is already characterized).  Trap-restarts prod.
set -u
LAUNCH=/Users/notible/Documents/unsaltedbutter/scripts/setup-launchagents-tts.sh
DS4DIR=/Users/notible/Documents/ds4-glm
OUT=/tmp/glm-bench2
PORT=8085; URL="http://localhost:$PORT"
export BENCH_TEMP=0.6 BENCH_REASONING=high BENCH_MAXTOK=512

restart_prod() {
    pkill -f "ds4-server -m /Volumes/4TB-1/glm" 2>/dev/null
    for i in $(seq 1 30); do pgrep -f ds4-server >/dev/null || break; sleep 2; done
    bash -lc "$LAUNCH --start jumbo_server" >>"$OUT/prod.log" 2>&1
    echo "[trap] prod restart $(date)" >>"$OUT/prod.log"
}
trap restart_prod EXIT

rm -rf "$OUT"; mkdir -p "$OUT"; echo "round2 start $(date)" >"$OUT/run.log"
wait_ready() { for i in $(seq 1 90); do curl -s -m5 "$URL/v1/models" >/dev/null 2>&1 && return 0; sleep 3; done; return 1; }
bench() { python3 "$DS4DIR/scripts/glm-bench-client.py" "$1" "$URL" "$OUT" >>"$OUT/run.log" 2>&1; }

echo "=== deepseek-v4-flash (temp0.6 reasoning=high) $(date) ===" >>"$OUT/run.log"
if wait_ready; then bench deepseek-v4-flash; else echo "deepseek not ready" >>"$OUT/run.log"; fi

bash -lc "$LAUNCH --stop jumbo_server" >>"$OUT/run.log" 2>&1
for i in $(seq 1 90); do pgrep -f ds4-server >/dev/null || break; sleep 2; done
if pgrep -f ds4-server >/dev/null; then echo "ABORT: prod still up" >>"$OUT/run.log"; echo ABORTED >"$OUT/DONE"; exit 1; fi
cd "$DS4DIR" || { echo ABORTED >"$OUT/DONE"; exit 1; }

echo "=== glm-5.2-q2 (temp0.6 reasoning=high) $(date) ===" >>"$OUT/run.log"
( nohup ./ds4-server -m /Volumes/4TB-1/glm-5.2-q2.gguf -c 2048 --port "$PORT" >"$OUT/glm-q2-server.log" 2>&1 </dev/null & )
if wait_ready; then bench glm-5.2-q2; else echo "glm-q2 not ready" >>"$OUT/run.log"; fi
pkill -f "ds4-server -m /Volumes/4TB-1/glm-5.2-q2"; for i in $(seq 1 60); do pgrep -f ds4-server >/dev/null || break; sleep 2; done

{ echo; echo "===== ROUND 2 (temp0.6, reasoning=high) ====="; cat "$OUT"/*.summary.txt 2>/dev/null; } >>"$OUT/run.log"
echo "round2 done $(date)" >>"$OUT/run.log"; echo DONE >"$OUT/DONE"
