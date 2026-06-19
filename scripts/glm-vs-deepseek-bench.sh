#!/usr/bin/env bash
# glm-vs-deepseek-bench.sh
#
# Benchmark the prod DeepSeek V4 Flash (Q4, launchd jumbo_server, port 8085) against
# GLM-5.2 Q2 and Q4 on the same OpenAI API + the same general battery + tok/s.
# Phase 1 benchmarks prod live; phases 2-3 stop prod and run the GLM servers on 8085;
# trap EXIT always restarts prod (survives disconnects).  Detached: launch with nohup.
set -u
LAUNCH=/Users/notible/Documents/unsaltedbutter/scripts/setup-launchagents-tts.sh
DS4DIR=/Users/notible/Documents/ds4-glm
OUT=/tmp/glm-bench
PORT=8085
URL="http://localhost:$PORT"

restart_prod() {
    pkill -f "ds4-server -m /Volumes/4TB-1/glm" 2>/dev/null
    for i in $(seq 1 30); do pgrep -f ds4-server >/dev/null || break; sleep 2; done
    bash -lc "$LAUNCH --start jumbo_server" >>"$OUT/prod.log" 2>&1
    echo "[trap] prod restart issued $(date)" >>"$OUT/prod.log"
}
trap restart_prod EXIT

rm -rf "$OUT"; mkdir -p "$OUT"
echo "bench start $(date)" >"$OUT/run.log"

wait_ready() { for i in $(seq 1 90); do curl -s -m 5 "$URL/v1/models" >/dev/null 2>&1 && return 0; sleep 3; done; return 1; }
bench() { python3 "$DS4DIR/scripts/glm-bench-client.py" "$1" "$URL" "$OUT" >>"$OUT/run.log" 2>&1; }

# --- Phase 1: DeepSeek V4 Flash (prod, already up) ---
echo "=== phase 1: deepseek-v4-flash (prod) $(date) ===" >>"$OUT/run.log"
if wait_ready; then bench deepseek-v4-flash; else echo "deepseek not ready" >>"$OUT/run.log"; fi

# --- stop prod for the GLM runs ---
bash -lc "$LAUNCH --stop jumbo_server" >>"$OUT/run.log" 2>&1
for i in $(seq 1 90); do pgrep -f ds4-server >/dev/null || break; sleep 2; done
if pgrep -f ds4-server >/dev/null; then echo "ABORT: prod still up" >>"$OUT/run.log"; echo ABORTED >"$OUT/DONE"; exit 1; fi
cd "$DS4DIR" || { echo ABORTED >"$OUT/DONE"; exit 1; }

# --- Phase 2: GLM-5.2 Q2 (resident) ---
echo "=== phase 2: glm-5.2-q2 (resident) $(date) ===" >>"$OUT/run.log"
( nohup ./ds4-server -m /Volumes/4TB-1/glm-5.2-q2.gguf -c 2048 --port "$PORT" >"$OUT/glm-q2-server.log" 2>&1 </dev/null & )
if wait_ready; then bench glm-5.2-q2; else echo "glm-q2 not ready (see glm-q2-server.log)" >>"$OUT/run.log"; fi
pkill -f "ds4-server -m /Volumes/4TB-1/glm-5.2-q2"; for i in $(seq 1 60); do pgrep -f ds4-server >/dev/null || break; sleep 2; done

# --- Phase 3: GLM-5.2 Q4 (streaming) ---
echo "=== phase 3: glm-5.2-q4 (streaming) $(date) ===" >>"$OUT/run.log"
( nohup ./ds4-server -m /Volumes/4TB-1/glm-5.2-q4.gguf --ssd-streaming -c 2048 --port "$PORT" >"$OUT/glm-q4-server.log" 2>&1 </dev/null & )
if wait_ready; then bench glm-5.2-q4; else echo "glm-q4 not ready (see glm-q4-server.log)" >>"$OUT/run.log"; fi
pkill -f "ds4-server -m /Volumes/4TB-1/glm-5.2-q4"; for i in $(seq 1 60); do pgrep -f ds4-server >/dev/null || break; sleep 2; done

{ echo; echo "===== COMBINED SUMMARY ====="; cat "$OUT"/*.summary.txt 2>/dev/null; } >>"$OUT/run.log"
echo "bench done $(date)" >>"$OUT/run.log"
echo DONE >"$OUT/DONE"
# trap restarts prod on exit
