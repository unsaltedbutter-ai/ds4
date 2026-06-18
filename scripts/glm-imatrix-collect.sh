#!/usr/bin/env bash
# glm-imatrix-collect.sh MODEL OUT.dat [extra ds4 imatrix args...]
#
# Collect a GLM activation imatrix by running the model over the calibration set
# (scripts/glm-imatrix-calib.txt) via sequential decode.  Loads the full model, so
# it STOPS prod (port 8085) and ALWAYS restarts it via trap EXIT.  Detached: launch
# with a nohup subshell.  Collection is slower than normal decode (a per-layer GPU
# sync reads the routed activations), so use --imatrix-max-prompts/-tokens to bound it.
#
# Smoke test:  glm-imatrix-collect.sh MODEL /tmp/x.dat --imatrix-max-prompts 1
# Full run:    glm-imatrix-collect.sh MODEL /Volumes/4TB-1/glm-5.2.imatrix.dat
set -u
MODEL="${1:?usage: glm-imatrix-collect.sh MODEL OUT.dat [extra args]}"
OUTDAT="${2:?usage: glm-imatrix-collect.sh MODEL OUT.dat [extra args]}"; shift 2
DS4DIR=/Users/notible/Documents/ds4-glm
LAUNCH=/Users/notible/Documents/unsaltedbutter/scripts/setup-launchagents-tts.sh
# Calibration dataset (DS4_IMATRIX_PROMPT-marker format).  Override with DATASET=... ;
# e.g. the large existing corpus gguf-tools/imatrix/dataset/rendered_prompts.txt for a
# denser imatrix (bound it with --imatrix-max-tokens N).
DATASET="${DATASET:-$DS4DIR/scripts/glm-imatrix-calib.txt}"
OUT=/tmp/glm-imatrix-collect

restart_prod() { bash -lc "$LAUNCH --start jumbo_server" >>"$OUT/prod.log" 2>&1; echo "[trap] $(date)" >>"$OUT/prod.log"; }
trap restart_prod EXIT

rm -rf "$OUT"; mkdir -p "$OUT"
{ echo "collect model=$MODEL out=$OUTDAT args=$* start $(date)"; } >"$OUT/run.log"
[ -f "$MODEL" ] || { echo "ABORT: no model $MODEL" >>"$OUT/run.log"; echo ABORTED >"$OUT/DONE"; exit 1; }

bash -lc "$LAUNCH --stop jumbo_server" >"$OUT/stop.log" 2>&1
for i in $(seq 1 90); do pgrep -f ds4-server >/dev/null || break; sleep 2; done
if pgrep -f ds4-server >/dev/null; then
    echo "ABORT: ds4-server still up" >>"$OUT/run.log"; echo ABORTED >"$OUT/DONE"; exit 1
fi
echo "prod down $(date)" >>"$OUT/run.log"

cd "$DS4DIR" || { echo ABORTED >"$OUT/DONE"; exit 1; }
./ds4 -m "$MODEL" -c 4096 --imatrix-dataset "$DATASET" --imatrix-out "$OUTDAT" "$@" >"$OUT/collect.log" 2>&1
rc=$?
{ echo "ds4 exit=$rc $(date)"; ls -lh "$OUTDAT" 2>&1; tail -3 "$OUT/collect.log"; } >>"$OUT/run.log"
echo DONE >"$OUT/DONE"
