#!/usr/bin/env bash
# glm-imatrix-iterate.sh LABEL MAXTOK [DATASET]
#
# One full imatrix iteration on notible, end to end:
#   1. collect MAXTOK tokens of activation imatrix from DATASET (on baseline Q2)
#   2. build a Q2 GGUF with that imatrix (IQ2_XXS gate/up weighted by it)
#   3. eval it (perplexity + battery) vs the baseline
# Each sub-step manages the prod server itself (down for collect/eval, up for the
# build).  Launch detached; poll /tmp/glm-imatrix-iterate.LABEL.log and the eval at
# /tmp/glm-q2-eval/imatrix-LABEL/.  Artifacts on /Volumes/4TB-1.
set -u
LABEL="${1:?usage: glm-imatrix-iterate.sh LABEL MAXTOK [DATASET]}"
MAXTOK="${2:?usage: glm-imatrix-iterate.sh LABEL MAXTOK [DATASET]}"
DS4DIR=/Users/notible/Documents/ds4-glm
DATASET="${3:-$DS4DIR/gguf-tools/imatrix/dataset/rendered_prompts.txt}"
BASE=/Volumes/4TB-1/glm-5.2-q2.gguf
DAT="/Volumes/4TB-1/glm-5.2.imatrix-$LABEL.dat"
GGUF="/Volumes/4TB-1/glm-5.2-q2-imatrix-$LABEL.gguf"
LOG="/tmp/glm-imatrix-iterate.$LABEL.log"

echo "=== iterate $LABEL: collect $MAXTOK tok from $(basename "$DATASET") @ $(date) ===" >"$LOG"
DATASET="$DATASET" bash "$DS4DIR/scripts/glm-imatrix-collect.sh" "$BASE" "$DAT" --imatrix-max-tokens "$MAXTOK" >>"$LOG" 2>&1
[ -s "$DAT" ] || { echo "COLLECT FAILED (see $LOG)" >>"$LOG"; exit 1; }
echo "=== build $GGUF @ $(date) ===" >>"$LOG"
bash "$DS4DIR/scripts/glm-q2-build.sh" "$GGUF" --imatrix "$DAT" >>"$LOG" 2>&1
[ -s "$GGUF" ] || { echo "BUILD FAILED (see $LOG)" >>"$LOG"; exit 1; }
echo "=== eval (label imatrix-$LABEL) @ $(date) ===" >>"$LOG"
bash "$DS4DIR/scripts/glm-q2-eval.sh" "$GGUF" "imatrix-$LABEL" >>"$LOG" 2>&1
echo "=== iterate $LABEL DONE @ $(date) ===" >>"$LOG"
