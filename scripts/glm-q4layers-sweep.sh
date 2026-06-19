#!/usr/bin/env bash
# glm-q4layers-sweep.sh
#
# --q4-layers quality sweep: build + battery-eval Q2 variants that upgrade selected
# layers' gate/up experts to Q4_K (on top of the dense activation imatrix; down stays
# Q2_K).  Maps the gate/up-bit -> quality curve between the imatrix Q2 (all IQ2_XXS
# gate/up, ~219 GiB) and Q4-level gate/up (~366 GiB).  Each build runs prod-UP (~110 min);
# each eval runs prod-DOWN (battery only, streamed for robustness regardless of size —
# the QUALITY/output is identical resident vs streamed, only t/s differs).  Fully
# autonomous; launch detached and poll /tmp/glm-q4sweep.log (+ .done sentinel) and the
# per-label evals at /tmp/glm-q2-eval/<label>/.
set -u
DS4DIR=/Users/notible/Documents/ds4-glm
IMAT=/Volumes/4TB-1/glm-5.2.imatrix-dense.dat
LOG=/tmp/glm-q4sweep.log
rm -f "$LOG.done"
echo "=== q4-layers sweep start $(date) ===" >"$LOG"
[ -s "$IMAT" ] || { echo "ABORT: no imatrix $IMAT" >>"$LOG"; echo ABORTED >"$LOG.done"; exit 1; }

run_config() { # label  layers-csv  approx-size
    local label="$1" layers="$2" size="$3"
    local gguf="/Volumes/4TB-1/glm-5.2-q2-$label.gguf"
    echo "=== build $label (q4-layers=$layers, ~$size) $(date) ===" >>"$LOG"
    bash "$DS4DIR/scripts/glm-q2-build.sh" "$gguf" --imatrix "$IMAT" --q4-layers "$layers" >>"$LOG" 2>&1
    if [ ! -s "$gguf" ]; then echo "BUILD FAILED $label (see $LOG)" >>"$LOG"; return 1; fi
    ls -lh "$gguf" 2>&1 | awk '{print "  built:", $5, $9}' >>"$LOG"
    echo "=== eval $label $(date) ===" >>"$LOG"
    DS4_EXTRA="--ssd-streaming" NOPPL=1 bash "$DS4DIR/scripts/glm-q2-eval.sh" "$gguf" "$label" >>"$LOG" 2>&1
    echo "=== $label DONE $(date) ===" >>"$LOG"
}

# Increasing gate/up-Q4 coverage: minimal-resident -> partial -> all.
run_config q4gu-last8  "$(seq -s, 70 77)" "235 GiB (resident-capable)"
run_config q4gu-last16 "$(seq -s, 62 77)" "250 GiB"
run_config q4gu-all    "$(seq -s, 3 77)"  "366 GiB (streams)"

echo "=== q4-layers sweep ALL DONE $(date) ===" | tee -a "$LOG" >"$LOG.done"
