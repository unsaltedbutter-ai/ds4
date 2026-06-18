#!/usr/bin/env bash
# glm-q2-build.sh OUTFILE [extra glm-quantize args...]
#
# Build a GLM-5.2 Q2 candidate GGUF on notible from the HF bf16 shards.
# Prod-SAFE: the converter is CPU + disk + a few GB RAM, so it does NOT stop the
# prod server.  Writes to /Volumes/4TB-1 (not the built-in drive).  Detached:
# launch with a nohup subshell; poll the printed log for completion.
#
# Examples:
#   glm-q2-build.sh /Volumes/4TB-1/glm-5.2-q2-a0.gguf            # current --q2 recipe
#   glm-q2-build.sh /Volumes/4TB-1/glm-5.2-q2-slice.gguf --layers 12   # fast partial build
set -u
OUT="${1:?usage: glm-q2-build.sh OUTFILE [extra glm-quantize args...]}"; shift || true
DS4DIR=/Users/notible/Documents/ds4-glm
HF=/Volumes/4TB-1/glm-5.2
LOG="/tmp/glm-q2-build.$(basename "$OUT").log"

cd "$DS4DIR/gguf-tools" || { echo "no $DS4DIR/gguf-tools"; exit 1; }
make glm-quantize >>"$LOG" 2>&1 || { echo "glm-quantize build failed (see $LOG)" | tee -a "$LOG"; exit 1; }
{
  echo "=== building $OUT @ $(date) ==="
  echo "args: --q2 $* --write-full $OUT"
} | tee "$LOG"
./glm-quantize --hf "$HF" --q2 "$@" --write-full "$OUT" >>"$LOG" 2>&1
rc=$?
{ echo "=== glm-quantize exit=$rc @ $(date) ==="; ls -lh "$OUT" 2>&1; } | tee -a "$LOG"
exit $rc
