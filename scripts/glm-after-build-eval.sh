#!/usr/bin/env bash
# glm-after-build-eval.sh GGUF LABEL
#
# Orchestration glue: wait for a glm-q2-build.sh build of GGUF to finish (prod stays
# up during the build), then run glm-q2-eval.sh GGUF LABEL (which stops/restarts prod
# via its own trap).  Launch detached; poll /tmp/glm-after-build.LABEL.log.
set -u
GGUF="${1:?usage: glm-after-build-eval.sh GGUF LABEL}"
LABEL="${2:?usage: glm-after-build-eval.sh GGUF LABEL}"
DS4DIR=/Users/notible/Documents/ds4-glm
BUILDLOG="/tmp/glm-q2-build.$(basename "$GGUF").log"
LOG="/tmp/glm-after-build.$LABEL.log"

echo "waiting for build of $GGUF ($BUILDLOG) @ $(date)" >"$LOG"
while true; do
    if grep -q "glm-quantize exit=0" "$BUILDLOG" 2>/dev/null; then break; fi
    if grep -qE "glm-quantize exit=[1-9]" "$BUILDLOG" 2>/dev/null; then
        echo "BUILD FAILED (see $BUILDLOG) @ $(date)" >>"$LOG"; exit 1
    fi
    sleep 30
done
echo "build finished @ $(date); launching eval" >>"$LOG"
bash "$DS4DIR/scripts/glm-q2-eval.sh" "$GGUF" "$LABEL" >>"$LOG" 2>&1
echo "eval finished @ $(date)" >>"$LOG"
