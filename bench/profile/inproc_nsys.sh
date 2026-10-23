#!/bin/bash
# inproc_nsys.sh OUT_DIR NAME GPU [run.py args]: bench/inproc/run.py under Nsight Systems, the
# capture being the measured run only (--profile), then the SQLite export and its summary.
# Env: MODEL (the checkpoint), NSYS (the nsys binary), PY (python with vLLM and kohakuvllm).
set -e
OUT=$1; NAME=$2; GPU=$3; shift 3
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "${NSYS:-nsys}" profile -o "$OUT/$NAME" -f true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true \
  "${PY:-python}" "$HERE/inproc/run.py" --model "$MODEL" --gpu "$GPU" --profile \
  --out "$OUT/$NAME.run.json" "$@" > "$OUT/$NAME.log" 2>&1
"${NSYS:-nsys}" export --type sqlite --force-overwrite true -o "$OUT/$NAME.sqlite" \
  "$OUT/$NAME.nsys-rep" > /dev/null
"${PY:-python}" "$HERE/profile/nsys_summary.py" "$OUT/$NAME.sqlite" --out "$OUT/$NAME.summary.json"
