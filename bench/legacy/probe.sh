#!/bin/bash
# probe.sh CARD MODEL_DIR...: stock vLLM on an older card (2080 Ti sm75, V100 sm70), one
# in-process run (bench/inproc/run.py: 64 requests, 1000 in / 100 out) per checkpoint x dtype
# (float16, bfloat16) x graphs (on, enforce_eager). Every log stays in $OUT/<tag>.log; each run's
# result row or its last error line goes to $OUT/summary.txt.
set -u
CARD=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
OUT=${OUT:-runs/legacy}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$CARD
for model in "$@"; do
  for dtype in float16 bfloat16; do
    for eager in false true; do
      tag=$(basename "$model")_${dtype}_eager-${eager}
      eng="{\"dtype\": \"$dtype\", \"enforce_eager\": $eager, \"mamba_ssm_cache_dtype\": \"float16\"}"
      timeout 1200 $PY "$HERE/../inproc/run.py" --model "$model" --gpu "$CARD" --input 1000 \
        --output 100 --n 64 --util 0.85 --seqs 32 --engine "$eng" > "$OUT/$tag.log" 2>&1
      code=$?
      row=$(grep -E '^\{.*req_s' "$OUT/$tag.log" | tail -1)
      err=$(grep -E 'Error|error:|not supported|requires' "$OUT/$tag.log" | grep -v Warning | tail -1 | cut -c1-240)
      echo "$tag exit=$code ${row:-FAIL: $err}" >> "$OUT/summary.txt"
    done
  done
done
echo "LEGACY_DONE" >> "$OUT/summary.txt"
