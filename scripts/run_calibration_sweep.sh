#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
models="${MODELS:-Qwen/Qwen3-8B-Base}"

for tokens in 4096 8192 16384 32768 65536; do
  python -m experiments.runner \
    --run-group "calib_${tokens}" \
    --output-dir "outputs/benchmark_runs/calib_${tokens}" \
    --cache-dir activation_cache \
    --models "${models}" \
    --methods head_archead \
    --dataset-name Salesforce/wikitext \
    --dataset-config wikitext-103-raw-v1 \
    --train-split train \
    --eval-split test \
    --train-rows 12000 \
    --eval-rows 2500 \
    --calib-tokens "${tokens}" \
    --eval-tokens 16384 \
    --seq-len 512 \
    --calib-batch-size 2 \
    --eval-batch-size 1 \
    --dtype float16 \
    --trust-remote-code
done
