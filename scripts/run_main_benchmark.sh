#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m experiments.runner \
  --run-group main \
  --output-dir outputs/benchmark_runs/main \
  --cache-dir activation_cache \
  --models "Qwen/Qwen3-8B-Base,google/gemma-4-E4B,WeiboAI/VibeThinker-3B" \
  --methods "head_row_int8,head_group_int4,head_svd8_group_int4,head_archead_core_only,head_archead" \
  --dataset-name Salesforce/wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --train-split train \
  --eval-split test \
  --train-rows 12000 \
  --eval-rows 2500 \
  --calib-tokens 32768 \
  --eval-tokens 16384 \
  --seq-len 512 \
  --calib-batch-size 2 \
  --eval-batch-size 1 \
  --dtype float16 \
  --trust-remote-code
