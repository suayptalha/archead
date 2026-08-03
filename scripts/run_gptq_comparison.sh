#!/usr/bin/env bash
set -euo pipefail

python -m experiments.gptq_comparison \
  --model Qwen/Qwen3-8B-Base \
  --methods gptq archead \
  --seeds 0,1,2 \
  --calib-sizes 2048,4096,8192,16384 \
  --calib-pool-tokens 32768 \
  --eval-tokens 16384 \
  --eval-chunk-size 64 \
  --build-cache \
  --streaming \
  --trust-remote-code \
  --output-csv outputs/gptq_comparison/metrics.csv \
  --output-jsonl outputs/gptq_comparison/metrics.jsonl \
  --summary-csv outputs/gptq_comparison/summary.csv
