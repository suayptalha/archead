#!/usr/bin/env bash
set -euo pipefail

python -m experiments.validate_packed_head
python -m experiments.measure_load_vram

python -m experiments.runner \
  --run-group latency \
  --output-dir outputs/benchmark_runs/latency \
  --cache-dir activation_cache \
  --models "Qwen/Qwen3-8B-Base,google/gemma-4-E4B,WeiboAI/VibeThinker-3B" \
  --methods "dense,head_archead,bnb_nf4,bnb_nf4+head_archead,awq_4bit,awq_4bit+head_archead" \
  --dataset-name Salesforce/wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --train-rows 4000 \
  --eval-rows 800 \
  --calib-tokens 32768 \
  --eval-tokens 4096 \
  --measure-latency \
  --latency-seq-len 512 \
  --latency-batch-size 1 \
  --latency-warmup 5 \
  --latency-iters 25 \
  --triton-bench \
  --triton-batch 128 \
  --triton-warmup 20 \
  --triton-iters 100 \
  --dtype float16 \
  --trust-remote-code
