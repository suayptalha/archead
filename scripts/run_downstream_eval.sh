#!/usr/bin/env bash
set -euo pipefail

python -m experiments.downstream \
  --model Qwen/Qwen3-8B-Base \
  --methods dense head_group_int4 head_gptq_int4 head_archead \
  --tasks hellaswag truthfulqa_mc2 winogrande \
  --batch-size 8 \
  --cache-dir activation_cache \
  --out-jsonl outputs/downstream_eval.jsonl
