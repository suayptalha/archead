#!/usr/bin/env bash
set -euo pipefail

python -m experiments.logit_fidelity \
  --model Qwen/Qwen3-8B-Base \
  --methods dense head_row_int8 head_group_int4 head_svd8_group_int4 head_archead \
  --eval-tokens 16384 \
  --cache-dir activation_cache \
  --out-jsonl outputs/logit_fidelity.jsonl
