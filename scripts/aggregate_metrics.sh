#!/usr/bin/env bash
set -euo pipefail

python -m experiments.aggregate \
  --runs-dir outputs/benchmark_runs \
  --out-dir outputs/summary
