#!/usr/bin/env bash
set -euo pipefail

# This downloads several large checkpoints and runs every benchmark workflow.
bash scripts/run_main_benchmark.sh
bash scripts/run_calibration_sweep.sh
bash scripts/run_gptq_comparison.sh
bash scripts/run_logit_fidelity.sh
bash scripts/run_downstream_eval.sh
bash scripts/run_efficiency_benchmark.sh
bash scripts/aggregate_metrics.sh
