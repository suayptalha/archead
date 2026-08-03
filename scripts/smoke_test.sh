#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q archead experiments
python -m unittest discover -s tests -v
python -m experiments.runner --help >/dev/null
python -m experiments.gptq_comparison --help >/dev/null
python -m experiments.logit_fidelity --help >/dev/null
python -m experiments.downstream --help >/dev/null
python -m experiments.aggregate --help >/dev/null
echo "Smoke tests passed."
