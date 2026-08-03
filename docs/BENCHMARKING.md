# Benchmarking ARCHead

This guide describes the included benchmark harness and the controls required for comparable measurements.

## Environment

The benchmark suite requires an NVIDIA GPU for full-size checkpoints. The reference environment used Python 3.12.13, PyTorch 2.11.0 with CUDA 12.8, Transformers 4.52 or newer, and one NVIDIA RTX PRO 6000 Blackwell Server Edition.

Runtime and memory measurements depend on the GPU, driver, CUDA version, package versions, allocator state, and backend implementation. Each runner writes its arguments and detected environment to `run_config.json` beside its metrics.

## Default checkpoints

- `Qwen/Qwen3-8B-Base`
- `google/gemma-4-E4B`
- `WeiboAI/VibeThinker-3B`
- `mistralai/Ministral-7B-v0.3`
- `LiquidAI/LFM2.5-8B-A1B`

Some checkpoints are gated or require authentication. Log in with `huggingface-cli login` and accept the provider's terms before starting a run. Checkpoints and datasets retain their original licenses.

## Shared controls

Use the same model revision, calibration split, evaluation split, calibration token count, evaluation token count, sequence length, random seed, dtype, and hardware when comparing methods. Do not reuse an activation cache generated for a different model or configuration; cache filenames encode the principal calibration settings.

The default WikiText-103 configuration is:

- Dataset: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`
- Calibration split: `train`
- Evaluation split: `test`
- Sequence length: 512
- Calibration budget: 32,768 tokens
- Evaluation budget: 16,384 tokens
- Calibration batch size: 2
- Evaluation batch size: 1
- Data type: FP16

## Workflows

### Main comparison

```bash
bash scripts/run_main_benchmark.sh
```

This runs dense and head-only baselines alongside ARCHead for the default model list.

### Calibration-size sweep

```bash
MODELS="Qwen/Qwen3-8B-Base" bash scripts/run_calibration_sweep.sh
```

The sweep uses calibration budgets of 4,096, 8,192, 16,384, 32,768, and 65,536 tokens while holding the evaluation budget fixed.

### GPTQ comparison

```bash
bash scripts/run_gptq_comparison.sh
```

This runs matched calibration seeds and budgets for GPTQ-style head quantization and ARCHead.

### Logit fidelity

```bash
bash scripts/run_logit_fidelity.sh
```

The runner measures top-1 agreement, KL divergence, and logit mean squared error against the dense output head.

### Downstream evaluation

```bash
bash scripts/run_downstream_eval.sh
```

This requires the `downstream` optional dependencies and evaluates HellaSwag, TruthfulQA MC2, and WinoGrande with `lm-evaluation-harness`.

### Memory and latency

```bash
bash scripts/run_efficiency_benchmark.sh
```

Persistent packed bytes, load-time parameter memory, and runtime peak memory are different quantities. Report each separately. Treat small latency differences as measurement variation unless repeated runs establish a stable effect.

### Aggregate metrics

```bash
bash scripts/aggregate_metrics.sh
```

This collects per-run JSONL metric files below `outputs/benchmark_runs/` and writes tables and plots to `outputs/summary/`.

## External integrations

- AWQ measurements require `autoawq` and a compatible AWQ checkpoint.
- NF4 measurements require `bitsandbytes` and supported CUDA hardware.
- Downstream tasks require `lm-eval`.
- Triton measurements require a compatible Triton installation and CUDA GPU.

Backend versions can affect memory, throughput, and numerical behavior. Record exact versions when publishing comparisons.
