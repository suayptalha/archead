# ARCHead

ARCHead compresses large language-model output heads with an activation-aware quantized core and a low-rank residual correction. It is designed for models whose transformer blocks are already quantized while the vocabulary projection remains in BF16 or FP16.

Paper coming soon.

## Features

- Packed low-bit storage for the output projection
- Activation-metric low-rank residual correction
- Group-wise signed INT4 residual quantization
- Drop-in replacement for Hugging Face causal language-model output heads
- Optional Triton path for CUDA inference
- Benchmark tools for perplexity, logit fidelity, downstream evaluation, memory, and latency

## Requirements

- Python 3.10 or newer
- PyTorch and Transformers
- A CUDA GPU is strongly recommended for full-size checkpoints
- Access to any gated Hugging Face checkpoints you choose to use

## Installation

```bash
cd ARCHead
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional integrations as needed:

```bash
python -m pip install -e '.[triton]'
python -m pip install -e '.[baselines]'
python -m pip install -e '.[downstream]'
python -m pip install -e '.[all]'
```

## Quick start

ARCHead needs the dense output-head matrix and representative hidden states from a calibration corpus:

```python
import torch

from archead.data import load_texts
from archead.lm_head_methods import (
    compress_lm_head,
    extract_lm_cache,
    lm_config,
    replace_model_lm_head,
)
from archead.modeling import load_model

model_id = "Qwen/Qwen3-8B-Base"
device = "cuda"

model, tokenizer, model_kind = load_model(
    model_id,
    device=device,
    dtype=torch.float16,
    trust_remote_code=True,
)

dense_head = model.get_output_embeddings().weight.detach().cpu()

texts = load_texts(
    dataset="Salesforce/wikitext",
    dataset_config="wikitext-103-raw-v1",
    split="train",
    text_columns=("text",),
    max_rows=1_000,
    min_chars=32,
    seed=0,
    streaming=False,
    row_offset=0,
)
cache = extract_lm_cache(
    model,
    tokenizer,
    model_kind,
    texts,
    device=device,
    seq_len=512,
    tokens=32_768,
    batch_size=2,
)
calibration_hidden_states = cache["h"]

config = lm_config("archead_lm", model_id)
compressed_head = compress_lm_head(
    dense_head,
    calibration_hidden_states,
    config,
    device=device,
)
replace_model_lm_head(model, compressed_head)

print(compressed_head.stats)
```

The included runners can collect calibration states and execute an end-to-end benchmark directly:

```bash
bash scripts/run_main_benchmark.sh
```

Set `MODELS` to override the default checkpoint in the calibration sweep:

```bash
MODELS="Qwen/Qwen3-8B-Base" bash scripts/run_calibration_sweep.sh
```

Generated metrics and run metadata are written below `outputs/`. Downloaded activation caches are written below `activation_cache/`. Both directories are ignored by Git.

## Benchmark commands

```bash
bash scripts/run_main_benchmark.sh
bash scripts/run_calibration_sweep.sh
bash scripts/run_gptq_comparison.sh
bash scripts/run_logit_fidelity.sh
bash scripts/run_downstream_eval.sh
bash scripts/run_efficiency_benchmark.sh
```

To run every benchmark sequentially and create aggregate tables and plots:

```bash
bash scripts/run_all_benchmarks.sh
```

These workflows download large checkpoints and can require substantial GPU memory and runtime. Start with one model and a reduced token budget when validating a new environment. See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for configuration details.

## Repository layout

```text
archead/                 Compression, model, evaluation, and kernel code
experiments/             Python benchmark entry points
scripts/                 Reusable shell workflows
docs/                    Usage and benchmarking documentation
outputs/                 Locally generated metrics (Git-ignored)
activation_cache/        Locally generated calibration states (Git-ignored)
```

## Supported model families

The default configurations cover Qwen3, Gemma 3, and VibeThinker-style causal language models. Other Hugging Face causal LMs can be used when their output embedding is exposed through `get_output_embeddings()` and the model returns final-layer hidden states.

Model weights and datasets are not distributed in this repository. Obtain them from their original providers and follow their licenses, access rules, and acceptable-use terms.

## Validation

Run the lightweight checks without downloading model weights:

```bash
bash scripts/smoke_test.sh
```

## Contributing

Bug reports and focused pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change.

## Acknowledgments

We thank TextCortex for providing GPU support for this study.

## Citation

If ARCHead is useful in your work, use the metadata in [CITATION.cff](CITATION.cff).

## License

ARCHead is released under the [MIT License](LICENSE). Third-party models, datasets, and packages remain subject to their own licenses.
