# Contributing to ARCHead

Thank you for helping improve ARCHead. Before opening a pull request, search existing issues and keep each change focused on one problem.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
bash scripts/smoke_test.sh
```

## Pull requests

- Describe the motivation and user-visible behavior.
- Add or update checks for behavior you change.
- Run `bash scripts/smoke_test.sh` before submitting.
- Do not commit model weights, activation caches, credentials, or generated benchmark outputs.
- Preserve attribution and comply with the licenses of external models, datasets, and dependencies.

For a bug report, include the command, traceback, Python and package versions, GPU model, CUDA version, model identifier, and the smallest configuration that reproduces the problem. Remove tokens, credentials, and private paths from logs.
