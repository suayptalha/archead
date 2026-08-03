from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .utils import safe_id


def _sanitize_tokenizer(model_id: str) -> Path:
    src = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
                "*.model",
            ],
        )
    )
    dst = Path("model_compat_tokenizers") / safe_id(model_id)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        p = dst / name
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj.get("extra_special_tokens"), list):
                obj.pop("extra_special_tokens", None)
            p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return dst


def load_tokenizer(model_id: str, trust_remote_code: bool):
    errors: List[str] = []
    for kwargs in ({}, {"extra_special_tokens": {}}, {"use_fast": False}, {"use_fast": True}):
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kwargs)
            if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    try:
        local = _sanitize_tokenizer(model_id)
        tok = AutoTokenizer.from_pretrained(str(local), trust_remote_code=trust_remote_code)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as exc:
        raise RuntimeError("Tokenizer load failed:\n" + "\n".join(errors) + f"\nSanitized: {exc}") from exc


def load_model(model_id: str, *, device: str, dtype: torch.dtype, trust_remote_code: bool):
    tok = load_tokenizer(model_id, trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
        )
        kind = "causal_lm"
    except Exception:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
        )
        tok = proc
        kind = "image_text_to_text"
    model.to(device).eval()
    return model, tok, kind


def encode(processor, kind: str, texts, *, max_length: int, device: str):
    if kind == "image_text_to_text":
        enc = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    else:
        enc = processor(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}

