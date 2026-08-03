from __future__ import annotations

import math
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .modeling import encode


@torch.no_grad()
def eval_ppl_texts(model, processor, kind: str, texts: List[str], *, device: str, seq_len: int, batch_size: int, max_tokens: int) -> Dict:
    from .utils import empty_cache
    empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model_loaded_mem_mb = torch.cuda.memory_allocated() / 1024 ** 2
    else:
        model_loaded_mem_mb = 0.0
        
    model.eval()
    total_loss = 0.0
    total = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="eval-ppl"):
        enc = encode(processor, kind, texts[i:i + batch_size], max_length=seq_len, device=device)
        ids = enc.get("input_ids")
        if ids is None or ids.shape[1] < 2:
            continue
        out = model(**enc, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1, :].float()
        labels = ids[:, 1:]
        if "attention_mask" in enc:
            mask = enc["attention_mask"][:, 1:].bool()
            logits = logits[mask]
            labels = labels[mask]
        else:
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)
        remain = max_tokens - total
        logits = logits[:remain]
        labels = labels[:remain]
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total += labels.numel()
        if total >= max_tokens:
            break
            
    if torch.cuda.is_available():
        runtime_peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        runtime_peak_mem_mb = 0.0
        
    ce = total_loss / max(total, 1)
    return {
        "ce": ce, 
        "ppl": math.exp(ce), 
        "tokens": total,
        "model_loaded_mem_mb": model_loaded_mem_mb,
        "runtime_peak_mem_mb": runtime_peak_mem_mb
    }


@torch.no_grad()
def measure_forward_latency(model, processor, kind: str, text: str, *, device: str, seq_len: int, batch_size: int, warmup: int, iters: int) -> Dict:
    from .utils import empty_cache
    texts = [text] * batch_size
    enc = encode(processor, kind, texts, max_length=seq_len, device=device)
    empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(**enc, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(**enc, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = int(enc["input_ids"].numel()) * iters
    return {
        "latency_ms": dt / iters * 1000.0,
        "tokens_per_sec": toks / dt,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0.0,
    }

