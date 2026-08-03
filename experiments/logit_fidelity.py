import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.runner import HEAD_METHODS, split_combined_method
from archead.lm_head_methods import HEAD_BASELINE_METHODS, compress_head_baseline, compress_lm_head, lm_config
from archead.modeling import load_model
from archead.utils import empty_cache, parse_dtype, safe_id


def compute_metrics(dense_logits: torch.Tensor, comp_logits: torch.Tensor) -> Dict[str, float]:
    dense_logits = dense_logits.float()
    comp_logits = comp_logits.float()
    
    metrics = {}
    B, V = dense_logits.shape
    
    dense_top1 = dense_logits.argmax(dim=-1)
    comp_top1 = comp_logits.argmax(dim=-1)
    metrics["top1_agree"] = (dense_top1 == comp_top1).float().sum().item()
    
    _, dense_top5 = dense_logits.topk(5, dim=-1)
    _, comp_top5 = comp_logits.topk(5, dim=-1)
    metrics["top5_agree"] = (dense_top1.unsqueeze(-1) == comp_top5).any(dim=-1).float().sum().item()
    
    _, dense_top10 = dense_logits.topk(10, dim=-1)
    _, comp_top10 = comp_logits.topk(10, dim=-1)
    metrics["top10_agree"] = (dense_top1.unsqueeze(-1) == comp_top10).any(dim=-1).float().sum().item()
    
    jaccard_5 = 0.0
    jaccard_10 = 0.0
    for i in range(B):
        s1 = set(dense_top5[i].tolist())
        s2 = set(comp_top5[i].tolist())
        jaccard_5 += len(s1 & s2) / len(s1 | s2)
        
        s3 = set(dense_top10[i].tolist())
        s4 = set(comp_top10[i].tolist())
        jaccard_10 += len(s3 & s4) / len(s3 | s4)
        
    metrics["jaccard_5"] = jaccard_5
    metrics["jaccard_10"] = jaccard_10
    
    log_p = F.log_softmax(dense_logits, dim=-1)
    p = torch.exp(log_p)
    log_q = F.log_softmax(comp_logits, dim=-1)
    metrics["kl"] = F.kl_div(log_q, p, reduction="sum").item()
    
    metrics["mse"] = F.mse_loss(dense_logits, comp_logits, reduction="sum").item()
    metrics["cosine"] = F.cosine_similarity(dense_logits, comp_logits, dim=-1).sum().item()
    metrics["count"] = B
    return metrics


def evaluate_logit_fidelity(args):
    device = args.device
    model_id = args.model
    cache_path = Path(args.cache_dir) / f"{safe_id(model_id)}__Salesforce__wikitext__seq512__tok32768__seed0.pt"
    
    if not cache_path.exists():
        print(f"Error: Cache not found at {cache_path}. Run `python -m experiments.runner` first.")
        return
        
    cache = torch.load(cache_path, map_location="cpu")
    h_all = (cache["h"] if isinstance(cache, dict) else cache).to(device)
    h_train = h_all
    h_val = h_all[:args.eval_tokens]
    
    print(f"Loading {model_id}...")
    model, _, _ = load_model(model_id, device="cpu", dtype=parse_dtype(args.dtype), trust_remote_code=True)
    out_proj = model.get_output_embeddings()
    W_cpu = out_proj.weight.detach().cpu().float()
    del model
    empty_cache()
    
    W_eval = W_cpu.to(device)
    
    methods = args.methods
    results = []
    
    for method in methods:
        print(f"\nEvaluating: {method}")
        
        if method == "dense":
            head = lambda x: x @ W_eval.T
            stats = {"name": "dense"}
        else:
            _, inner = HEAD_METHODS.get(method, (None, method))
            if inner in HEAD_BASELINE_METHODS:
                comp_head = compress_head_baseline(W_cpu, inner, device=device)
            else:
                comp_head = compress_lm_head(W_cpu, h_train, lm_config(inner, model_id), device=device)
            head = comp_head
            stats = comp_head.stats
            
        total_metrics = {
            "top1_agree": 0.0, "top5_agree": 0.0, "top10_agree": 0.0,
            "jaccard_5": 0.0, "jaccard_10": 0.0, "kl": 0.0, "mse": 0.0, "cosine": 0.0, "count": 0
        }
        
        chunk = args.chunk_size
        for i in tqdm(range(0, h_val.shape[0], chunk), desc="Chunk"):
            hb = h_val[i:i+chunk].to(dtype=W_eval.dtype)
            
            with torch.no_grad():
                dense_logits = hb @ W_eval.T
                comp_logits = head(hb)
                
            m = compute_metrics(dense_logits, comp_logits)
            for k, v in m.items():
                total_metrics[k] += v
                
        count = total_metrics.pop("count")
        final_metrics = {k: v / count for k, v in total_metrics.items()}
        
        print(f"[{method}] Top-1: {final_metrics['top1_agree']:.4f}  Top-5: {final_metrics['top5_agree']:.4f}  KL: {final_metrics['kl']:.6f}  MSE: {final_metrics['mse']:.6f}")
        
        results.append({
            "model_id": model_id,
            "method": method,
            **final_metrics,
            **stats
        })
        
        if method != "dense":
            del comp_head
            empty_cache()
            
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--methods", nargs="+", default=["dense", "head_group_int4", "head_svd8_group_int4", "head_gptq_int4", "head_archead"])
    parser.add_argument("--eval-tokens", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--cache-dir", type=str, default="./activation_cache")
    parser.add_argument("--out-jsonl", type=str, default="./outputs/logit_fidelity.jsonl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    
    evaluate_logit_fidelity(args)
