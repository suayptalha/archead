import argparse
import json
from pathlib import Path
import torch

from experiments.runner import HEAD_METHODS
from archead.lm_head_methods import HEAD_BASELINE_METHODS, compress_head_baseline, compress_lm_head, lm_config
from archead.modeling import load_model
from archead.utils import empty_cache, parse_dtype, safe_id

def evaluate_downstream(args):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise ImportError("Install the downstream extra first: pip install -e '.[downstream]'") from exc

    device = args.device
    model_id = args.model
    
    cache_path = Path(args.cache_dir) / f"{safe_id(model_id)}__Salesforce__wikitext__seq512__tok32768__seed0.pt"
    
    print(f"Loading cache from {cache_path}")
    if not cache_path.exists():
        print(f"Error: Cache not found at {cache_path}. Run `python -m experiments.runner` first to generate it.")
        return
        
    cache = torch.load(cache_path, map_location="cpu")
    h_train = (cache["h"] if isinstance(cache, dict) else cache).to(device)
    
    print(f"Loading {model_id}...")
    model, tokenizer, kind = load_model(model_id, device="cpu", dtype=parse_dtype(args.dtype), trust_remote_code=True)
    
    out_proj = model.get_output_embeddings()
    W_cpu = out_proj.weight.detach().cpu().float()
    
    results = []
    
    for method in args.methods:
        print(f"\n[{method}] Preparing model...")
        
        if method != "dense":
            _, inner = HEAD_METHODS.get(method, (None, method))
            if inner in HEAD_BASELINE_METHODS:
                comp_head = compress_head_baseline(W_cpu, inner, device=device)
            else:
                comp_head = compress_lm_head(W_cpu, h_train, lm_config(inner, model_id), device=device)
                
            stats = comp_head.stats
            old_head = model.get_output_embeddings()
            
            if hasattr(model, "lm_head"):
                model.lm_head = comp_head
            elif hasattr(model, "embed_out"):
                model.embed_out = comp_head
            else:
                raise ValueError("Could not find lm_head attribute to replace")
                
            model.to(device)
        else:
            stats = {"name": "dense"}
            model.to(device)
            
        print(f"[{method}] Running LM-eval on tasks: {args.tasks}")
        lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
        
        eval_results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=args.tasks,
            limit=args.limit,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
        )
        
        task_metrics = {}
        for task, res in eval_results["results"].items():
            acc = res.get("acc,none") or res.get("exact_match,none") or res.get("acc") or res.get("acc_norm,none")
            task_metrics[task] = acc
            print(f"  {task}: {acc}")
            
        results.append({
            "model_id": model_id,
            "method": method,
            **task_metrics,
            **stats
        })
        
        if method != "dense":
            if hasattr(model, "lm_head"):
                model.lm_head = old_head
            elif hasattr(model, "embed_out"):
                model.embed_out = old_head
            del comp_head
            empty_cache()
            
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--methods", nargs="+", default=["dense", "head_group_int4", "head_gptq_int4", "head_archead"])
    parser.add_argument("--tasks", nargs="+", default=["hellaswag", "truthfulqa_mc2", "winogrande"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=str, default="./activation_cache")
    parser.add_argument("--out-jsonl", type=str, default="./outputs/downstream_eval.jsonl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    
    evaluate_downstream(args)
