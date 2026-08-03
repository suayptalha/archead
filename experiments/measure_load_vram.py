import torch
import gc
from archead.lm_head_methods import compress_lm_head, ARCHeadPacked, lm_config
from transformers import AutoModelForCausalLM

def measure_vram(model_id="Qwen/Qwen3-8B-Base"):
    if not torch.cuda.is_available():
        print("CUDA is not available! This script requires a GPU to measure VRAM.")
        return

    print(f"\nEvaluating Load-Time VRAM for {model_id}...")
    
    # 1. Fetch real LM head weight shape
    print("  -> Extracting real shape from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
    )
    W_cpu = model.get_output_embeddings().weight.detach().float()
    V, D = W_cpu.shape
    del model
    gc.collect()
    
    # 2. Pre-compute compressed ARCHead components on CPU (so GPU memory isn't polluted by compression temps)
    print("  -> Compressing to ARCHead on CPU...")
    torch.manual_seed(42)
    h_train_cpu = torch.randn(128, D, dtype=torch.float32)
    cfg = lm_config("archead_lm", model_id)
    archead_cpu = compress_lm_head(W_cpu, h_train_cpu, cfg, device="cpu")
    del W_cpu
    gc.collect()
    
    # ========================================================
    # DENSE HEAD VRAM MEASUREMENT
    # ========================================================
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # We use bfloat16 to match exactly what Qwen/Gemma use internally
    dense_head = torch.nn.Linear(D, V, bias=False, dtype=torch.bfloat16)
    
    dense_head = dense_head.cuda()
    dense_loaded_bytes = torch.cuda.memory_allocated()
    dense_loaded_mb = dense_loaded_bytes / (1024 * 1024)
    
    # Clean up
    del dense_head
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # ========================================================
    # ARCHead HEAD VRAM MEASUREMENT
    # ========================================================
    archead_gpu = archead_cpu.cuda()
    archead_loaded_bytes = torch.cuda.memory_allocated()
    archead_loaded_mb = archead_loaded_bytes / (1024 * 1024)
    
    # Clean up
    del archead_gpu
    del archead_cpu
    gc.collect()
    torch.cuda.empty_cache()
    
    # ========================================================
    # REPORTING
    # ========================================================
    saving_mb = dense_loaded_mb - archead_loaded_mb
    saving_ratio = dense_loaded_mb / archead_loaded_mb if archead_loaded_mb > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"LOAD-TIME VRAM SUMMARY (No Forward Pass) - {model_id}")
    print("=" * 60)
    print(f"Vocab Size: {V}, Hidden Size: {D}")
    print(f"Dense BF16 VRAM:    {dense_loaded_mb:>7.2f} MB")
    print(f"ARCHeadPacked VRAM:     {archead_loaded_mb:>7.2f} MB")
    print("-" * 60)
    print(f"Memory Saved:       {saving_mb:>7.2f} MB")
    print(f"Saving Ratio:       {saving_ratio:>7.2f}x")
    print("=" * 60)
    
    return {
        "Model": model_id,
        "Vocab Size": V,
        "Hidden Size": D,
        "Dense VRAM (MB)": f"{dense_loaded_mb:.2f}",
        "ARCHead VRAM (MB)": f"{archead_loaded_mb:.2f}",
        "Memory Saved (MB)": f"{saving_mb:.2f}",
        "Saving Ratio": f"{saving_ratio:.2f}"
    }

if __name__ == "__main__":
    import csv
    print("Measuring persistent/load-time GPU parameter memory.")
    print("Runtime peak memory is backend-dependent and is reported separately.")
    
    results = []
    for model in [
        "Qwen/Qwen3-8B-Base",
        "google/gemma-4-E4B",
        "WeiboAI/VibeThinker-3B",
        "mistralai/Ministral-7B-v0.3",
        "LiquidAI/LFM2.5-8B-A1B"
    ]:
        res = measure_vram(model)
        if res:
            results.append(res)
            
    if results:
        csv_path = "outputs/load_time_vram.csv"
        from pathlib import Path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nMetrics saved to {csv_path}")
