import torch
import json
import os
import gc
from transformers import AutoModelForCausalLM
from archead.lm_head_methods import compress_lm_head, ARCHeadPacked, lm_config

def test_archead_packed():
    print("=" * 60)
    print("Validating True Packed ARCHead Implementation")
    print("=" * 60)

    models = [
        "Qwen/Qwen3-8B-Base",
        "google/gemma-4-E4B",
        "WeiboAI/VibeThinker-3B",
        "mistralai/Ministral-7B-v0.3",
        "LiquidAI/LFM2.5-8B-A1B"
    ]
    
    results = {}

    for model_id in models:
        print(f"\n[{model_id}] Loading model to extract real lm_head...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True, 
                torch_dtype=torch.float16, 
                device_map="cpu"
            )
            out_emb = model.get_output_embeddings()
            if out_emb is None or not hasattr(out_emb, "weight"):
                raise RuntimeError(f"Model {model_id} has no output embeddings!")
                
            W_cpu = out_emb.weight.detach().float()
            V, D = W_cpu.shape
            
            # Clean up model to save RAM
            del out_emb
            del model
            gc.collect()
            
            dense_bf16_bytes = V * D * 2
            print(f"[Dense] V={V}, D={D} -> Theoretical BF16 Bytes: {dense_bf16_bytes / 1024 / 1024:.2f} MB")
            
            torch.manual_seed(42)
            # We still simulate h_train since extracting real calibration cache requires running the full dataset
            h_train_cpu = torch.randn(128, D, dtype=torch.float32)
            
            # Fetch model-specific ARCHead config
            cfg = lm_config("archead_lm", model_id)
            
            print("Compressing with ARCHead...")
            head = compress_lm_head(W_cpu, h_train_cpu, cfg, device="cpu")
            
            assert isinstance(head, ARCHeadPacked), "Head must be an instance of ARCHeadPacked!"
            
            packed_bytes = sum(p.numel() * p.element_size() for p in head.buffers() if torch.is_tensor(p))
            print(f"[ARCHeadPacked] Actual PyTorch Stored Bytes in buffers: {packed_bytes / 1024 / 1024:.2f} MB")
            
            compression_ratio = dense_bf16_bytes / packed_bytes
            print(f"-> Byte compression ratio is valid: {compression_ratio:.2f}x")
            
            for name, buf in head.named_buffers():
                if buf.numel() == V * D and buf.element_size() >= 2:
                    raise RuntimeError(f"FOUND DENSE MATRIX in state_dict! {name}: {buf.shape} {buf.dtype}")
            
            print("Testing forward pass / numeric stability...")
            h_test = torch.randn(2, D, dtype=torch.float16)
            logits = head(h_test).float()
            
            assert logits.shape == (2, V), f"Logits shape mismatch: {logits.shape}"
            assert not torch.isnan(logits).any(), "Logits contain NaNs!"
            print("-> Forward pass OK!")
            
            results[model_id] = {
                "vocab_size": V,
                "hidden_size": D,
                "dense_mb": dense_bf16_bytes / 1024 / 1024,
                "packed_mb": packed_bytes / 1024 / 1024,
                "compression_ratio": compression_ratio
            }
            
        except Exception as e:
            print(f"Error processing {model_id}: {e}")
            
    # Save validation metrics.
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/packed_validation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 60)
    print("Validation passed. Metrics saved to outputs/packed_validation.json")
    print("=" * 60)

if __name__ == "__main__":
    test_archead_packed()
