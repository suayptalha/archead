from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Silence excessive logs and progress bars unless --verbose is requested
if "--verbose" not in sys.argv:
    warnings.filterwarnings("ignore")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["UT_LOGGING_LEVEL"] = "error"
    
    # Disable standard logging below ERROR
    logging.disable(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("gptqmodel").setLevel(logging.ERROR)
    logging.getLogger("auto_gptq").setLevel(logging.ERROR)
    logging.getLogger("awq").setLevel(logging.ERROR)
    
    # Silence logbar progress bars/logs (used by GPTQModel)
    try:
        from logbar import LogBar
        LogBar.shared().setLevel("ERROR")
    except ImportError:
        pass
        
    try:
        import datasets
        datasets.utils.logging.set_verbosity_error()
        datasets.utils.logging.disable_progress_bar()
    except ImportError:
        pass

    try:
        import tqdm
        tqdm.tqdm = partial(tqdm.tqdm, disable=True)
        try:
            import tqdm.notebook as tqdm_notebook
            tqdm_notebook.tqdm = partial(tqdm_notebook.tqdm, disable=True)
        except ImportError:
            pass
        try:
            import tqdm.cli as tqdm_cli
            tqdm_cli.tqdm = partial(tqdm_cli.tqdm, disable=True)
        except ImportError:
            pass
        try:
            import tqdm.auto as tqdm_auto
            tqdm_auto.tqdm = partial(tqdm_auto.tqdm, disable=True)
        except ImportError:
            pass
    except ImportError:
        pass

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archead.data import load_texts
from archead.eval import eval_ppl_texts, measure_forward_latency
from archead.external_baselines import EXTERNAL_METHODS, run_external_loader
from archead.lm_head_methods import (
    HEAD_BASELINE_METHODS,
    benchmark_archead_triton,
    compress_head_baseline,
    compress_lm_head,
    extract_lm_cache,
    lm_config,
    replace_model_lm_head,
)
from archead.modeling import load_model
from archead.utils import empty_cache, environment, parse_dtype, safe_id, set_seed, write_json


HEAD_METHODS = {
    "head_row_int8": ("baseline", "row_int8"),
    "head_group_int4": ("baseline", "group_int4"),
    "head_svd8_group_int4": ("baseline", "svd8_group_int4"),
    "head_am_lrgq_adaptive": ("merged", "am_lrgq_adaptive"),
    "head_archead": ("merged", "archead_lm"),
    "head_archead_core_only": ("merged", "archead_core_only"),
    "head_hybrid_archead_core_amres": ("merged", "hybrid_archead_core_amres"),
    "head_hybrid_amcore_archead_res": ("merged", "hybrid_amcore_archead_res"),
    "head_gptq_int4": ("merged", "gptq_int4"),
    "head_archead_unpacked": ("merged", "archead_unpacked"),
}
CSV_FIELDS = [
    "run_group",
    "dataset_name",
    "dataset_config",
    "dataset_split",
    "seed",
    "model_id",
    "method",
    "status",
    "error",
    "dense_ce",
    "dense_ppl",
    "ce",
    "ppl",
    "delta_ce",
    "relative_ppl",
    "eval_tokens",
    "head_byte_ratio_vs_bf16",
    "head_total_bytes",
    "ff_byte_ratio_vs_bf16",
    "ff_total_bytes",
    "known_byte_ratio_vs_bf16",
    "known_dense_bf16_bytes",
    "latency_ms",
    "tokens_per_sec",
    "peak_gpu_mb",
    "head_ref_ms",
    "head_kernel_ms",
    "head_latency_ratio",
    "seconds",
    "model_loaded_mem_mb",
    "runtime_peak_mem_mb",
    "config_json",
]


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def ensure_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not exists:
        writer.writeheader()
    return f, writer


def write_row(writer, row: Dict):
    clean = {k: row.get(k, "") for k in CSV_FIELDS}
    for k, v in list(clean.items()):
        if isinstance(v, (dict, list, tuple)):
            clean[k] = json.dumps(v, sort_keys=True)
    writer.writerow(clean)


def unload(*objs):
    del objs
    gc.collect()
    empty_cache()


def cache_path(args, model_id: str) -> Path:
    name = f"{safe_id(model_id)}__{safe_id(args.dataset_name)}__seq{args.seq_len}__tok{args.calib_tokens}__seed{args.seed}.pt"
    return Path(args.cache_dir) / name


def get_lm_cache(args, model, processor, kind: str, model_id: str, train_texts: List[str]) -> torch.Tensor:
    path = cache_path(args, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.rebuild_cache:
        obj = torch.load(path, map_location="cpu")
        return obj["h"] if isinstance(obj, dict) else obj
    cache = extract_lm_cache(
        model,
        processor,
        kind,
        train_texts,
        device=args.device,
        seq_len=args.seq_len,
        tokens=args.calib_tokens,
        batch_size=args.calib_batch_size,
    )
    torch.save(cache, path)
    return cache["h"]


def dense_reference(args, model_id: str, eval_texts: List[str]) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        model, proc, kind = load_model(model_id, device=args.device, dtype=parse_dtype(args.dtype), trust_remote_code=args.trust_remote_code)
        ref = eval_ppl_texts(
            model,
            proc,
            kind,
            eval_texts,
            device=args.device,
            seq_len=args.seq_len,
            batch_size=args.eval_batch_size,
            max_tokens=args.eval_tokens,
        )
        if args.measure_latency:
            ref.update(
                measure_forward_latency(
                    model,
                    proc,
                    kind,
                    eval_texts[0],
                    device=args.device,
                    seq_len=args.latency_seq_len,
                    batch_size=args.latency_batch_size,
                    warmup=args.latency_warmup,
                    iters=args.latency_iters,
                )
            )
        unload(model, proc)
        return ref, None
    except Exception as exc:
        empty_cache()
        return None, f"{type(exc).__name__}: {exc}"


def split_combined_method(method: str) -> Tuple[Optional[str], bool]:
    if method in HEAD_METHODS:
        return method, False
    if method == "dense":
        return None, False
    return None, False


def known_byte_ratio(head_stats: Dict, ff_stats: Dict) -> Tuple[float, int]:
    total = int(head_stats.get("total_bytes", 0) or 0) + int(ff_stats.get("ff_total_bytes", 0) or 0)
    dense = int(head_stats.get("vocab_size", 0) or 0) * int(head_stats.get("hidden_size", 0) or 0) * 2
    dense += int(ff_stats.get("ff_dense_bf16_bytes", 0) or 0)
    return (total / dense if dense else math.nan), dense


def run_internal_method(args, model_id: str, method: str, train_texts: List[str], eval_texts: List[str], dense_ref: Optional[Dict]) -> Dict:
    t0 = time.perf_counter()
    head_method, _ = split_combined_method(method)
    head_stats: Dict = {}
    ff_stats: Dict = {}
    triton_stats: Dict = {}
    latency_stats: Dict = {}

    model, proc, kind = load_model(model_id, device=args.device, dtype=parse_dtype(args.dtype), trust_remote_code=args.trust_remote_code)

    if head_method:
        out = model.get_output_embeddings()
        if out is None or not hasattr(out, "weight"):
            raise RuntimeError("Model has no output embedding weight for LM-head compression")
        W_cpu = out.weight.detach().cpu().float()
        _, inner = HEAD_METHODS[head_method]
        if inner in HEAD_BASELINE_METHODS:
            head = compress_head_baseline(W_cpu, inner, device=args.device)
        else:
            h_train = get_lm_cache(args, model, proc, kind, model_id, train_texts)
            head = compress_lm_head(W_cpu, h_train, lm_config(inner, model_id), device=args.device)
            if args.triton_bench:
                triton_stats = benchmark_archead_triton(
                    head,
                    device=args.device,
                    batch=args.triton_batch,
                    warmup=args.triton_warmup,
                    iters=args.triton_iters,
                )
        replace_model_lm_head(model, head)
        head_stats = dict(head.stats)
        if 'h_train' in locals():
            del h_train
        unload(W_cpu, head)

    if args.measure_latency:
        latency_stats = measure_forward_latency(
            model,
            proc,
            kind,
            eval_texts[0],
            device=args.device,
            seq_len=args.latency_seq_len,
            batch_size=args.latency_batch_size,
            warmup=args.latency_warmup,
            iters=args.latency_iters,
        )

    result = eval_ppl_texts(
        model,
        proc,
        kind,
        eval_texts,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.eval_batch_size,
        max_tokens=args.eval_tokens,
    )
    kb_ratio, dense_bytes = known_byte_ratio(head_stats, ff_stats)
    dense_ce = dense_ref.get("ce") if dense_ref else math.nan
    dense_ppl = dense_ref.get("ppl") if dense_ref else math.nan
    unload(model, proc)
    return {
        "status": "ok",
        "error": "",
        "dense_ce": dense_ce,
        "dense_ppl": dense_ppl,
        "ce": result["ce"],
        "ppl": result["ppl"],
        "delta_ce": result["ce"] - dense_ce if dense_ref else math.nan,
        "relative_ppl": result["ppl"] / dense_ppl if dense_ref else math.nan,
        "eval_tokens": result["tokens"],
        "head_byte_ratio_vs_bf16": head_stats.get("byte_ratio_vs_bf16", ""),
        "head_total_bytes": head_stats.get("total_bytes", ""),
        "ff_byte_ratio_vs_bf16": ff_stats.get("ff_byte_ratio_vs_bf16", ""),
        "ff_total_bytes": ff_stats.get("ff_total_bytes", ""),
        "known_byte_ratio_vs_bf16": kb_ratio,
        "known_dense_bf16_bytes": dense_bytes,
        "latency_ms": latency_stats.get("latency_ms", ""),
        "tokens_per_sec": latency_stats.get("tokens_per_sec", ""),
        "peak_gpu_mb": latency_stats.get("peak_gpu_mb", ""),
        "head_ref_ms": triton_stats.get("head_ref_ms", ""),
        "head_kernel_ms": triton_stats.get("head_kernel_ms", ""),
        "head_latency_ratio": triton_stats.get("head_latency_ratio", ""),
        "seconds": time.perf_counter() - t0,
        "model_loaded_mem_mb": result.get("model_loaded_mem_mb", ""),
        "runtime_peak_mem_mb": result.get("runtime_peak_mem_mb", ""),
        "config_json": {"head": head_stats, "ff": ff_stats, "triton": triton_stats},
    }


def run_external_method(args, model_id: str, method: str, train_texts: List[str], eval_texts: List[str], dense_ref: Optional[Dict]) -> Dict:
    t0 = time.perf_counter()
    base_method = method.split("+")[0]
    has_head = "+head_" in method
    head_method_name = method.split("+")[1] if has_head else None

    loader = run_external_loader(base_method, model_id, train_texts[: args.external_calib_rows], device=args.device, trust_remote_code=args.trust_remote_code)
    if not loader.ok:
        return {"status": "error", "error": loader.error, "seconds": time.perf_counter() - t0, "config_json": loader.stats or {}}
    model = loader.model
    proc = loader.tokenizer
    kind = loader.kind
    if hasattr(model, "to") and not getattr(model, "hf_device_map", None):
        model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()
        
    head_stats = {}
    if has_head:
        out = model.get_output_embeddings()
        if out is None or not hasattr(out, "weight"):
            raise RuntimeError("Model has no output embedding weight for LM-head compression")
        W_cpu = out.weight.detach().cpu().float()
        
        _, inner = HEAD_METHODS[head_method_name]
        if inner in HEAD_BASELINE_METHODS:
            head = compress_head_baseline(W_cpu, inner, device=args.device)
        else:
            h_train = get_lm_cache(args, model, proc, kind, model_id, train_texts)
            head = compress_lm_head(W_cpu, h_train, lm_config(inner, model_id), device=args.device)
            
        replace_model_lm_head(model, head)
        head_stats = dict(head.stats)
        del h_train
        unload(W_cpu, head)
            
    latency_stats: Dict = {}
    if args.measure_latency:
        latency_stats = measure_forward_latency(
            model,
            proc,
            kind,
            eval_texts[0],
            device=args.device,
            seq_len=args.latency_seq_len,
            batch_size=args.latency_batch_size,
            warmup=args.latency_warmup,
            iters=args.latency_iters,
        )
    result = eval_ppl_texts(
        model,
        proc,
        kind,
        eval_texts,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.eval_batch_size,
        max_tokens=args.eval_tokens,
    )
    dense_ce = dense_ref.get("ce") if dense_ref else math.nan
    dense_ppl = dense_ref.get("ppl") if dense_ref else math.nan
    unload(model, proc)
    return {
        "status": "ok",
        "error": "",
        "dense_ce": dense_ce,
        "dense_ppl": dense_ppl,
        "ce": result["ce"],
        "ppl": result["ppl"],
        "delta_ce": result["ce"] - dense_ce if dense_ref else math.nan,
        "relative_ppl": result["ppl"] / dense_ppl if dense_ref else math.nan,
        "eval_tokens": result["tokens"],
        "latency_ms": latency_stats.get("latency_ms", ""),
        "tokens_per_sec": latency_stats.get("tokens_per_sec", ""),
        "peak_gpu_mb": latency_stats.get("peak_gpu_mb", ""),
        "seconds": time.perf_counter() - t0,
        "model_loaded_mem_mb": result.get("model_loaded_mem_mb", ""),
        "runtime_peak_mem_mb": result.get("runtime_peak_mem_mb", ""),
        "config_json": {"loader_stats": loader.stats or {}, "head_stats": head_stats},
    }


def base_row(args, model_id: str, method: str) -> Dict:
    return {
        "run_group": args.run_group,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config or "",
        "dataset_split": args.eval_split,
        "seed": args.seed,
        "model_id": model_id,
        "method": method,
    }


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "environment": environment(),
            "head_methods": sorted(HEAD_METHODS),
            "external_methods": sorted(EXTERNAL_METHODS),
        },
    )
    csv_path = Path(args.out_csv) if args.out_csv else (out_dir / "metrics.csv")
    jsonl_path = Path(args.out_jsonl) if args.out_jsonl else (out_dir / "metrics.jsonl")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file, writer = ensure_writer(csv_path)
    jsonl = jsonl_path.open("a", encoding="utf-8")

    print("[data] loading calibration/evaluation text")
    train_texts = load_texts(
        dataset=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_columns=parse_csv(args.text_columns),
        max_rows=args.train_rows,
        min_chars=args.min_chars,
        seed=args.seed,
        streaming=args.streaming,
        row_offset=args.train_offset,
        text_file=args.train_text_file,
    )
    eval_texts = load_texts(
        dataset=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_columns=parse_csv(args.text_columns),
        max_rows=args.eval_rows,
        min_chars=args.min_chars,
        seed=args.seed + 1,
        streaming=args.streaming,
        row_offset=args.eval_offset,
        text_file=args.eval_text_file,
    )
    print(f"[data] train_texts={len(train_texts)} eval_texts={len(eval_texts)}")

    models = parse_csv(args.models)
    methods = []
    if args.methods:
        for m in args.methods:
            methods.extend([x.strip() for x in m.split(",") if x.strip()])
    print(f"[models] {models}")
    print(f"[methods] {methods}")

    try:
        for model_id in models:
            print("\n" + "=" * 100)
            print(f"MODEL {model_id}")
            print("=" * 100)
            dense_ref, dense_error = dense_reference(args, model_id, eval_texts)
            if dense_ref:
                row = base_row(args, model_id, "dense")
                row.update(
                    {
                        "status": "ok",
                        "dense_ce": dense_ref["ce"],
                        "dense_ppl": dense_ref["ppl"],
                        "ce": dense_ref["ce"],
                        "ppl": dense_ref["ppl"],
                        "delta_ce": 0.0,
                        "relative_ppl": 1.0,
                        "eval_tokens": dense_ref["tokens"],
                        "latency_ms": dense_ref.get("latency_ms", ""),
                        "tokens_per_sec": dense_ref.get("tokens_per_sec", ""),
                        "peak_gpu_mb": dense_ref.get("peak_gpu_mb", ""),
                        "seconds": "",
                        "config_json": {"dense_reference": True},
                    }
                )
            else:
                row = base_row(args, model_id, "dense")
                row.update({"status": "error", "error": dense_error, "config_json": {"dense_reference": True}})
            write_row(writer, row)
            jsonl.write(json.dumps(row, sort_keys=True) + "\n")
            csv_file.flush()
            jsonl.flush()

            for method in methods:
                if method == "dense":
                    continue
                print(f"\n[method] {method}")
                row = base_row(args, model_id, method)
                try:
                    if method.split("+")[0] in EXTERNAL_METHODS:
                        result = run_external_method(args, model_id, method, train_texts, eval_texts, dense_ref)
                    else:
                        result = run_internal_method(args, model_id, method, train_texts, eval_texts, dense_ref)
                    row.update(result)
                except Exception as exc:
                    empty_cache()
                    row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "config_json": {}})
                    print(f"[error] {model_id} {method}: {row['error']}")
                write_row(writer, row)
                jsonl.write(json.dumps(row, sort_keys=True) + "\n")
                csv_file.flush()
                jsonl.flush()
    finally:
        csv_file.close()
        jsonl.close()


def build_parser():
    p = argparse.ArgumentParser(description="ARCHead benchmark runner.")
    p.add_argument("--run-group", default="main")
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "benchmark_runs" / "main"))
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-jsonl", default=None)
    p.add_argument("--cache-dir", default=str(ROOT / "activation_cache"))
    p.add_argument("--models", default="Qwen/Qwen3-8B-Base,google/gemma-4-E4B,WeiboAI/VibeThinker-3B")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["head_archead"],
        help="Methods to test (e.g. dense, head_group_int4, head_archead)",
    )
    p.add_argument("--dataset-name", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--text-columns", default="text,content,code,question,answer")
    p.add_argument("--train-text-file", default=None)
    p.add_argument("--eval-text-file", default=None)
    p.add_argument("--train-rows", type=int, default=12000)
    p.add_argument("--eval-rows", type=int, default=2500)
    p.add_argument("--train-offset", type=int, default=0)
    p.add_argument("--eval-offset", type=int, default=0)
    p.add_argument("--min-chars", type=int, default=32)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--calib-tokens", type=int, default=32768)
    p.add_argument("--eval-tokens", type=int, default=16384)
    p.add_argument("--calib-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--external-calib-rows", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument("--latency-seq-len", type=int, default=512)
    p.add_argument("--latency-batch-size", type=int, default=1)
    p.add_argument("--latency-warmup", type=int, default=3)
    p.add_argument("--latency-iters", type=int, default=10)
    p.add_argument("--triton-bench", action="store_true")
    p.add_argument("--triton-batch", type=int, default=128)
    p.add_argument("--triton-warmup", type=int, default=10)
    p.add_argument("--triton-iters", type=int, default=50)
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging and progress bars.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
