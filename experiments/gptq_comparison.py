from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from archead.data import load_texts
from archead.lm_head_methods import (
    compress_gptq,
    compress_lm_head,
    extract_lm_cache,
    lm_config,
)
from archead.modeling import load_model
from archead.utils import empty_cache, parse_dtype, safe_id, set_seed


METHODS = ("gptq", "archead")
CSV_FIELDS = (
    "model_id",
    "method",
    "calibration_seed",
    "calib_tokens",
    "calibration_split",
    "evaluation_split",
    "eval_tokens",
    "compression_seconds",
    "head_bytes",
    "head_ratio_vs_bf16",
    "dense_ce",
    "ce",
    "delta_ce",
    "dense_ppl",
    "ppl",
    "relative_ppl",
    "dense_top1_in_comp_top5",
    "top5_jaccard",
    "kl_dense_to_compressed",
    "logit_mse",
    "logit_cosine",
)

SUMMARY_METRICS = (
    "relative_ppl",
    "delta_ce",
    "dense_top1_in_comp_top5",
    "top5_jaccard",
    "kl_dense_to_compressed",
    "logit_mse",
    "logit_cosine",
    "compression_seconds",
)


def parse_int_csv(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers.")
    return values


def parse_seed_csv(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("Expected non-negative comma-separated seeds.")
    return values


def cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_exp(value: float) -> float:
    """Exponentiate a CE/delta-CE value without aborting long sweeps."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def unload(*objects: object) -> None:
    del objects
    gc.collect()
    empty_cache()


def default_cache_path(args: argparse.Namespace) -> Path:
    name = (
        f"{safe_id(args.model)}__{safe_id(args.dataset_name)}"
        f"__trainpool{args.calib_pool_tokens}__{safe_id(args.eval_split)}"
        f"{args.eval_tokens}__seq{args.seq_len}__cache-seed{args.cache_seed}.pt"
    )
    return Path(args.cache_dir) / name


def load_or_build_cache(
    args: argparse.Namespace,
) -> tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    cache_path = Path(args.cache_path) if args.cache_path else default_cache_path(args)
    if cache_path.exists():
        print(f"[cache] loading {cache_path}")
        obj = torch.load(cache_path, map_location="cpu")
        if not isinstance(obj, dict) or "calib_h" not in obj or "eval_h" not in obj:
            raise RuntimeError(
                "This is a legacy single-split cache. Rebuild with --build-cache so "
                "calibration comes from train and evaluation comes from the test split."
            )
        cache = obj
        if cache["calib_h"].shape[0] < max(args.calib_sizes):
            raise RuntimeError(
                f"Calibration cache has {cache['calib_h'].shape[0]} tokens but "
                f"{max(args.calib_sizes)} are required."
            )
        if cache["eval_h"].shape[0] < args.eval_tokens:
            raise RuntimeError(
                f"Evaluation cache has {cache['eval_h'].shape[0]} tokens but "
                f"{args.eval_tokens} are required."
            )
        return cache, None

    if not args.build_cache:
        raise FileNotFoundError(
            f"Activation cache not found: {cache_path}\n"
            "Pass --build-cache to create it on Colab, or --cache-path FILE to reuse one."
        )

    print(
        f"[cache] building train calibration pool ({args.calib_pool_tokens}) and "
        f"{args.eval_split} evaluation set ({args.eval_tokens})"
    )
    calib_texts = load_texts(
        dataset=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_columns=("text", "content", "code", "question", "answer"),
        max_rows=args.train_rows,
        min_chars=32,
        seed=args.cache_seed,
        streaming=args.streaming,
        row_offset=0,
    )
    eval_texts = load_texts(
        dataset=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_columns=("text", "content", "code", "question", "answer"),
        max_rows=args.eval_rows,
        min_chars=32,
        seed=args.cache_seed + 10000,
        streaming=args.streaming,
        row_offset=0,
    )
    model, processor, kind = load_model(
        args.model,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise RuntimeError("The model has no accessible LM-head weight.")
    weight = output.weight.detach().cpu().float()
    calib_cache = extract_lm_cache(
        model,
        processor,
        kind,
        calib_texts,
        device=args.device,
        seq_len=args.seq_len,
        tokens=args.calib_pool_tokens,
        batch_size=args.cache_batch_size,
    )
    eval_cache = extract_lm_cache(
        model,
        processor,
        kind,
        eval_texts,
        device=args.device,
        seq_len=args.seq_len,
        tokens=args.eval_tokens,
        batch_size=args.cache_batch_size,
    )
    cache = {
        "calib_h": calib_cache["h"],
        "calib_y": calib_cache.get("y"),
        "eval_h": eval_cache["h"],
        "eval_y": eval_cache.get("y"),
        "metadata": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "calibration_split": args.train_split,
            "evaluation_split": args.eval_split,
            "calib_pool_tokens": args.calib_pool_tokens,
            "eval_tokens": args.eval_tokens,
            "cache_seed": args.cache_seed,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[cache] saved {cache_path}")
    del model, processor, output
    unload()
    return cache, weight


def load_lm_head_weight(args: argparse.Namespace) -> torch.Tensor:
    print(f"[model] loading {args.model} to read the LM-head")
    model, processor, _ = load_model(
        args.model,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise RuntimeError("The model has no accessible LM-head weight.")
    weight = output.weight.detach().cpu().float()
    del model, processor, output
    unload()
    return weight


def compress_timed(
    method: str,
    weight_cpu: torch.Tensor,
    h_calib_cpu: torch.Tensor,
    args: argparse.Namespace,
):
    unload()
    cuda_sync(args.device)
    start = time.perf_counter()
    if method == "gptq":
        head = compress_gptq(
            weight_cpu,
            h_calib_cpu,
            bits=4,
            group_size=args.group_size,
            damp=args.gptq_damp,
            block_size=args.gptq_block_size,
            device=args.device,
        )
    elif method == "archead":
        cfg = lm_config("archead_unpacked", args.model)
        head = compress_lm_head(weight_cpu, h_calib_cpu, cfg, device=args.device)
    else:
        raise ValueError(method)
    cuda_sync(args.device)
    elapsed = time.perf_counter() - start
    return head, elapsed


@torch.no_grad()
def evaluate(
    head,
    weight_cpu: torch.Tensor,
    h_eval_cpu: torch.Tensor,
    y_eval_cpu: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, float]:
    weight = weight_cpu.to(device=args.device, dtype=torch.float16)
    totals = {
        "tokens": 0,
        "dense_nll": 0.0,
        "compressed_nll": 0.0,
        "top5_hit": 0.0,
        "top5_jaccard": 0.0,
        "kl": 0.0,
        "mse": 0.0,
        "cosine": 0.0,
    }
    for start in range(0, h_eval_cpu.shape[0], args.eval_chunk_size):
        stop = min(start + args.eval_chunk_size, h_eval_cpu.shape[0])
        h = h_eval_cpu[start:stop].to(device=args.device, dtype=torch.float16)
        dense_logits = h @ weight.T
        compressed_logits = head(h)
        batch = h.shape[0]

        dense_top1 = dense_logits.argmax(dim=-1)
        dense_top5 = dense_logits.topk(5, dim=-1).indices
        compressed_top5 = compressed_logits.topk(5, dim=-1).indices
        intersection = (
            dense_top5.unsqueeze(2) == compressed_top5.unsqueeze(1)
        ).any(dim=2).sum(dim=1)

        totals["tokens"] += batch
        totals["top5_hit"] += (
            dense_top1.unsqueeze(1) == compressed_top5
        ).any(dim=1).sum().item()
        totals["top5_jaccard"] += (intersection.float() / (10 - intersection)).sum().item()

        dense_float = dense_logits.float()
        compressed_float = compressed_logits.float()
        dense_logp = F.log_softmax(dense_float, dim=-1)
        compressed_logp = F.log_softmax(compressed_float, dim=-1)
        dense_prob = dense_logp.exp()
        totals["kl"] += (
            dense_prob * (dense_logp - compressed_logp)
        ).sum(dim=-1).sum().item()
        # Sum over the vocabulary, then average over tokens. This matches the
        # Use the same logit-MSE convention as the fidelity benchmark.
        totals["mse"] += (
            dense_float - compressed_float
        ).square().sum(dim=-1).sum().item()
        totals["cosine"] += F.cosine_similarity(
            dense_float, compressed_float, dim=-1
        ).sum().item()

        if y_eval_cpu is not None:
            labels = y_eval_cpu[start:stop].to(args.device)
            totals["dense_nll"] += F.cross_entropy(
                dense_float, labels, reduction="sum"
            ).item()
            totals["compressed_nll"] += F.cross_entropy(
                compressed_float, labels, reduction="sum"
            ).item()

        del h, dense_logits, compressed_logits, dense_float, compressed_float

    count = totals["tokens"]
    result = {
        "eval_tokens": count,
        "dense_top1_in_comp_top5": totals["top5_hit"] / count,
        "top5_jaccard": totals["top5_jaccard"] / count,
        "kl_dense_to_compressed": totals["kl"] / count,
        "logit_mse": totals["mse"] / count,
        "logit_cosine": totals["cosine"] / count,
    }
    if y_eval_cpu is not None:
        dense_ce = totals["dense_nll"] / count
        ce = totals["compressed_nll"] / count
        result.update(
            {
                "dense_ce": dense_ce,
                "ce": ce,
                "delta_ce": ce - dense_ce,
                "dense_ppl": safe_exp(dense_ce),
                "ppl": safe_exp(ce),
                "relative_ppl": safe_exp(ce - dense_ce),
            }
        )
    else:
        result.update(
            {
                key: math.nan
                for key in (
                    "dense_ce",
                    "ce",
                    "delta_ce",
                    "dense_ppl",
                    "ppl",
                    "relative_ppl",
                )
            }
        )
    del weight
    unload()
    return result


def append_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def completed_runs(path: Path) -> set[tuple[str, int, int]]:
    completed: set[tuple[str, int, int]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(
                (
                    str(row["method"]),
                    int(row["calibration_seed"]),
                    int(row["calib_tokens"]),
                )
            )
    return completed


def scaled_mean_std(values: list[float]) -> tuple[float, float]:
    """Compute sample mean/std without squaring huge values in raw scale."""
    if not values:
        return math.nan, math.nan
    if not all(math.isfinite(value) for value in values):
        return math.inf, math.nan
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0, 0.0
    normalized = [value / scale for value in values]
    mean_normalized = sum(normalized) / len(normalized)
    variance_normalized = (
        sum((value - mean_normalized) ** 2 for value in normalized)
        / (len(normalized) - 1)
        if len(normalized) > 1
        else 0.0
    )
    return mean_normalized * scale, math.sqrt(variance_normalized) * scale


def write_summary(jsonl_path: Path, summary_path: Path) -> None:
    groups: Dict[tuple[str, int], list[Dict[str, object]]] = {}
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["method"]), int(row["calib_tokens"]))
            groups.setdefault(key, []).append(row)

    fields = [
        "model_id",
        "method",
        "calib_tokens",
        "n_seeds",
        "catastrophic_runs",
    ]
    for metric in SUMMARY_METRICS:
        fields.extend(
            (f"{metric}_mean", f"{metric}_std", f"{metric}_median")
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (method, calib_tokens), rows in sorted(
            groups.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            out: Dict[str, object] = {
                "model_id": rows[0]["model_id"],
                "method": method,
                "calib_tokens": calib_tokens,
                "n_seeds": len(rows),
                "catastrophic_runs": sum(
                    not math.isfinite(float(row["relative_ppl"]))
                    or float(row["relative_ppl"]) > 2.0
                    for row in rows
                ),
            }
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in rows]
                mean, std = scaled_mean_std(values)
                finite_sorted = sorted(
                    value for value in values if math.isfinite(value)
                )
                middle = len(finite_sorted) // 2
                median = (
                    finite_sorted[middle]
                    if len(finite_sorted) % 2
                    else (
                        finite_sorted[middle - 1] + finite_sorted[middle]
                    )
                    / 2
                ) if finite_sorted else math.nan
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
                out[f"{metric}_median"] = median
            writer.writerow(out)


def run(args: argparse.Namespace) -> None:
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)
    summary_csv = Path(args.summary_csv)
    if args.summary_only:
        write_summary(output_jsonl, summary_csv)
        print(f"[done] summary: {summary_csv}")
        return
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU.")
    if args.calib_pool_tokens < max(args.calib_sizes):
        raise ValueError("--calib-pool-tokens must be at least max(--calib-sizes).")
    set_seed(args.cache_seed)
    cache, weight = load_or_build_cache(args)
    if weight is None:
        weight = load_lm_head_weight(args)

    h_pool = cache["calib_h"]
    h_eval = cache["eval_h"][: args.eval_tokens]
    y_all_eval = cache.get("eval_y")
    y_eval = y_all_eval[: args.eval_tokens] if y_all_eval is not None else None
    print(
        f"[split] calibration={args.train_split} pool={len(h_pool)}; "
        f"evaluation={args.eval_split} tokens={len(h_eval)}"
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_csv.unlink(missing_ok=True)
        output_jsonl.unlink(missing_ok=True)
        summary_csv.unlink(missing_ok=True)
    completed = completed_runs(output_jsonl)

    for calibration_seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(calibration_seed)
        permutation = torch.randperm(len(h_pool), generator=generator)
        seeded_pool = h_pool[permutation]
        for calib_tokens in args.calib_sizes:
            h_calib = seeded_pool[:calib_tokens]
            for method in args.methods:
                key = (method, calibration_seed, calib_tokens)
                if key in completed:
                    print(
                        f"[skip] method={method} seed={calibration_seed} "
                        f"calibration_tokens={calib_tokens} already exists in JSONL"
                    )
                    continue
                set_seed(calibration_seed)
                print(
                    f"\n[run] method={method} seed={calibration_seed} "
                    f"calibration_tokens={calib_tokens}"
                )
                head, seconds = compress_timed(
                    method, weight, h_calib, args
                )
                metrics = evaluate(head, weight, h_eval, y_eval, args)
                row: Dict[str, object] = {
                    "model_id": args.model,
                    "method": method,
                    "calibration_seed": calibration_seed,
                    "calib_tokens": calib_tokens,
                    "calibration_split": args.train_split,
                    "evaluation_split": args.eval_split,
                    "compression_seconds": seconds,
                    "head_bytes": head.stats.get("total_bytes", ""),
                    "head_ratio_vs_bf16": head.stats.get("byte_ratio_vs_bf16", ""),
                    **metrics,
                }
                append_rows(output_csv, [row])
                with output_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                if not math.isfinite(float(row["relative_ppl"])):
                    print(
                        "[warning] non-finite perplexity recorded; "
                        f"ce={row['ce']}, delta_ce={row['delta_ce']}. "
                        "The sweep will continue so this instability is retained."
                    )
                print(
                    f"[result] rel_ppl={row['relative_ppl']:.6f} "
                    f"KL={row['kl_dense_to_compressed']:.6f} "
                    f"MSE={row['logit_mse']:.3f} "
                    f"time={seconds:.1f}s"
                )
                del head
                unload()

    write_summary(output_jsonl, summary_csv)
    print(f"\n[done] CSV: {output_csv}")
    print(f"[done] JSONL: {output_jsonl}")
    print(f"[done] summary: {summary_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fair GPTQ-vs-ARCHead comparison: disjoint calibration/evaluation, "
            "logit fidelity, perplexity, and compression time."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--calib-sizes",
        type=parse_int_csv,
        default=parse_int_csv("2048,4096,8192,16384"),
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_csv,
        default=parse_seed_csv("0,1,2"),
        help="Calibration sampling seeds.",
    )
    parser.add_argument("--eval-tokens", type=int, default=16384)
    parser.add_argument("--eval-chunk-size", type=int, default=64)
    parser.add_argument("--calib-pool-tokens", type=int, default=32768)
    parser.add_argument("--cache-dir", default="./activation_cache")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--cache-batch-size", type=int, default=1)
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-rows", type=int, default=12000)
    parser.add_argument("--eval-rows", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--gptq-damp", type=float, default=0.01)
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cache-seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--output-csv", default="./outputs/gptq_comparison/metrics.csv"
    )
    parser.add_argument(
        "--output-jsonl", default="./outputs/gptq_comparison/metrics.jsonl"
    )
    parser.add_argument(
        "--summary-csv", default="./outputs/gptq_comparison/summary.csv"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Regenerate summary CSV from an existing JSONL without loading a model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing CSV/JSONL instead of resuming completed method/size pairs.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
