from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def plot_quality_size(df: pd.DataFrame, out_dir: Path):
    good = df[df["status"].eq("ok")].copy()
    if good.empty:
        return
    good["relative_ppl"] = finite_numeric(good["relative_ppl"])
    good["known_byte_ratio_vs_bf16"] = finite_numeric(good["known_byte_ratio_vs_bf16"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method, sub in good.dropna(subset=["relative_ppl", "known_byte_ratio_vs_bf16"]).groupby("method"):
        ax.scatter(sub["known_byte_ratio_vs_bf16"], sub["relative_ppl"], s=42, label=method, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Known compressed byte ratio vs bf16 components")
    ax.set_ylabel("Relative perplexity vs dense")
    ax.set_title("Quality-size tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_size_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_method_ranking(df: pd.DataFrame, out_dir: Path):
    good = df[df["status"].eq("ok")].copy()
    if good.empty:
        return
    good["relative_ppl"] = finite_numeric(good["relative_ppl"])
    rank = good.groupby("method")["relative_ppl"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(rank))))
    ax.barh(rank.index, rank.values, color="#4C78A8")
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Mean relative perplexity vs dense")
    ax.set_title("Mean method ranking across completed runs")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "method_ranking.png", dpi=220)
    plt.close(fig)


def plot_latency(df: pd.DataFrame, out_dir: Path):
    good = df[df["status"].eq("ok")].copy()
    good["tokens_per_sec"] = finite_numeric(good["tokens_per_sec"])
    if good["tokens_per_sec"].notna().sum() == 0:
        return
    speed = good.groupby("method")["tokens_per_sec"].mean().dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(speed))))
    ax.barh(speed.index, speed.values, color="#59A14F")
    ax.set_xlabel("Tokens/sec")
    ax.set_title("Forward throughput")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_tokens_per_sec.png", dpi=220)
    plt.close(fig)


def latex_table(df: pd.DataFrame, out_dir: Path):
    good = df[df["status"].eq("ok")].copy()
    if good.empty:
        return
    for col in ["relative_ppl", "delta_ce", "known_byte_ratio_vs_bf16", "head_byte_ratio_vs_bf16", "ff_byte_ratio_vs_bf16", "tokens_per_sec"]:
        good[col] = finite_numeric(good[col])
    cols = [
        "model_id",
        "method",
        "relative_ppl",
        "delta_ce",
        "known_byte_ratio_vs_bf16",
        "head_byte_ratio_vs_bf16",
        "ff_byte_ratio_vs_bf16",
        "tokens_per_sec",
    ]
    table = good[cols].sort_values(["model_id", "relative_ppl", "method"])
    (out_dir / "main_table.tex").write_text(
        table.to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=True),
        encoding="utf-8",
    )


def summarize_errors(df: pd.DataFrame, out_dir: Path):
    bad = df[~df["status"].eq("ok")]
    if bad.empty:
        (out_dir / "errors.md").write_text("No failed rows.\n", encoding="utf-8")
        return
    lines = ["# Failed Rows\n"]
    for _, row in bad.iterrows():
        lines.append(f"- `{row.get('model_id')}` / `{row.get('method')}`: {row.get('error')}")
    (out_dir / "errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="outputs/benchmark_runs")
    ap.add_argument("--out-dir", default="outputs/summary")
    args = ap.parse_args()
    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("**/metrics.jsonl"))
    if not files:
        raise SystemExit(f"No metrics.jsonl found below {root}")
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df.to_csv(out_dir / "all_metrics.csv", index=False)
    ok = df[df["status"].eq("ok")].copy()
    if not ok.empty:
        for col in ["relative_ppl", "delta_ce", "known_byte_ratio_vs_bf16", "tokens_per_sec"]:
            ok[col] = finite_numeric(ok[col])
        summary = (
            ok.groupby(["run_group", "method"], dropna=False)
            .agg(
                n=("method", "size"),
                mean_relative_ppl=("relative_ppl", "mean"),
                mean_delta_ce=("delta_ce", "mean"),
                mean_known_byte_ratio=("known_byte_ratio_vs_bf16", "mean"),
                mean_tokens_per_sec=("tokens_per_sec", "mean"),
            )
            .reset_index()
            .sort_values(["run_group", "mean_relative_ppl", "method"])
        )
        summary.to_csv(out_dir / "summary_by_method.csv", index=False)
    plot_quality_size(df, out_dir)
    plot_method_ranking(df, out_dir)
    plot_latency(df, out_dir)
    latex_table(df, out_dir)
    summarize_errors(df, out_dir)
    print(f"[aggregate] wrote {out_dir}")


if __name__ == "__main__":
    main()
