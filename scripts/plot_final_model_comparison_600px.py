#!/usr/bin/env python3
"""Create the 600 px model comparison figure from the model metrics summary."""

import csv
import os
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PACKAGE_ROOT / "results" / "model_metrics_summary.csv"
OUT_PNG = PACKAGE_ROOT / "figures" / "fig_final_model_comparison_600px.png"
OUT_SVG = PACKAGE_ROOT / "figures" / "fig_final_model_comparison_600px.svg"
MODELS = ["LogReg", "SVM", "RF", "MLP", "CNN", "Transformer"]


def setup_style():
    for font_path in (Path.home() / ".fonts").glob("Arial*.ttf"):
        fontManager.addfont(str(font_path))
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": 1.7,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "hatch.linewidth": 1.0,
        }
    )


def load_summary():
    summary = {}
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["Model"]
            summary[model] = {
                "F1": {"mean": float(row["Macro_F1_mean"]), "std": float(row["Macro_F1_std"])},
                "Class3_PRE": {"mean": float(row["Class3_PRE_mean"]), "std": float(row["Class3_PRE_std"])},
            }
    return summary


def draw(summary):
    colors = {
        "LogReg": "#7F7F7F",
        "SVM": "#B83EDB",
        "RF": "#4DAA67",
        "MLP": "#5B5FC7",
        "CNN": "#55A868",
        "Transformer": "#F28E00",
    }
    edges = {
        "LogReg": "#555555",
        "SVM": "#8A2DB0",
        "RF": "#2D7D46",
        "MLP": "#3F43A8",
        "CNN": "#347D4E",
        "Transformer": "#C87400",
    }
    hatches = {"LogReg": "...", "SVM": "\\\\\\", "RF": "xx", "MLP": "", "CNN": "", "Transformer": "///"}

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(MODELS))
    width = 0.34
    metrics = [("F1", "F1", -width / 2, 0.92), ("Class3_PRE", "Class 3 Precision", width / 2, 0.55)]
    for metric_key, legend_label, offset, alpha in metrics:
        for idx, model in enumerate(MODELS):
            is_transformer = model == "Transformer"
            ax.bar(
                x[idx] + offset,
                summary[model][metric_key]["mean"],
                width,
                yerr=summary[model][metric_key]["std"],
                label=legend_label if idx == 0 else None,
                color=colors[model],
                edgecolor=edges[model],
                linewidth=2.2 if is_transformer else 1.1,
                hatch=hatches[model],
                alpha=alpha,
                capsize=3,
                error_kw={"elinewidth": 1.15, "capthick": 1.15, "ecolor": "#222222"},
                zorder=3,
            )

    transformer_x = x[-1]
    transformer_f1 = summary["Transformer"]["F1"]["mean"]
    transformer_precision = summary["Transformer"]["Class3_PRE"]["mean"]
    ax.text(transformer_x - width / 2, transformer_f1 + 0.018, f"{transformer_f1:.3f}", ha="center", fontsize=10, fontweight="bold", color="#C87400")
    ax.text(transformer_x + width / 2, transformer_precision + 0.018, f"{transformer_precision:.3f}", ha="center", fontsize=10, fontweight="bold", color="#C87400")
    ax.annotate(
        "Selected by high F1\nand Class 3 precision",
        xy=(transformer_x + 0.05, max(transformer_f1, transformer_precision)),
        xytext=(3.0, 0.94),
        fontsize=9.5,
        fontweight="bold",
        color="#C87400",
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#C87400"},
        bbox={"boxstyle": "round,pad=0.28", "fc": "#FFF3DF", "ec": "#F28E00", "alpha": 0.95},
    )

    ax.set_title("Model Comparison", fontsize=20, fontweight="bold", pad=14)
    ax.set_ylabel("Score (mean +/- std)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9.5, fontweight="bold", rotation=25, ha="right")
    ax.set_ylim(0.45, 1.03)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.28, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, fancybox=False, edgecolor="#999999")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(OUT_PNG, dpi=100, facecolor="white")
    fig.savefig(OUT_SVG, format="svg", facecolor="white")
    plt.close(fig)

    with Image.open(OUT_PNG) as image:
        if image.size != (600, 600):
            image.resize((600, 600), Image.Resampling.LANCZOS).save(OUT_PNG)
    print(OUT_PNG)
    print(OUT_SVG)


def main():
    setup_style()
    draw(load_summary())


if __name__ == "__main__":
    main()
