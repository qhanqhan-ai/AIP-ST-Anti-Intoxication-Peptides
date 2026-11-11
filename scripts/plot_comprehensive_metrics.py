#!/usr/bin/env python3
"""Draw comprehensive per-class metrics comparison across all models."""

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
OUT = PACKAGE_ROOT / "figures" / "fig_comprehensive_class_metrics.png"
MODELS = ["LogReg", "SVM", "RF", "MLP", "CNN", "Transformer"]
CLASSES = ["Class0", "Class1", "Class3", "Class5"]
CLASS_LABELS = ["Class 0", "Class 1", "Class 3", "Class 5"]


def setup_style():
    for font_path in (Path.home() / ".fonts").glob("Arial*.ttf"):
        fontManager.addfont(str(font_path))
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def load_data():
    data = {}
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m = row["Model"]
            data[m] = row
    return data


def draw(data):
    model_colors = {
        "LogReg": "#7F7F7F",
        "SVM": "#B83EDB",
        "RF": "#E83535",
        "MLP": "#3B6DDB",
        "CNN": "#1B9E3A",
        "Transformer": "#F28E00",
    }

    metrics = [("PRE", "Precision"), ("REC", "Recall"), ("F1", "F1-score")]
    x = np.arange(len(CLASSES))
    width = 0.12

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    fig.patch.set_facecolor("white")

    for ax_idx, (met_key, met_label) in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("white")

        for m_idx, model in enumerate(MODELS):
            means = []
            stds = []
            for cls in CLASSES:
                means.append(float(data[model][f"{cls}_{met_key}_mean"]))
                stds.append(float(data[model][f"{cls}_{met_key}_std"]))
            offset = (m_idx - len(MODELS) / 2 + 0.5) * width
            is_t = model == "Transformer"
            ax.bar(
                x + offset, means, width, yerr=stds,
                color=model_colors[model],
                edgecolor="#C87400" if is_t else "#333333",
                linewidth=1.8 if is_t else 0.6,
                capsize=2,
                error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#555555"},
                label=model if ax_idx == 0 else None,
                zorder=3 if is_t else 2,
            )

        ax.set_title(met_label, fontsize=15, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_LABELS, fontsize=10, fontweight="bold")
        ax.set_ylim(0.3, 1.05)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)

    axes[0].set_ylabel("Score (mean +/- std)", fontsize=12, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9.5,
               frameon=True, fancybox=False, edgecolor="#999999",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-Class Metrics Comparison", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(OUT, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    with Image.open(OUT) as img:
        w, h = img.size
        print(f"  {OUT} ({w}x{h})")


def main():
    setup_style()
    draw(load_data())


if __name__ == "__main__":
    main()
