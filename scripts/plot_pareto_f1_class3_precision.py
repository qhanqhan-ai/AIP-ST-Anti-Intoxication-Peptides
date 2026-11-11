#!/usr/bin/env python3
"""Draw the Macro-F1 vs Class 3 precision model-selection plot."""

import csv
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = PACKAGE_ROOT / "figures" / "fig_pareto_f1_class3_precision.png"
OUT_SVG = PACKAGE_ROOT / "figures" / "fig_pareto_f1_class3_precision.svg"


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
        }
    )


def main():
    setup_style()

    data = {}
    with open(PACKAGE_ROOT / "results" / "model_metrics_summary.csv") as f:
        for row in csv.DictReader(f):
            m = row["Model"]
            data[m] = {
                "f1": float(row["Macro_F1_mean"]),
                "f1_std": float(row["Macro_F1_std"]),
                "c3p": float(row["Class3_PRE_mean"]),
                "c3p_std": float(row["Class3_PRE_std"]),
            }

    colors = {
        "LogReg": "#7F7F7F",
        "SVM": "#B83EDB",
        "RF": "#4DAA67",
        "MLP": "#5B5FC7",
        "CNN": "#55A868",
        "Transformer": "#F28E00",
    }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axvspan(0.72, 0.76, color="#F28E00", alpha=0.06, zorder=0)
    ax.axhspan(0.78, 0.90, color="#F28E00", alpha=0.06, zorder=0)

    for model, row in data.items():
        is_transformer = model == "Transformer"
        ax.errorbar(
            row["f1"],
            row["c3p"],
            xerr=row["f1_std"],
            yerr=row["c3p_std"],
            fmt="o",
            markersize=12 if is_transformer else 9,
            markerfacecolor=colors[model],
            markeredgecolor="#C87400" if is_transformer else "#333333",
            markeredgewidth=2.2 if is_transformer else 1.0,
            ecolor="#C87400" if is_transformer else "#555555",
            elinewidth=1.7 if is_transformer else 1.1,
            capsize=4,
            alpha=1.0 if is_transformer else 0.86,
            zorder=4 if is_transformer else 3,
        )

        dx, dy = {
            "LogReg": (0.006, -0.018),
            "SVM": (0.006, 0.012),
            "RF": (0.006, 0.014),
            "MLP": (-0.036, -0.024),
            "CNN": (0.006, -0.018),
            "Transformer": (-0.060, 0.018),
        }[model]
        ax.text(
            row["f1"] + dx,
            row["c3p"] + dy,
            model,
            fontsize=10.5 if is_transformer else 9.5,
            fontweight="bold" if is_transformer else "normal",
            color="#C87400" if is_transformer else "#222222",
        )

    ax.annotate(
        "Pareto-preferred region",
        xy=(data["Transformer"]["f1"], data["Transformer"]["c3p"]),
        xytext=(0.645, 0.865),
        fontsize=10.5,
        fontweight="bold",
        color="#C87400",
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#C87400"},
        bbox={"boxstyle": "round,pad=0.28", "fc": "#FFF3DF", "ec": "#F28E00", "alpha": 0.95},
    )

    ax.set_title("F1 vs Class 3 Precision", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Macro F1-score (mean +/- std)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Class 3 Precision (mean +/- std)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.55, 0.77)
    ax.set_ylim(0.52, 0.92)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5)

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


if __name__ == "__main__":
    main()
