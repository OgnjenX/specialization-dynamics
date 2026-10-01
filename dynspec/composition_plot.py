"""Reporting-only plot of frozen composition summary statistics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_summary(summary):
    points = summary["means"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "module-composition-v2",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    styles = {
        "local": ("Strongly local", "#424242", "s"),
        "uniform": ("Uniform sparse", "#7a6995", "o"),
        "exponential": ("Exponential", "#cf771b", "^"),
        "power": ("Power law", "#20857b", "D"),
    }
    for ax, (view, budgets, title) in zip(
        axes,
        [
            ("edges", [14, 18], "Fixed directed edge count"),
            ("cost", [22, 26], "Fixed total cable cost"),
        ],
    ):
        for rule, (label, color, marker) in styles.items():
            vals = [points[f"{view}:{b}:{rule}"] for b in budgets]
            x = np.array([p["cable_cost"] for p in vals])
            y = np.array([p["held"] * 100 for p in vals])
            err = np.array(
                [
                    [(p["held"] - p["held_ci95"][0]) * 100 for p in vals],
                    [(p["held_ci95"][1] - p["held"]) * 100 for p in vals],
                ]
            )
            ax.errorbar(
                x,
                y,
                yerr=err,
                color=color,
                marker=marker,
                markersize=6,
                capsize=3,
                linewidth=1.5,
                label=label,
            )
            if rule == "local":
                for xx, yy, b in zip(x, y, budgets):
                    ax.annotate(
                        str(b),
                        (xx, yy),
                        xytext=(5, 6),
                        textcoords="offset points",
                        color=color,
                        fontsize=8,
                    )
        ax.set_title(title)
        ax.set_xlabel("Summed directed module cable length")
        ax.set_ylabel("Held-out composition accuracy (%)")
        ax.grid(axis="y", alpha=0.18)
        ax.set_ylim(0, 100)
        ax.axhline(80, color="#999999", linestyle=":", linewidth=0.9)
        if view == "cost":
            ax.text(
                0.98,
                81,
                "80% criterion",
                transform=ax.get_yaxis_transform(),
                ha="right",
                fontsize=8,
                color="#666666",
            )
        if view == "cost":
            ax.set_xticks(budgets)
            ax.set_xlim(20, 28)
    fig.suptitle("Generalization versus module-level wiring cost", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot completed module-level composition results."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("module_pareto.svg"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "Choose a fresh figure path to preserve earlier artifacts"
        )
    figure = plot_summary(json.loads(args.summary.read_text()))
    figure.savefig(args.output, metadata={"Date": None})
    plt.close(figure)
    print(
        f"Saved {args.output}; nominal 95% seed intervals average repeated splits within seeds."
    )


if __name__ == "__main__":
    main()
