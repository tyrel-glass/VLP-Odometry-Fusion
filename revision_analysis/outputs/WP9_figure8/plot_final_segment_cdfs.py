"""Plot final-100-step error CDFs from the retained temporal-error CSV."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = {
    "Held-out run": ("Offline", "#9467bd", "-"),
    "Square": ("Square", "#1f77b4", "--"),
    "Hourglass": ("Hourglass", "#2ca02c", "-."),
    "Random": ("Random", "#ff7f0e", ":"),
}
METHODS = ["LSTM", "VLP", "Enc. Odom."]


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path,
                        default=folder / "figure8_source_data.csv")
    parser.add_argument("--output", type=Path,
                        default=folder / "final_segment_error_cdfs.png")
    args = parser.parse_args()
    data = pd.read_csv(args.source_csv)
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.7))
    metrics = [("position_error_m", "(m)", 0.15),
               ("heading_error_deg", "(deg)", 40)]
    for col, method in enumerate(METHODS):
        axes[0, col].set_title(method, fontsize=7)
        for row, (metric, unit, maximum) in enumerate(metrics):
            ax = axes[row, col]
            for scenario, (label, color, style) in SCENARIOS.items():
                series = data[(data.method == method) & (data.scenario == scenario)]
                series = series.sort_values("step").tail(100)
                if len(series) != 100:
                    raise ValueError(f"Expected 100 final samples: {scenario}, {method}")
                values = np.sort(series[metric].to_numpy())
                ax.plot(values, np.arange(1, 101) / 100, color=color,
                        linestyle=style, linewidth=0.8, label=label)
            ax.set(xlim=(0, maximum), ylim=(0, 1), xlabel=unit)
            if row == 0:
                ax.set_xticks([0, 0.05, 0.10])
            ax.xaxis.label.set_size(6.5)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25, linewidth=0.4)
            if col == 0:
                ax.set_ylabel("Cumulative fraction", fontsize=6.5)
            else:
                ax.tick_params(labelleft=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, fontsize=6, columnspacing=1.0, handlelength=1.8)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.2, top=0.91,
                        hspace=0.65, wspace=0.42)
    fig.savefig(args.output, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
