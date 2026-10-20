"""Redraw initialization-sensitivity curves from the retained prediction errors."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    folder = Path(__file__).resolve().parent
    parser.add_argument("--source-csv", type=Path,
                        default=folder / "segment_error_traces.csv")
    parser.add_argument("--output", type=Path,
                        default=folder / "initialization_over_live_segments.png")
    args = parser.parse_args()
    traces = pd.read_csv(args.source_csv)
    colours = {"Vive known pose": "#0072B2", "Two-fix VLP": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    for initialization, colour in colours.items():
        by_step = traces[traces.initialization == initialization].groupby("evaluation_step")
        position_median = by_step.position_error_m.median()
        position_p90 = by_step.position_error_m.quantile(0.90)
        heading_median = by_step.absolute_heading_error_deg.median()
        heading_p90 = by_step.absolute_heading_error_deg.quantile(0.90)
        steps = position_median.index.to_numpy()
        axes[0].plot(steps, position_median, color=colour, lw=1.0,
                     label=f"{initialization}: median")
        axes[0].plot(steps, position_p90, color=colour, lw=0.8, ls=":",
                     label=f"{initialization}: $P_{{90}}$")
        axes[1].plot(steps, heading_median, color=colour, lw=1.0)
        axes[1].plot(steps, heading_p90, color=colour, lw=0.8, ls=":")
    axes[0].set_ylabel("Position error (m)", fontsize=8)
    axes[1].set_ylabel("Heading error (deg)", fontsize=8)
    axes[1].set_xlabel("Prediction index", fontsize=8)
    axes[0].set_xlim(1, 50)
    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.grid(axis="y", color="0.9", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=6.2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
