"""Redraw the RSS-feature sensitivity plot from retained results."""

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
                        default=folder / "systematic_gain_results.csv")
    parser.add_argument("--output", type=Path,
                        default=folder / "all_channel_gain_sensitivity.png")
    args = parser.parse_args()
    results = pd.read_csv(args.source_csv)
    common = results[results.gain_target.isin(["none", "all LEDs"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for method, group in common.groupby("method"):
        group = group.sort_values("gain_factor")
        axes[0].plot(group.gain_factor, group.position_median_m,
                     marker="o", label=method)
        axes[1].plot(group.gain_factor, group.heading_median_deg,
                     marker="o", label=method)
    axes[0].set(xlabel="Common RSS scaling factor",
                ylabel="Median position error (m)")
    axes[1].set(xlabel="Common RSS scaling factor",
                ylabel="Median heading error (degrees)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
