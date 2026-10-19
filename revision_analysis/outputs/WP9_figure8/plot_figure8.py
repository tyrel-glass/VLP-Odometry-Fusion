"""Regenerate the manuscript temporal-error figure.

The held-out row preserves the known-start paper evaluation. The three live rows
use the archived estimator-specific runs. The separate LSTM--EKF figure contains
the offline classical-filter comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSITION_FLOOR_M = 1e-3
HEADING_FLOOR_DEG = 1e-1
COLORS = {
    "VLP": "#E69F00",
    "Enc. Odom.": "#7F7F7F",
    "LSTM": "#0072B2",
}
LINESTYLES = {
    "VLP": "--",
    "Enc. Odom.": "-.",
    "LSTM": "-",
}
SCENARIO_NAMES = {
    "Offline held-out": "Held-out run",
    "Square": "Square",
    "Hourglass": "Hourglass",
    "Random": "Random",
}


def wrap_degrees(values):
    values = np.asarray(values, dtype=float)
    return (values + 180.0) % 360.0 - 180.0


def live_errors(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    prefix = {"LSTM": "rnn", "VLP": "vlp", "Enc. Odom.": "encoder"}[method]
    start = 20 if method == "LSTM" else 0
    stop = start + 300
    block = frame.iloc[start:stop]
    position = np.hypot(
        block[f"{prefix}_x"].to_numpy() - block["real_x"].to_numpy(),
        block[f"{prefix}_y"].to_numpy() - block["real_y"].to_numpy(),
    )
    heading = np.abs(
        wrap_degrees(
            block[f"{prefix}_heading"].to_numpy()
            - block["real_heading"].to_numpy()
        )
    )
    return pd.DataFrame(
        {
            "step": np.arange(len(block)),
            "method": method,
            "position_error_m": position,
            "heading_error_deg": heading,
        }
    )


def load_source_data(revision_root: Path, robot_root: Path) -> pd.DataFrame:
    trace_path = (
        revision_root
        / "revision_round_1/analysis/outputs/WP1_ekf_comparison/held_out_error_traces.csv"
    )
    held = pd.read_csv(trace_path)
    held_names = {
        "VLP": "VLP",
        "Encoder odometry": "Enc. Odom.",
        "LSTM": "LSTM",
    }
    offline = held[held["model"].isin(held_names)].copy()
    offline["method"] = offline["model"].map(held_names)
    offline["scenario"] = "Held-out run"
    offline["step"] = offline.groupby("method").cumcount()
    offline = offline.rename(columns={"absolute_heading_error_deg": "heading_error_deg"})
    blocks = [
        offline[["scenario", "step", "method", "position_error_m", "heading_error_deg"]]
    ]

    live_dir = robot_root / "notebooks/exploratory"
    file_map = {
        "Square": {
            "LSTM": "final_rnn_square_m3_2.csv",
            "VLP": "final_vlp_square_m3_2.csv",
            "Enc. Odom.": "final_encoder_square_m3.csv",
        },
        "Hourglass": {
            "LSTM": "final_rnn_hourglass_m3.csv",
            "VLP": "final_vlp_hourglass_m3.csv",
            "Enc. Odom.": "final_encoder_hourglass_m3.csv",
        },
        "Random": {
            "LSTM": "final_rnn_random_m3_2.csv",
            "VLP": "final_vlp_random_m3.csv",
            "Enc. Odom.": "final_encoder_random_m3.csv",
        },
    }
    for scenario, methods in file_map.items():
        for method, filename in methods.items():
            block = live_errors(live_dir / filename, method)
            block.insert(0, "scenario", scenario)
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def load_crossovers(path: Path) -> dict[str, dict[str, int]]:
    results = pd.read_csv(path)
    crossovers: dict[str, dict[str, int]] = {}
    for _, row in results.iterrows():
        scenario = SCENARIO_NAMES[row["path"]]
        metric = row["metric"].lower()
        crossovers.setdefault(scenario, {})[metric] = int(
            row["crossover_evaluation_step_1_based"]
        )
    expected = {"position", "heading"}
    for scenario in SCENARIO_NAMES.values():
        if crossovers.get(scenario, {}).keys() < expected:
            raise ValueError(f"Missing position/heading crossover for {scenario!r}")
    return crossovers


def plot(data: pd.DataFrame, crossovers: dict[str, dict[str, int]], output: Path) -> None:
    scenarios = ["Held-out run", "Square", "Hourglass", "Random"]
    methods = ["VLP", "Enc. Odom.", "LSTM"]
    fig, axes = plt.subplots(4, 2, figsize=(7.16, 6.1), sharex="col")
    for row, scenario in enumerate(scenarios):
        subset = data[data["scenario"] == scenario]
        for col, (metric, title, floor, limits, unit) in enumerate(
            [
                ("position_error_m", "Position error", POSITION_FLOOR_M, (8e-4, 0.5), "m"),
                ("heading_error_deg", "Heading error", HEADING_FLOOR_DEG, (8e-2, 180), "deg"),
            ]
        ):
            ax = axes[row, col]
            for method in methods:
                series = subset[subset["method"] == method]
                if series.empty:
                    continue
                ax.plot(
                    series["step"],
                    np.maximum(series[metric], floor),
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=0.9,
                    alpha=0.9,
                    label=method,
                )
            crossover_metric = "position" if col == 0 else "heading"
            crossover_step = crossovers[scenario][crossover_metric]
            # Figure data use zero-based indices; the reported crossover steps
            # are one-based evaluation steps.
            crossover_x = crossover_step - 1
            ax.axvline(
                crossover_x,
                color="#222222",
                linestyle="--",
                linewidth=0.8,
                alpha=0.9,
                zorder=4,
            )
            ax.annotate(
                str(crossover_step),
                xy=(crossover_x, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=6.2,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
                zorder=5,
            )
            ax.set_yscale("log")
            ax.set_ylim(*limits)
            ax.set_xlim(0, 299)
            ax.grid(True, which="both", linestyle=":", alpha=0.35)
            ax.tick_params(axis="both", labelsize=7, pad=1)
            ax.set_ylabel(f"{scenario} ({unit})" if col == 0 else f"({unit})", fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=9, pad=4)
            if row == len(scenarios) - 1:
                ax.set_xlabel("Step index", fontsize=8)

    handles = [
        plt.Line2D([], [], color=COLORS[m], linestyle=LINESTYLES[m], label=m)
        for m in methods
    ]
    handles.append(
        plt.Line2D([], [], color="#222222", linestyle="--", linewidth=0.8,
                   label="Crossover")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=6.8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.96, bottom=0.12, hspace=0.14, wspace=0.19)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path)
    parser.add_argument("--robot-root", type=Path)
    parser.add_argument(
        "--source-csv",
        type=Path,
        help="Replot directly from retained figure8_source_data.csv.",
    )
    parser.add_argument("--crossover-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source_csv:
        data = pd.read_csv(args.source_csv)
    else:
        if args.revision_root is None or args.robot_root is None:
            parser.error("--revision-root and --robot-root are required unless --source-csv is used")
        data = load_source_data(args.revision_root, args.robot_root)
        data.to_csv(args.output_dir / "figure8_source_data.csv", index=False)
    crossover_csv = args.crossover_csv or (
        Path(__file__).resolve().parents[1]
        / "WP6_crossover/crossover_results.csv"
    )
    plot(data, load_crossovers(crossover_csv), args.output_dir / "temporal_error_logscale.png")


if __name__ == "__main__":
    main()
