"""Refresh the compact analyses requested in ``rough_notes.md``.

This script uses retained WP1/WP5 traces, the six recorded live-run CSVs used by
WP6, and the raw dataset-1--8 command logs. It does not retrain or retune any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROLLING_WINDOW = 20
EVALUATION_SAMPLES = 300
INITIALIZATION_STEPS = (1, 15, 30, 45)
TARGET_SPEED_COUNTS_PER_S = 200.0
MEAN_STEP_DISTANCE_M = 0.1118

LIVE_FILES = {
    "Square": ("final_rnn_square_m3_2.csv", "final_encoder_square_m3.csv"),
    "Hourglass": ("final_rnn_hourglass_m3.csv", "final_encoder_hourglass_m3.csv"),
    "Random": ("final_rnn_random_m3_2.csv", "final_encoder_random_m3.csv"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_absolute_error_deg(estimate, truth):
    return np.abs((np.asarray(estimate) - np.asarray(truth) + 180.0) % 360.0 - 180.0)


def sustained_crossover(left, right):
    valid = left.notna() & right.notna()
    better = (left < right).to_numpy()
    candidates = [i for i in np.flatnonzero(valid.to_numpy() & better) if better[i:].all()]
    if not candidates:
        return None
    return int(candidates[0])


def refresh_initialization(output_root: Path):
    folder = output_root / "WP5_initialization"
    traces = pd.read_csv(folder / "segment_error_traces.csv")
    rows = []
    for metric, column, unit in (
        ("Position median", "position_error_m", "m"),
        ("Position $P_{90}$", "position_error_m", "m"),
        ("Heading median", "absolute_heading_error_deg", "deg"),
        ("Heading $P_{90}$", "absolute_heading_error_deg", "deg"),
    ):
        quantile = 0.90 if "$P_{90}$" in metric else 0.50
        for initialization in ("Vive known pose", "Two-fix VLP"):
            block = traces[traces.initialization == initialization]
            row = {"metric": metric, "unit": unit, "initialization": initialization}
            for step in INITIALIZATION_STEPS:
                values = block.loc[block.evaluation_step == step, column]
                row[f"step_{step}"] = float(values.quantile(quantile))
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "initialization_step_summary.csv", index=False)

    labels = {"Vive known pose": "Known Vive pose", "Two-fix VLP": "Two-fix VLP"}
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Metric & Initialization & Step 1 & Step 15 & Step 30 & Step 45 \\\\",
        "\\midrule",
    ]
    for row in summary.itertuples(index=False):
        decimals = 4 if row.unit == "m" else 2
        values = " & ".join(f"{getattr(row, f'step_{step}'):.{decimals}f}" for step in INITIALIZATION_STEPS)
        lines.append(f"{row.metric} & {labels[row.initialization]} & {values} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (folder / "initialization_step_table.tex").write_text("\n".join(lines))
    return summary


def refresh_crossover(robot_root: Path, output_root: Path):
    folder = output_root / "WP6_crossover"
    trace_rows, result_rows, sources = [], [], {}

    def add_trace(path_name, metric, lstm_error, encoder_error, evaluation_samples):
        lstm_error = pd.Series(np.asarray(lstm_error, dtype=float)).reset_index(drop=True)
        encoder_error = pd.Series(np.asarray(encoder_error, dtype=float)).reset_index(drop=True)
        lstm_roll = lstm_error.rolling(ROLLING_WINDOW).median()
        encoder_roll = encoder_error.rolling(ROLLING_WINDOW).median()
        index = sustained_crossover(lstm_roll, encoder_roll)
        if index is None:
            raise AssertionError(f"No sustained {metric} crossover for {path_name}")
        step = index + 1
        result_rows.append({
            "path": path_name,
            "metric": metric.capitalize(),
            "crossover_status": "Sustained crossover observed",
            "crossover_evaluation_step_1_based": step,
            "estimated_distance_m": step * MEAN_STEP_DISTANCE_M,
            "lstm_rolling_median_at_crossover": lstm_roll.iloc[index],
            "encoder_rolling_median_at_crossover": encoder_roll.iloc[index],
            "rolling_window_steps": ROLLING_WINDOW,
            "evaluation_samples": evaluation_samples,
            "lstm_final_rolling_median": lstm_roll.iloc[-1],
            "encoder_final_rolling_median": encoder_roll.iloc[-1],
            "error_unit": "m" if metric == "position" else "deg",
        })
        for index_row in range(evaluation_samples):
            trace_rows.append({
                "path": path_name,
                "metric": metric.capitalize(),
                "evaluation_step_1_based": index_row + 1,
                "trace_index_0_based": index_row,
                "lstm_error": lstm_error.iloc[index_row],
                "encoder_error": encoder_error.iloc[index_row],
                "lstm_rolling_median": lstm_roll.iloc[index_row],
                "encoder_rolling_median": encoder_roll.iloc[index_row],
                "estimated_distance_m": (index_row + 1) * MEAN_STEP_DISTANCE_M,
                "error_unit": "m" if metric == "position" else "deg",
            })

    live_root = robot_root / "notebooks/exploratory"
    for path_name, (lstm_name, encoder_name) in LIVE_FILES.items():
        lstm_path, encoder_path = live_root / lstm_name, live_root / encoder_name
        lstm = pd.read_csv(lstm_path).iloc[20:320].reset_index(drop=True)
        encoder = pd.read_csv(encoder_path).iloc[:300].reset_index(drop=True)
        if len(lstm) != EVALUATION_SAMPLES or len(encoder) != EVALUATION_SAMPLES:
            raise AssertionError(f"Unexpected analysis-window length for {path_name}")
        sources[path_name] = {
            "lstm_file": str(lstm_path), "lstm_sha256": sha256(lstm_path),
            "encoder_file": str(encoder_path), "encoder_sha256": sha256(encoder_path),
        }
        errors = {
            "position": (
                np.hypot(lstm.rnn_x - lstm.real_x, lstm.rnn_y - lstm.real_y),
                np.hypot(encoder.encoder_x - encoder.real_x, encoder.encoder_y - encoder.real_y),
            ),
            "heading": (
                circular_absolute_error_deg(lstm.rnn_heading, lstm.real_heading),
                circular_absolute_error_deg(encoder.encoder_heading, encoder.real_heading),
            ),
        }
        for metric, (lstm_error, encoder_error) in errors.items():
            add_trace(path_name, metric, lstm_error, encoder_error, EVALUATION_SAMPLES)

    held_out_path = output_root / "WP1_ekf_comparison/held_out_error_traces.csv"
    held_out = pd.read_csv(held_out_path)
    sources["Offline held-out"] = {
        "error_trace_file": str(held_out_path),
        "error_trace_sha256": sha256(held_out_path),
    }
    held_lstm = held_out[held_out.model == "LSTM"].sort_values("evaluation_step")
    held_encoder = held_out[held_out.model == "Encoder odometry"].sort_values("evaluation_step")
    if len(held_lstm) != EVALUATION_SAMPLES or len(held_encoder) != EVALUATION_SAMPLES:
        raise AssertionError("Unexpected held-out crossover trace length")
    add_trace(
        "Offline held-out",
        "position",
        held_lstm.position_error_m,
        held_encoder.position_error_m,
        EVALUATION_SAMPLES,
    )
    add_trace(
        "Offline held-out",
        "heading",
        held_lstm.absolute_heading_error_deg,
        held_encoder.absolute_heading_error_deg,
        EVALUATION_SAMPLES,
    )

    results = pd.DataFrame(result_rows)
    traces = pd.DataFrame(trace_rows)
    results.to_csv(folder / "crossover_results.csv", index=False)
    traces.to_csv(folder / "rolling_error_traces.csv", index=False)
    """

    lines = ["\\begin{tabular}{llrrc}", "\\toprule", "Path & Metric & Step & Index & Distance (m) \\\", "\\midrule"]
    for row in results.itertuples(index=False):
        lines.append(f"{row.path} & {row.metric} & {row.crossover_evaluation_step_1_based} & {row.crossover_trace_index_0_based} & {row.distance_min_m:.2f}--{row.distance_max_m:.2f} \\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (folder / "crossover_table.tex").write_text("\n".join(lines))
    """
    row_end = "\\\\"
    lines = [
        "\\begin{tabular}{llrc}",
        "\\toprule",
        f"Run & Metric & Step & Distance (m) {row_end}",
        "\\midrule",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"{row.path} & {row.metric} & "
            f"{row.crossover_evaluation_step_1_based} & "
            f"{row.estimated_distance_m:.2f} {row_end}"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (folder / "crossover_table.tex").write_text("\n".join(lines))

    def plot_crossovers(path_subset, output_name, figsize):
        fig, axes = plt.subplots(len(path_subset), 2, figsize=figsize, sharex=True)
        axes = np.atleast_2d(axes)
        for row_index, path_name in enumerate(path_subset):
            for column_index, metric in enumerate(("Position", "Heading")):
                axis = axes[row_index, column_index]
                frame = traces[(traces.path == path_name) & (traces.metric == metric)]
                axis.plot(frame.evaluation_step_1_based, frame.lstm_rolling_median,
                          color="#0072B2", linewidth=1.35)
                axis.plot(frame.evaluation_step_1_based, frame.encoder_rolling_median,
                          color="#D55E00", linewidth=1.35)
                result = results[(results.path == path_name) & (results.metric == metric)].iloc[0]
                cross_step = int(result.crossover_evaluation_step_1_based)
                marker_style = "--" if metric == "Position" else ":"
                axis.axvline(cross_step, color="#333333", linestyle=marker_style,
                             linewidth=1.1, alpha=0.9)
                axis.annotate(
                    str(cross_step),
                    xy=(cross_step, 0.98),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -2),
                    textcoords="offset points",
                    rotation=90,
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    color="#333333",
                )
                axis.set_yscale("log")
                axis.set_xlim(1, EVALUATION_SAMPLES)
                axis.grid(alpha=0.22, linewidth=0.55)
                if row_index == 0:
                    axis.set_title(f"{metric} error", fontsize=9)
                if column_index == 0:
                    axis.set_ylabel(
                        f"{path_name}\nError ({'m' if metric == 'Position' else 'deg'})",
                        fontsize=8,
                    )
                if row_index == len(path_subset) - 1:
                    axis.set_xlabel("Evaluation step", fontsize=8)
                axis.tick_params(labelsize=7)

        legend_handles = [
            Line2D([0], [0], color="#0072B2", linewidth=1.35, label="LSTM"),
            Line2D([0], [0], color="#D55E00", linewidth=1.35, label="Encoder odometry"),
            Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.1,
                   label="Position crossover"),
            Line2D([0], [0], color="#333333", linestyle=":", linewidth=1.1,
                   label="Heading crossover"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, 0.015), frameon=False,
                   fontsize=8, columnspacing=2.2, handlelength=2.2)
        fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.7, w_pad=0.8)
        fig.savefig(folder / output_name, dpi=240, bbox_inches="tight")
        plt.close(fig)

    plot_crossovers(
        ["Offline held-out", "Square", "Hourglass", "Random"],
        "crossover_rolling_median.png",
        (12.5, 8.0),
    )
    plot_crossovers(
        ["Square", "Hourglass", "Random"],
        "crossover_live_paths_annotated.png",
        (12.5, 6.4),
    )

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "WP6 position and heading sustained crossover horizon",
        "rolling_window_steps": ROLLING_WINDOW,
        "evaluation_samples": EVALUATION_SAMPLES,
        "lstm_csv_rows": "20--319 inclusive",
        "encoder_csv_rows": "0--299 inclusive",
        "crossover_rule": "earliest step at which the trailing median LSTM error is strictly below odometry and remains below it through step 300",
        "distance_rule": "one-based crossover step multiplied by the measured mean robot step length",
        "mean_step_distance_m": MEAN_STEP_DISTANCE_M,
        "sources": sources,
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return results


def refresh_collection_time(robot_root: Path, output_root: Path):
    folder = output_root / "WP7_cost_latency"
    raw_root = robot_root / "data/raw/experiments/Robot"
    command_pattern = re.compile(r"([A-Z_]+):\s*([-+0-9.eE]+)")
    totals, source_rows = {}, []
    for run in range(8):
        source = raw_root / f"exp1_{run}.csv"
        source_rows.append({"file": str(source), "sha256": sha256(source)})
        with source.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle, delimiter="|"):
                match = command_pattern.fullmatch((row.get("last_cmd") or "").strip())
                if match:
                    command, value = match.group(1), abs(float(match.group(2)))
                    record = totals.setdefault(command, {"commands": 0, "encoder_counts": 0.0})
                    record["commands"] += 1
                    record["encoder_counts"] += value
    rows = []
    for command, record in sorted(totals.items()):
        seconds = record["encoder_counts"] / TARGET_SPEED_COUNTS_PER_S
        rows.append({"component": command, **record, "nominal_motion_seconds": seconds, "nominal_motion_hours": seconds / 3600})
    command_summary = pd.DataFrame(rows)
    command_summary.to_csv(folder / "collection_command_time.csv", index=False)
    protocol_hours = 10.95
    motion_hours = float(command_summary.nominal_motion_hours.sum())
    estimate = pd.DataFrame([
        {"component": "Explicit acquisition and transmission delays", "hours": protocol_hours, "basis": "Archived sleeps and median frame-size pacing"},
        {"component": "Commanded motion at 200 encoder counts/s", "hours": motion_hours, "basis": "Absolute MOVE/TURN/CAL counts in datasets 1--8"},
        {"component": "Approximate collection duration", "hours": protocol_hours + motion_hours, "basis": "Sum of non-overlapping protocol-delay and nominal-motion terms"},
    ])
    estimate.to_csv(folder / "collection_time_estimate.csv", index=False)
    (folder / "collection_time_provenance.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_speed_counts_per_s": TARGET_SPEED_COUNTS_PER_S,
        "protocol_delay_hours": protocol_hours,
        "nominal_motion_hours": motion_hours,
        "approximate_total_hours": protocol_hours + motion_hours,
        "scope": "Approximate source-derived duration; excludes unmodelled retries, scheduling, and network round-trip overhead.",
        "sources": source_rows,
    }, indent=2) + "\n")
    return estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.revision_root / "revision_round_1/analysis/outputs"
    print(refresh_initialization(output_root).to_string(index=False))
    print(refresh_crossover(args.robot_root, output_root).to_string(index=False))
    print(refresh_collection_time(args.robot_root, output_root).to_string(index=False))


if __name__ == "__main__":
    main()
