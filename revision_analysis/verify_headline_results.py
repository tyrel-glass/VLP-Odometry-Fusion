"""Recompute revision headline statistics from retained machine-readable traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ATOL = 1e-10


def close(actual, expected, label):
    if not np.isclose(actual, expected, atol=ATOL, rtol=1e-9, equal_nan=True):
        raise AssertionError(f"{label}: recomputed {actual!r}, saved {expected!r}")


def metric_rows(trace, group_column, segment_column=None):
    rows = []
    segments = {"all": slice(None), "first_100": slice(0, 100), "last_100": slice(-100, None)}
    for group, frame in trace.groupby(group_column, sort=False):
        frame = frame.reset_index(drop=True)
        for segment, selector in segments.items():
            block = frame.iloc[selector]
            rows.append(
                {
                    group_column: group,
                    "segment": segment,
                    "samples": len(block),
                    "position_median_m": np.median(block.position_error_m),
                    "position_p90_m": np.percentile(block.position_error_m, 90),
                    "heading_median_deg": np.median(block.absolute_heading_error_deg),
                    "heading_p90_deg": np.percentile(block.absolute_heading_error_deg, 90),
                }
            )
    return pd.DataFrame(rows)


def check_rows(recomputed, saved, keys, metrics, label):
    merged = recomputed.merge(saved, on=keys, suffixes=("_new", "_saved"), validate="one_to_one")
    if len(merged) != len(saved) or len(merged) != len(recomputed):
        raise AssertionError(f"{label}: row/key mismatch")
    for _, row in merged.iterrows():
        key = "/".join(str(row[k]) for k in keys)
        for metric in metrics:
            close(row[f"{metric}_new"], row[f"{metric}_saved"], f"{label} {key} {metric}")


def verify_wp1(root):
    folder = root / "WP1_ekf_comparison"
    trace = pd.read_csv(folder / "held_out_error_traces.csv")
    saved = pd.read_csv(folder / "held_out_metrics.csv")
    recomputed = metric_rows(trace, "model")
    check_rows(
        recomputed,
        saved,
        ["model", "segment"],
        ["samples", "position_median_m", "position_p90_m", "heading_median_deg", "heading_p90_deg"],
        "WP1",
    )
    return {"models": trace.model.nunique(), "samples_per_model": int(trace.groupby("model").size().iloc[0])}


def verify_wp2(root):
    folder = root / "WP2_stress_test"
    events = pd.read_csv(folder / "event_results.csv")
    saved = pd.read_csv(folder / "stress_test_summary.csv")
    rows = []
    for scenario, frame in events.groupby("scenario", sort=False):
        recovered = frame[frame.recovered.astype(bool)]
        q = lambda values, p: np.percentile(values, p)
        rows.append(
            {
                "scenario": scenario,
                "events": len(frame),
                "non_recovery_fraction": 1 - frame.recovered.astype(bool).mean(),
                "recovery_steps_median": np.median(recovered.recovery_steps),
                "recovery_steps_q1": q(recovered.recovery_steps, 25),
                "recovery_steps_q3": q(recovered.recovery_steps, 75),
                "residual_position_median_m": np.median(recovered.residual_position_median_m),
                "residual_position_q1_m": q(recovered.residual_position_median_m, 25),
                "residual_position_q3_m": q(recovered.residual_position_median_m, 75),
                "residual_heading_median_deg": np.median(recovered.residual_heading_median_deg),
                "residual_heading_q1_deg": q(recovered.residual_heading_median_deg, 25),
                "residual_heading_q3_deg": q(recovered.residual_heading_median_deg, 75),
            }
        )
    recomputed = pd.DataFrame(rows)
    metrics = [c for c in saved.columns if c != "scenario"]
    check_rows(recomputed, saved, ["scenario"], metrics, "WP2")
    return {"events": len(events), "scenarios": len(saved)}


def verify_wp3(root):
    folder = root / "WP3_fingerprint_models"
    trace = pd.read_csv(folder / "validation_error_traces.csv")
    saved = pd.read_csv(folder / "fingerprint_model_comparison.csv")
    rows = []
    for method in [c for c in trace.columns if c != "sample_index"]:
        values = trace[method].to_numpy()
        rows.append(
            {
                "method": method,
                "samples": len(values),
                "position_median_m": np.median(values),
                "position_rmse_m": np.sqrt(np.mean(values**2)),
                "position_p90_m": np.percentile(values, 90),
                "position_max_m": np.max(values),
            }
        )
    recomputed = pd.DataFrame(rows)
    check_rows(
        recomputed,
        saved,
        ["method"],
        ["samples", "position_median_m", "position_rmse_m", "position_p90_m", "position_max_m"],
        "WP3",
    )
    return {"validation_samples": len(trace), "models": len(rows)}


def verify_wp5(root):
    folder = root / "WP5_initialization"
    trace = pd.read_csv(folder / "initialization_error_traces.csv")
    saved = pd.read_csv(folder / "initialization_metrics.csv")
    segments = {"all": slice(None), "first_20": slice(0, 20), "first_100": slice(0, 100), "last_100": slice(-100, None)}
    rows = []
    for (initialization, estimator), frame in trace.groupby(["initialization", "estimator"], sort=False):
        frame = frame.sort_values("evaluation_step").reset_index(drop=True)
        for segment, selector in segments.items():
            block = frame.iloc[selector]
            rows.append(
                {
                    "initialization": initialization,
                    "estimator": estimator,
                    "segment": segment,
                    "samples": len(block),
                    "position_median_m": np.median(block.position_error_m),
                    "position_p90_m": np.percentile(block.position_error_m, 90),
                    "heading_median_deg": np.median(block.absolute_heading_error_deg),
                    "heading_p90_deg": np.percentile(block.absolute_heading_error_deg, 90),
                }
            )
    recomputed = pd.DataFrame(rows)
    check_rows(
        recomputed,
        saved,
        ["initialization", "estimator", "segment"],
        ["samples", "position_median_m", "position_p90_m", "heading_median_deg", "heading_p90_deg"],
        "WP5",
    )
    return {"conditions": trace.initialization.nunique(), "trace_rows": len(trace)}


def verify_wp6(root):
    folder = root / "WP6_crossover"
    trace = pd.read_csv(folder / "rolling_error_traces.csv")
    saved = pd.read_csv(folder / "crossover_results.csv")
    for _, expected in saved.iterrows():
        frame = trace[trace.path == expected.path].sort_values("trace_index_0_based")
        valid = frame.lstm_rolling_median_m.notna() & frame.encoder_rolling_median_m.notna()
        better = (frame.lstm_rolling_median_m < frame.encoder_rolling_median_m).to_numpy()
        candidates = [i for i in np.flatnonzero(valid.to_numpy() & better) if better[i:].all()]
        index = int(candidates[0])
        close(index, expected.crossover_trace_index_0_based, f"WP6 {expected.path} trace index")
        close(index + 1, expected.crossover_evaluation_step_1_based, f"WP6 {expected.path} step")
    return {"paths": len(saved), "crossovers": saved.crossover_evaluation_step_1_based.astype(int).tolist()}


def verify_heading(root):
    folder = root / "heading_uncertainty"
    trace = pd.read_csv(folder / "empirical_heading_errors.csv")
    saved = pd.read_csv(folder / "heading_uncertainty_summary.csv").iloc[0]
    radians = trace.heading_error_rad.to_numpy()
    resultant = np.abs(np.mean(np.exp(1j * radians)))
    circular_sd = np.degrees(np.sqrt(-2 * np.log(resultant)))
    close(np.median(trace.absolute_heading_error_deg), saved.empirical_median_abs_deg, "heading median")
    close(np.percentile(trace.absolute_heading_error_deg, 90), saved.empirical_p90_abs_deg, "heading p90")
    close(circular_sd, saved.empirical_circular_sd_deg, "heading circular SD")
    return {"samples": len(trace), "circular_sd_deg": circular_sd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = {
        "WP1": verify_wp1(args.output_root),
        "WP2": verify_wp2(args.output_root),
        "WP3": verify_wp3(args.output_root),
        "WP5": verify_wp5(args.output_root),
        "WP6": verify_wp6(args.output_root),
        "heading_uncertainty": verify_heading(args.output_root),
    }
    result["status"] = "all recomputed values match saved summaries"
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.json:
        args.json.write_text(text)


if __name__ == "__main__":
    main()
