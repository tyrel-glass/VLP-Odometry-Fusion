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
    for (scenario, method), frame in events.groupby(["scenario", "method"], sort=False):
        recovered = frame[frame.recovered.astype(bool)]
        q = lambda values, p: np.percentile(values, p)
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "events": len(frame),
                "source_runs": frame.run.nunique(),
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
    metrics = [c for c in saved.columns if c not in {"scenario", "method"}]
    check_rows(recomputed, saved, ["scenario", "method"], metrics, "WP2")
    return {
        "event_method_rows": len(events),
        "unique_events": events.event_id.nunique(),
        "summary_rows": len(saved),
    }


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
    trace = pd.read_csv(folder / "segment_error_traces.csv")
    saved = pd.read_csv(folder / "segment_metrics.csv")
    rows = []
    for (segment_id, initialization), frame in trace.groupby(
        ["segment_id", "initialization"], sort=False
    ):
        frame = frame.sort_values("evaluation_step").reset_index(drop=True)
        rows.append(
            {
                "segment_id": segment_id,
                "initialization": initialization,
                "samples": len(frame),
                "position_median_m": np.median(frame.position_error_m),
                "position_p90_m": np.percentile(frame.position_error_m, 90),
                "heading_median_deg": np.median(frame.absolute_heading_error_deg),
                "heading_p90_deg": np.percentile(frame.absolute_heading_error_deg, 90),
            }
        )
    recomputed = pd.DataFrame(rows)
    check_rows(
        recomputed,
        saved,
        ["segment_id", "initialization"],
        ["samples", "position_median_m", "position_p90_m", "heading_median_deg", "heading_p90_deg"],
        "WP5 segment metrics",
    )

    metric_columns = [
        "position_median_m",
        "position_p90_m",
        "heading_median_deg",
        "heading_p90_deg",
    ]
    summary_saved = pd.read_csv(folder / "initialization_summary.csv")
    summary_rows = []
    for initialization, frame in saved.groupby("initialization", sort=False):
        row = {
            "initialization": initialization,
            "segments": len(frame),
            "source_runs": frame.run.nunique(),
        }
        row.update(
            {
                f"median_segment_{column}": np.median(frame[column])
                for column in metric_columns
            }
        )
        summary_rows.append(row)
    summary_recomputed = pd.DataFrame(summary_rows)
    check_rows(
        summary_recomputed,
        summary_saved,
        ["initialization"],
        [c for c in summary_saved.columns if c != "initialization"],
        "WP5 initialization summary",
    )

    vive = saved[saved.initialization == "Vive known pose"]
    vlp = saved[saved.initialization == "Two-fix VLP"]
    paired = vive.merge(
        vlp,
        on=["segment_id", "run", "path", "controller"],
        suffixes=("_vive", "_vlp"),
        validate="one_to_one",
    )
    paired_saved = pd.read_csv(folder / "paired_metric_summary.csv")
    paired_rows = []
    unique_runs = np.unique(paired.run.to_numpy())
    for offset, column in enumerate(metric_columns):
        difference = paired[f"{column}_vlp"].to_numpy() - paired[f"{column}_vive"].to_numpy()
        rng = np.random.default_rng(20261006 + offset)
        estimates = np.empty(10000)
        for replicate in range(10000):
            selected_runs = rng.choice(unique_runs, size=len(unique_runs), replace=True)
            block = np.concatenate(
                [difference[paired.run.to_numpy() == run] for run in selected_runs]
            )
            estimates[replicate] = np.median(block)
        lower, upper = np.percentile(estimates, [2.5, 97.5])
        paired_rows.append(
            {
                "metric": column,
                "segments": len(paired),
                "source_runs": paired.run.nunique(),
                "vive_median_across_segments": np.median(paired[f"{column}_vive"]),
                "vlp_median_across_segments": np.median(paired[f"{column}_vlp"]),
                "median_paired_difference_vlp_minus_vive": np.median(difference),
                "cluster_bootstrap_95ci_low": lower,
                "cluster_bootstrap_95ci_high": upper,
            }
        )
    check_rows(
        pd.DataFrame(paired_rows),
        paired_saved,
        ["metric"],
        [c for c in paired_saved.columns if c != "metric"],
        "WP5 paired summary",
    )
    step_summary = pd.read_csv(folder / "initialization_step_summary.csv")
    step_rows = []
    for metric, column in (
        ("Position median", "position_error_m"),
        ("Position $P_{90}$", "position_error_m"),
        ("Heading median", "absolute_heading_error_deg"),
        ("Heading $P_{90}$", "absolute_heading_error_deg"),
    ):
        quantile = 0.90 if "$P_{90}$" in metric else 0.50
        for initialization, frame in trace.groupby("initialization", sort=False):
            row = {"metric": metric, "initialization": initialization}
            for step in (1, 15, 30, 45):
                values = frame.loc[frame.evaluation_step == step, column]
                row[f"step_{step}"] = values.quantile(quantile)
            step_rows.append(row)
    check_rows(
        pd.DataFrame(step_rows),
        step_summary.drop(columns="unit"),
        ["metric", "initialization"],
        ["step_1", "step_15", "step_30", "step_45"],
        "WP5 selected-step summary",
    )
    return {
        "conditions": trace.initialization.nunique(),
        "segments": trace.segment_id.nunique(),
        "source_runs": trace.run.nunique(),
        "evaluation_steps_per_condition_segment": int(
            trace.groupby(["segment_id", "initialization"]).size().iloc[0]
        ),
        "trace_rows": len(trace),
    }


def verify_wp6(root):
    folder = root / "WP6_crossover"
    trace = pd.read_csv(folder / "rolling_error_traces.csv")
    saved = pd.read_csv(folder / "crossover_results.csv")
    for _, expected in saved.iterrows():
        frame = trace[
            (trace.path == expected.path) & (trace.metric == expected.metric)
        ].sort_values("trace_index_0_based")
        valid = frame.lstm_rolling_median.notna() & frame.encoder_rolling_median.notna()
        better = (frame.lstm_rolling_median < frame.encoder_rolling_median).to_numpy()
        candidates = [i for i in np.flatnonzero(valid.to_numpy() & better) if better[i:].all()]
        index = int(candidates[0])
        close(index + 1, expected.crossover_evaluation_step_1_based, f"WP6 {expected.path} step")
    return {
        "path_metric_pairs": len(saved),
        "crossovers": saved.crossover_evaluation_step_1_based.astype(int).tolist(),
    }


def verify_heading(root):
    folder = root / "heading_uncertainty"
    pairs = pd.read_csv(folder / "cnc_pairwise_heading_errors.csv")
    saved = pd.read_csv(folder / "heading_error_by_baseline.csv")
    rows = []
    for baseline_bin, frame in pairs.groupby("baseline_bin_m", sort=False):
        rows.append(
            {
                "baseline_bin_m": baseline_bin,
                "pairs": len(frame),
                "median_true_baseline_m": np.median(frame.true_baseline_m),
                "heading_median_deg": np.median(frame.absolute_heading_error_deg),
                "heading_p90_deg": np.percentile(frame.absolute_heading_error_deg, 90),
            }
        )
    recomputed = pd.DataFrame(rows)
    check_rows(
        recomputed,
        saved,
        ["baseline_bin_m"],
        ["pairs", "median_true_baseline_m", "heading_median_deg", "heading_p90_deg"],
        "heading uncertainty",
    )
    return {"pairs": len(pairs), "baseline_bins": len(saved)}


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
