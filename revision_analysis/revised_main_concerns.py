"""Regenerate analyses revised while reviewing Reviewer 1's major concerns.

This module supersedes the original WP1/WP2/heading-uncertainty implementations.
It deliberately keeps the four revised protocols together:

* a known-pose LSTM--EKF comparison on manuscript dataset 10;
* paired LSTM--EKF synthetic stress replays from the nine nominal live runs; and
* VLP-derived heading error as a function of displacement on the CNC validation set;
* paired Vive- and VLP-initialized LSTM replays over 50 live-recording segments.

All generated artifacts are written below ``revision_round_1/analysis/outputs``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


RANDOM_SEED = 20261006
TRAIN_RUNS = tuple(range(8))
VALIDATION_RUN = 8
TEST_RUN = 9
RNN_WINDOW = 20
EVALUATION_SAMPLES = 300
POSITION_DISPLAY_FLOOR_M = 1e-3
HEADING_DISPLAY_FLOOR_DEG = 1e-1
TURN_THRESHOLD_RAD = np.deg2rad(1.0)
MIN_HEADING_BASELINE_M = 0.02
INITIALIZATION_EVALUATION_STEPS = 50
INITIALIZATION_RAW_SEGMENT_STEPS = RNN_WINDOW + INITIALIZATION_EVALUATION_STEPS
INITIALIZATION_SEGMENT_STRIDE = INITIALIZATION_RAW_SEGMENT_STEPS
INITIALIZATION_BOOTSTRAP_REPLICATES = 10000
RNN_FEATURES = [
    "vlp_x_hist",
    "vlp_y_hist",
    "vlp_heading_hist_rad",
    "vlp_heading_change_rad",
    "encoder_heading_change_rad",
    "encoder_heading_hist_rad",
    "encoder_x_hist",
    "encoder_y_hist",
]

LIVE_FILES = {
    "Square": {
        "LSTM-controlled": "final_rnn_square_m3_2.csv",
        "VLP-controlled": "final_vlp_square_m3_2.csv",
        "Odometry-controlled": "final_encoder_square_m3.csv",
    },
    "Hourglass": {
        "LSTM-controlled": "final_rnn_hourglass_m3.csv",
        "VLP-controlled": "final_vlp_hourglass_m3.csv",
        "Odometry-controlled": "final_encoder_hourglass_m3.csv",
    },
    "Random": {
        "LSTM-controlled": "final_rnn_random_m3_2.csv",
        "VLP-controlled": "final_vlp_random_m3.csv",
        "Odometry-controlled": "final_encoder_random_m3.csv",
    },
}


def wrap_angle(values):
    values = np.asarray(values, dtype=float)
    return (values + np.pi) % (2 * np.pi) - np.pi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def positive_definite_covariance(rows, minimum_eigenvalue=1e-12):
    rows = np.asarray(rows, dtype=float)
    rows = rows[np.isfinite(rows).all(axis=1)]
    covariance = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(minimum_eigenvalue, float(eigenvalues.max()) * 1e-10)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def summarize_errors(position_error, heading_error):
    position_error = np.asarray(position_error, dtype=float)
    heading_deg = np.degrees(np.abs(wrap_angle(heading_error)))
    return {
        "samples": int(len(position_error)),
        "position_median_m": float(np.median(position_error)),
        "position_p90_m": float(np.percentile(position_error, 90)),
        "heading_median_deg": float(np.median(heading_deg)),
        "heading_p90_deg": float(np.percentile(heading_deg, 90)),
    }


def make_rnn_windows(frame, window=RNN_WINDOW):
    features = np.nan_to_num(
        frame[RNN_FEATURES].to_numpy(dtype=np.float32), nan=0.0
    )
    return np.stack(
        [features[end - window : end] for end in range(window, len(features) + 1)]
    )


def predict_rnn(frame, model, window=RNN_WINDOW):
    position, heading_vector = model.predict(make_rnn_windows(frame, window), verbose=0)
    heading = np.arctan2(heading_vector[:, 0], heading_vector[:, 1])
    return pd.DataFrame(
        {"x": position[:, 0], "y": position[:, 1], "heading": heading},
        index=frame.index[window - 1 :],
    )


def original_vive_alignment(frame, alignment_index=RNN_WINDOW - 1):
    result = frame.copy()
    encoder_columns = ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist_rad"]
    truth_columns = ["x_hist", "y_hist", "heading_hist_rad"]
    patch = (
        result.loc[alignment_index, truth_columns].to_numpy(dtype=float)
        - result.loc[alignment_index, encoder_columns].to_numpy(dtype=float)
    )
    result[encoder_columns] = result[encoder_columns].to_numpy(dtype=float) + patch
    return result, patch


def heading_from_two_positions(first, second, minimum=MIN_HEADING_BASELINE_M):
    displacement = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    if not np.isfinite(displacement).all() or np.linalg.norm(displacement) < minimum:
        return np.nan
    return float(np.arctan2(displacement[0], displacement[1]))


def measurement_residuals(frame, include_heading=False):
    rows = []
    for index in range(1 if include_heading else 0, len(frame)):
        sample = frame.iloc[index]
        position = np.array(
            [sample.vlp_x_hist - sample.x_hist, sample.vlp_y_hist - sample.y_hist]
        )
        if include_heading:
            previous = frame.iloc[index - 1]
            heading = heading_from_two_positions(
                [previous.vlp_x_hist, previous.vlp_y_hist],
                [sample.vlp_x_hist, sample.vlp_y_hist],
            )
            row = np.r_[position, wrap_angle(heading - sample.heading_hist_rad)]
        else:
            row = position
        if np.isfinite(row).all():
            rows.append(row)
    return np.asarray(rows)


def estimate_covariances(calibration_frames):
    straight, turning = [], []
    for frame in calibration_frames:
        residual = np.column_stack(
            [
                frame.encoder_location_change_err.to_numpy(dtype=float),
                wrap_angle(frame.encoder_heading_change_err_rad.to_numpy(dtype=float)),
            ]
        )
        commanded = np.abs(frame.encoder_heading_change_rad.to_numpy(dtype=float))
        finite = np.isfinite(residual).all(axis=1) & np.isfinite(commanded)
        straight.append(residual[finite & (commanded < TURN_THRESHOLD_RAD)])
        turning.append(residual[finite & (commanded >= TURN_THRESHOLD_RAD)])
    process = {
        "straight": positive_definite_covariance(np.vstack(straight)),
        "turn": positive_definite_covariance(np.vstack(turning)),
    }
    measurement = {
        "position_only": positive_definite_covariance(
            np.vstack([measurement_residuals(frame) for frame in calibration_frames])
        ),
        "one_step": positive_definite_covariance(
            np.vstack(
                [measurement_residuals(frame, include_heading=True) for frame in calibration_frames]
            )
        ),
    }
    return process, measurement


@dataclass(frozen=True)
class EKFVariant:
    name: str
    include_heading: bool
    covariance_key: str


class PoseEKF:
    def __init__(self, initial_state, initial_covariance=None):
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.covariance = (
            np.zeros((3, 3), dtype=float)
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=float).copy()
        )

    def predict(self, distance, delta_theta, process_covariance):
        x, y, theta = self.state
        theta = float(wrap_angle(theta + delta_theta))
        sine, cosine = np.sin(theta), np.cos(theta)
        self.state = np.array([x + distance * sine, y + distance * cosine, theta])
        state_jacobian = np.array(
            [[1, 0, distance * cosine], [0, 1, -distance * sine], [0, 0, 1]],
            dtype=float,
        )
        control_jacobian = np.array(
            [[sine, distance * cosine], [cosine, -distance * sine], [0, 1]],
            dtype=float,
        )
        self.covariance = (
            state_jacobian @ self.covariance @ state_jacobian.T
            + control_jacobian @ process_covariance @ control_jacobian.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update(self, measurement, measurement_covariance):
        measurement = np.asarray(measurement, dtype=float)
        observation = np.zeros((len(measurement), 3), dtype=float)
        observation[0, 0], observation[1, 1] = 1, 1
        if len(measurement) == 3:
            observation[2, 2] = 1
        innovation = measurement - observation @ self.state
        if len(measurement) == 3:
            innovation[2] = wrap_angle(innovation[2])
        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance.T, (self.covariance @ observation.T).T
        ).T
        self.state += gain @ innovation
        self.state[2] = wrap_angle(self.state[2])
        identity = np.eye(3)
        residual = identity - gain @ observation
        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)


POSITION_EKF = EKFVariant("Position-only EKF", False, "position_only")
ONE_STEP_EKF = EKFVariant("EKF with one-step VLP heading", True, "one_step")


def ekf_measurement(frame, index, variant):
    current = frame.iloc[index]
    position = np.array([current.vlp_x_hist, current.vlp_y_hist], dtype=float)
    if not np.isfinite(position).all():
        return None
    if not variant.include_heading or index == 0:
        return position
    previous = frame.iloc[index - 1]
    heading = heading_from_two_positions(
        [previous.vlp_x_hist, previous.vlp_y_hist], position
    )
    return position if not np.isfinite(heading) else np.r_[position, heading]


def run_known_pose_ekf(
    frame,
    variant,
    process_covariances,
    measurement_covariances,
    initial_index,
    initial_state,
):
    model = PoseEKF(initial_state)
    states = np.full((len(frame), 3), np.nan)
    states[initial_index] = model.state
    for index in range(initial_index + 1, len(frame)):
        sample = frame.iloc[index]
        distance = float(sample.encoder_location_change)
        delta_theta = float(sample.encoder_heading_change_rad)
        distance = distance if np.isfinite(distance) else 0.0
        delta_theta = delta_theta if np.isfinite(delta_theta) else 0.0
        process_key = "turn" if abs(delta_theta) >= TURN_THRESHOLD_RAD else "straight"
        model.predict(distance, delta_theta, process_covariances[process_key])
        measurement = ekf_measurement(frame, index, variant)
        if measurement is not None:
            covariance = measurement_covariances[variant.covariance_key]
            if len(measurement) == 2 and covariance.shape == (3, 3):
                covariance = measurement_covariances["position_only"]
            model.update(measurement, covariance)
        states[index] = model.state
    return pd.DataFrame(states, columns=["x", "y", "heading"], index=frame.index)


def load_navigation_runs(robot_root):
    data_dir = Path(robot_root) / "data/interim/exp_vive_navigated_paths"
    return {
        run: pd.read_csv(data_dir / f"exp1_{run}_high_acc.csv").reset_index(drop=True)
        for run in range(10)
    }


def evaluate_prediction(frame, prediction, indices):
    truth = frame.loc[indices]
    estimate = prediction.loc[indices]
    position = np.linalg.norm(
        estimate[["x", "y"]].to_numpy() - truth[["x_hist", "y_hist"]].to_numpy(),
        axis=1,
    )
    heading = wrap_angle(
        estimate.heading.to_numpy() - truth.heading_hist_rad.to_numpy()
    )
    return position, heading


def run_wp1_known_pose(robot_root: Path, output_root: Path):
    output = Path(output_root) / "WP1_ekf_comparison"
    output.mkdir(parents=True, exist_ok=True)
    runs = load_navigation_runs(robot_root)
    process_covariances, measurement_covariances = estimate_covariances(
        [runs[index] for index in TRAIN_RUNS]
    )
    test = runs[TEST_RUN]
    anchor = RNN_WINDOW - 1
    known_state = test.loc[anchor, ["x_hist", "y_hist", "heading_hist_rad"]].to_numpy()
    variants = [POSITION_EKF, ONE_STEP_EKF]
    predictions = {
        variant.name: run_known_pose_ekf(
            test,
            variant,
            process_covariances,
            measurement_covariances,
            anchor,
            known_state,
        )
        for variant in variants
    }
    aligned, patch = original_vive_alignment(test, anchor)
    model_path = Path(robot_root) / "models/navigation_neural_nets/rnn.keras"
    model = load_model(model_path, compile=False)
    predictions["LSTM"] = predict_rnn(aligned, model)
    predictions["VLP"] = pd.DataFrame(
        {"x": test.vlp_x_hist, "y": test.vlp_y_hist, "heading": test.vlp_heading_hist_rad}
    )
    predictions["Encoder odometry"] = pd.DataFrame(
        {
            "x": aligned.encoder_x_hist,
            "y": aligned.encoder_y_hist,
            "heading": aligned.encoder_heading_hist_rad,
        }
    )
    indices = test.index[anchor : anchor + EVALUATION_SAMPLES].to_numpy()
    segments = {"all": indices, "first_100": indices[:100], "last_100": indices[-100:]}
    metric_rows, trace_rows = [], []
    for name, prediction in predictions.items():
        position, heading = evaluate_prediction(test, prediction, indices)
        for step, sample, pos, ang in zip(range(1, len(indices) + 1), indices, position, heading):
            estimate = prediction.loc[sample]
            trace_rows.append(
                {
                    "evaluation_step": step,
                    "sample_index": int(sample),
                    "model": name,
                    "x": float(estimate.x),
                    "y": float(estimate.y),
                    "heading_rad": float(estimate.heading),
                    "position_error_m": float(pos),
                    "heading_error_rad": float(ang),
                    "absolute_heading_error_deg": float(np.degrees(abs(ang))),
                }
            )
        for segment, segment_indices in segments.items():
            pos, ang = evaluate_prediction(test, prediction, segment_indices)
            metric_rows.append(
                {
                    "file_run": TEST_RUN,
                    "manuscript_dataset": 10,
                    "role": "offline evaluation",
                    "initialization": "known Vive pose at raw sample 19",
                    "segment": segment,
                    "model": name,
                    **summarize_errors(pos, ang),
                }
            )
    metrics, traces = pd.DataFrame(metric_rows), pd.DataFrame(trace_rows)
    metrics.to_csv(output / "held_out_metrics.csv", index=False)
    traces.to_csv(output / "held_out_error_traces.csv", index=False)

    comparison_names = ["LSTM", POSITION_EKF.name, ONE_STEP_EKF.name]
    compact = metrics[(metrics.segment == "all") & metrics.model.isin(comparison_names)][
        [
            "model",
            "samples",
            "position_median_m",
            "position_p90_m",
            "heading_median_deg",
            "heading_p90_deg",
        ]
    ].copy()
    compact.to_csv(output / "lstm_ekf_comparison.csv", index=False)
    latex = compact.rename(
        columns={
            "model": "Method",
            "samples": "$N$",
            "position_median_m": "Median position (m)",
            "position_p90_m": "$P_{90}$ position (m)",
            "heading_median_deg": "Median heading ($^\\circ$)",
            "heading_p90_deg": "$P_{90}$ heading ($^\\circ$)",
        }
    )
    (output / "lstm_ekf_comparison_table.tex").write_text(
        latex.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.4f}")
    )

    styles = {
        "LSTM": ("#0072B2", "-"),
        POSITION_EKF.name: ("#D55E00", "--"),
        ONE_STEP_EKF.name: ("#009E73", ":"),
    }
    display_names = {
        "LSTM": "LSTM",
        POSITION_EKF.name: "Position-only EKF",
        ONE_STEP_EKF.name: "One-step heading EKF",
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)
    for name in comparison_names:
        block = traces[traces.model == name]
        colour, linestyle = styles[name]
        axes[0].plot(block.evaluation_step, np.maximum(block.position_error_m, POSITION_DISPLAY_FLOOR_M), label=display_names[name], color=colour, ls=linestyle, lw=0.85)
        axes[1].plot(block.evaluation_step, np.maximum(block.absolute_heading_error_deg, HEADING_DISPLAY_FLOOR_DEG), label=display_names[name], color=colour, ls=linestyle, lw=0.85)
    axes[0].set_ylabel("Position error (m)", fontsize=8)
    axes[1].set_ylabel("Heading error (deg)", fontsize=8)
    axes[1].set_xlabel("Evaluation step", fontsize=8)
    axes[0].set_xlim(1, EVALUATION_SAMPLES)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylim(8e-4, 0.5)
    axes[1].set_ylim(8e-2, 180)
    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.grid(True, which="both", linestyle=":", alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output / "offline_lstm_ekf_error_traces.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    configuration = {
        "protocol": "known-pose offline LSTM--EKF comparison",
        "calibration_datasets": list(range(1, 9)),
        "evaluation_dataset": 10,
        "evaluation_dataset_role": "offline evaluation",
        "validation_dataset_not_used_by_ekf": 9,
        "initialization": "Vive ground-truth pose at raw sample 19; zero initial covariance",
        "evaluation_indices": [int(indices[0]), int(indices[-1])],
        "process_covariances": {key: value.tolist() for key, value in process_covariances.items()},
        "measurement_covariances": {key: value.tolist() for key, value in measurement_covariances.items()},
        "lstm_encoder_alignment_patch": patch.tolist(),
        "model_sha256": sha256_file(model_path),
    }
    write_json(output / "frozen_configuration.json", configuration)
    write_json(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "model_sha256": sha256_file(model_path),
            "dataset_sha256": sha256_file(
                Path(robot_root) / "data/interim/exp_vive_navigated_paths/exp1_9_high_acc.csv"
            ),
        },
    )
    (output / "methods_and_scope.md").write_text(
        "# Known-pose LSTM--EKF comparison\n\n"
        "The saved LSTM, position-only EKF, and one-step-heading EKF use the same "
        "known Vive pose at raw sample 19 and the same 300 evaluation samples "
        "from dataset 10. The "
        "LSTM weights are unchanged. EKF process and measurement covariances use "
        "datasets 1--8 only; dataset 9 is not used to select an EKF design. "
        "The one-step heading is a sensitivity case derived from the same VLP positions, "
        "with their full empirical position--heading covariance. These are offline "
        "estimates and no closed-loop EKF trial was recorded.\n"
    )
    return compact, traces, process_covariances, measurement_covariances


def load_cnc_validation(robot_root: Path):
    if str(robot_root) not in sys.path:
        sys.path.insert(0, str(robot_root))
    from robot_vlp.data_collection.communication import process_cnc, process_vlp
    import pickle

    raw = Path(robot_root) / "data/raw/experiments/CNC"
    validation = pd.read_csv(raw / "cnc_fingerprint_02.csv", delimiter="|")
    validation = process_vlp(process_cnc(validation))
    model_dir = Path(robot_root) / "models/vlp/CNC"
    metadata_path = model_dir / "CNC_vlp_models.pkl"
    metadata = pickle.loads(metadata_path.read_bytes())
    scaler = metadata["high_acc"]["scaler"]
    model_path = model_dir / metadata["high_acc"]["model_path"]
    model = load_model(model_path, compile=False)
    rss_columns = ["L1", "L2", "L3", "L4"]
    scaled = scaler.transform(pd.DataFrame(validation[rss_columns], columns=rss_columns))
    predicted = np.asarray(model(scaled, training=False), dtype=float) / 1000
    truth = validation[["cnc_x", "cnc_y"]].to_numpy(dtype=float) / 1000
    return validation, truth, predicted, metadata_path, model_path


def run_heading_baseline_analysis(robot_root: Path, output_root: Path):
    output = Path(output_root) / "heading_uncertainty"
    output.mkdir(parents=True, exist_ok=True)
    validation, truth, predicted, metadata_path, model_path = load_cnc_validation(robot_root)
    first, second = np.triu_indices(len(truth), k=1)
    true_displacement = truth[second] - truth[first]
    predicted_displacement = predicted[second] - predicted[first]
    baseline = np.linalg.norm(true_displacement, axis=1)
    true_heading = np.arctan2(true_displacement[:, 0], true_displacement[:, 1])
    predicted_heading = np.arctan2(predicted_displacement[:, 0], predicted_displacement[:, 1])
    error_deg = np.degrees(np.abs(wrap_angle(predicted_heading - true_heading)))
    bins = np.linspace(0.0, 0.50, 11)
    labels = [f"{lower:.2f}--{upper:.2f}" for lower, upper in zip(bins[:-1], bins[1:])]
    category = pd.cut(baseline, bins=bins, labels=labels, right=False)
    pairs = pd.DataFrame(
        {
            "first_validation_index": first,
            "second_validation_index": second,
            "true_baseline_m": baseline,
            "absolute_heading_error_deg": error_deg,
            "baseline_bin_m": category,
        }
    )
    rows = []
    for label in labels:
        block = pairs[pairs.baseline_bin_m == label]
        rows.append(
            {
                "baseline_bin_m": label,
                "pairs": len(block),
                "median_true_baseline_m": float(block.true_baseline_m.median()),
                "heading_median_deg": float(block.absolute_heading_error_deg.median()),
                "heading_p90_deg": float(block.absolute_heading_error_deg.quantile(0.90)),
            }
        )
    summary = pd.DataFrame(rows)
    pairs.to_csv(output / "cnc_pairwise_heading_errors.csv", index=False)
    summary.to_csv(output / "heading_error_by_baseline.csv", index=False)
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    x = summary.median_true_baseline_m
    series = [
        (summary.heading_median_deg, "Median", "#0072B2"),
        (summary.heading_p90_deg, "$P_{90}$", "#D55E00"),
    ]
    for values, label, colour in series:
        axis.plot(x, values, marker="o", ms=3.5, lw=0.9, color=colour, label=label)
    axis.axvline(0.11, color="0.35", ls=":", lw=0.8)
    axis.set_xlabel("True displacement baseline (m)", fontsize=8)
    axis.set_ylabel("Absolute heading error (deg)", fontsize=8)
    axis.set_ylim(bottom=0)
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", color="0.9", lw=0.5)
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    axis.text(
        0.116,
        0.94,
        "0.11 m step",
        transform=axis.get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="left",
        fontsize=6.5,
        color="0.30",
    )
    fig.tight_layout()
    fig.savefig(output / "heading_error_by_baseline.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    write_json(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "validation_samples": len(validation),
            "pair_count": len(pairs),
            "displayed_pair_count": int(category.notna().sum()),
            "displayed_maximum_baseline_m": 0.50,
            "displayed_bin_count": len(labels),
            "validation_sha256": sha256_file(Path(robot_root) / "data/raw/experiments/CNC/cnc_fingerprint_02.csv"),
            "model_metadata_sha256": sha256_file(metadata_path),
            "model_sha256": sha256_file(model_path),
        },
    )
    (output / "methods_and_scope.md").write_text(
        "# VLP heading error versus displacement baseline\n\n"
        "All unordered pairs of the 528 independent CNC validation locations are "
        "formed and retained in the pairwise CSV. True bearing is calculated from "
        "CNC coordinates and estimated bearing from the corresponding VLP predictions. "
        "The plot summarizes absolute circular heading error in ten equal-width "
        "0.05 m bins over baselines from 0 to 0.50 m; longer-baseline pairs are "
        "retained but omitted from the plotted summary. Pairs reuse validation points and "
        "are therefore descriptive geometric comparisons, not independent robot "
        "trajectories. The analysis isolates the VLP displacement effect from "
        "odometry and temporal dynamics.\n"
    )
    return summary


def live_frame(path: Path):
    raw = pd.read_csv(path, index_col=0).reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "x_hist": raw.real_x,
            "y_hist": raw.real_y,
            "heading_hist_rad": np.deg2rad(raw.real_heading),
            "vlp_x_hist": raw.vlp_x,
            "vlp_y_hist": raw.vlp_y,
            "vlp_heading_hist_rad": np.deg2rad(raw.vlp_heading),
            "vlp_heading_change_rad": np.deg2rad(raw.vlp_heading_change),
            "encoder_heading_change_rad": np.deg2rad(raw.encoder_heading_change),
            "encoder_heading_hist_rad": np.deg2rad(raw.encoder_heading),
            "encoder_x_hist": raw.encoder_x,
            "encoder_y_hist": raw.encoder_y,
        }
    )
    frame["encoder_location_change"] = np.r_[
        0.0,
        np.hypot(np.diff(frame.encoder_x_hist), np.diff(frame.encoder_y_hist)),
    ]
    return frame


def reintegrate_encoder_pose(frame, initial_state):
    """Rebuild absolute encoder-pose features from one supplied segment pose."""
    result = frame.copy()
    state = np.asarray(initial_state, dtype=float).copy()
    states = np.full((len(result), 3), np.nan)
    states[0] = state
    for index in range(1, len(result)):
        distance = float(result.iloc[index].encoder_location_change)
        delta_theta = float(result.iloc[index].encoder_heading_change_rad)
        distance = distance if np.isfinite(distance) else 0.0
        delta_theta = delta_theta if np.isfinite(delta_theta) else 0.0
        state[2] = wrap_angle(state[2] + delta_theta)
        state[0] += distance * np.sin(state[2])
        state[1] += distance * np.cos(state[2])
        states[index] = state
    result[["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist_rad"]] = states
    return result


def clustered_bootstrap_median(values, clusters, replicates, seed):
    """Bootstrap a median by resampling complete source-run clusters."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        block = np.concatenate([values[clusters == cluster] for cluster in sampled])
        estimates[replicate] = np.median(block)
    return np.percentile(estimates, [2.5, 97.5])


def run_live_initialization_sensitivity(robot_root: Path, output_root: Path):
    """Compare known-pose and two-fix VLP initialization on paired live segments."""
    output = Path(output_root) / "WP5_initialization"
    output.mkdir(parents=True, exist_ok=True)
    live_dir = Path(robot_root) / "notebooks/exploratory"
    model_path = Path(robot_root) / "models/navigation_neural_nets/rnn.keras"
    model = load_model(model_path, compile=False)
    segment_rows, metric_rows, trace_rows, source_records = [], [], [], []
    segment_number = 0

    for path_name, controllers in LIVE_FILES.items():
        for controller, filename in controllers.items():
            run_name = f"{path_name} / {controller}"
            source_path = live_dir / filename
            frame = live_frame(source_path)
            source_records.append(
                {"run": run_name, "file": str(source_path), "sha256": sha256_file(source_path)}
            )
            maximum_start = len(frame) - INITIALIZATION_RAW_SEGMENT_STEPS
            for start in range(0, maximum_start + 1, INITIALIZATION_SEGMENT_STRIDE):
                raw_segment = frame.iloc[
                    start : start + INITIALIZATION_RAW_SEGMENT_STEPS
                ].reset_index(drop=True)
                first_vlp = raw_segment.loc[0, ["vlp_x_hist", "vlp_y_hist"]].to_numpy(dtype=float)
                second_vlp = raw_segment.loc[1, ["vlp_x_hist", "vlp_y_hist"]].to_numpy(dtype=float)
                vlp_heading = heading_from_two_positions(first_vlp, second_vlp)
                if not np.isfinite(vlp_heading):
                    continue

                # Both conditions begin at the physical instant of the second VLP
                # fix.  The first fix is used only to form the deployable heading.
                segment = raw_segment.iloc[1:].reset_index(drop=True)
                vive_state = segment.loc[
                    0, ["x_hist", "y_hist", "heading_hist_rad"]
                ].to_numpy(dtype=float)
                vlp_state = np.array([second_vlp[0], second_vlp[1], vlp_heading])
                initializations = {
                    "Vive known pose": vive_state,
                    "Two-fix VLP": vlp_state,
                }
                segment_number += 1
                segment_id = f"segment_{segment_number:03d}"
                initial_position_error = float(np.linalg.norm(vlp_state[:2] - vive_state[:2]))
                initial_heading_error = float(
                    np.degrees(abs(wrap_angle(vlp_state[2] - vive_state[2])))
                )
                segment_rows.append(
                    {
                        "segment_id": segment_id,
                        "run": run_name,
                        "path": path_name,
                        "controller": controller,
                        "raw_start_index": start,
                        "initialization_index": start + 1,
                        "evaluation_start_index": start + RNN_WINDOW,
                        "evaluation_end_index": start + INITIALIZATION_RAW_SEGMENT_STEPS - 1,
                        "vlp_initial_baseline_m": float(np.linalg.norm(second_vlp - first_vlp)),
                        "vlp_initial_position_error_m": initial_position_error,
                        "vlp_initial_heading_error_deg": initial_heading_error,
                    }
                )

                for initialization, initial_state in initializations.items():
                    initialized = reintegrate_encoder_pose(segment, initial_state)
                    prediction = predict_rnn(initialized, model)
                    indices = prediction.index.to_numpy()
                    if len(indices) != INITIALIZATION_EVALUATION_STEPS:
                        raise AssertionError(
                            f"{segment_id}: expected {INITIALIZATION_EVALUATION_STEPS} "
                            f"predictions, received {len(indices)}"
                        )
                    position, heading = evaluate_prediction(initialized, prediction, indices)
                    metric_rows.append(
                        {
                            "segment_id": segment_id,
                            "run": run_name,
                            "path": path_name,
                            "controller": controller,
                            "initialization": initialization,
                            **summarize_errors(position, heading),
                        }
                    )
                    for step, sample, pos, angle in zip(
                        range(1, len(indices) + 1), indices, position, heading
                    ):
                        trace_rows.append(
                            {
                                "segment_id": segment_id,
                                "run": run_name,
                                "path": path_name,
                                "controller": controller,
                                "initialization": initialization,
                                "evaluation_step": step,
                                "segment_sample_index": int(sample),
                                "source_sample_index": int(start + 1 + sample),
                                "position_error_m": float(pos),
                                "heading_error_rad": float(angle),
                                "absolute_heading_error_deg": float(
                                    np.degrees(abs(angle))
                                ),
                            }
                        )

    segments = pd.DataFrame(segment_rows)
    metrics = pd.DataFrame(metric_rows)
    traces = pd.DataFrame(trace_rows)
    if len(segments) != 50:
        raise AssertionError(f"Expected 50 non-overlapping live segments, found {len(segments)}")
    if segments.run.nunique() != 9:
        raise AssertionError("Initialization analysis must include all nine nominal live runs")
    segments.to_csv(output / "segment_definitions.csv", index=False)
    metrics.to_csv(output / "segment_metrics.csv", index=False)
    traces.to_csv(output / "segment_error_traces.csv", index=False)

    metric_columns = [
        "position_median_m",
        "position_p90_m",
        "heading_median_deg",
        "heading_p90_deg",
    ]
    summary_rows = []
    for initialization, group in metrics.groupby("initialization", sort=False):
        row = {
            "initialization": initialization,
            "segments": len(group),
            "source_runs": group.run.nunique(),
        }
        row.update({f"median_segment_{column}": float(group[column].median()) for column in metric_columns})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "initialization_summary.csv", index=False)

    vive = metrics[metrics.initialization == "Vive known pose"]
    vlp = metrics[metrics.initialization == "Two-fix VLP"]
    paired = vive.merge(
        vlp,
        on=["segment_id", "run", "path", "controller"],
        suffixes=("_vive", "_vlp"),
        validate="one_to_one",
    )
    paired_rows = []
    for offset, column in enumerate(metric_columns):
        difference = paired[f"{column}_vlp"] - paired[f"{column}_vive"]
        lower, upper = clustered_bootstrap_median(
            difference,
            paired.run,
            INITIALIZATION_BOOTSTRAP_REPLICATES,
            RANDOM_SEED + offset,
        )
        paired_rows.append(
            {
                "metric": column,
                "segments": len(paired),
                "source_runs": paired.run.nunique(),
                "vive_median_across_segments": float(paired[f"{column}_vive"].median()),
                "vlp_median_across_segments": float(paired[f"{column}_vlp"].median()),
                "median_paired_difference_vlp_minus_vive": float(np.median(difference)),
                "cluster_bootstrap_95ci_low": float(lower),
                "cluster_bootstrap_95ci_high": float(upper),
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(output / "paired_metric_summary.csv", index=False)

    colours = {"Vive known pose": "#0072B2", "Two-fix VLP": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    for initialization, colour in colours.items():
        block = traces[traces.initialization == initialization]
        by_step = block.groupby("evaluation_step")
        position_median = by_step.position_error_m.median()
        position_p90 = by_step.position_error_m.quantile(0.90)
        heading_median = by_step.absolute_heading_error_deg.median()
        heading_p90 = by_step.absolute_heading_error_deg.quantile(0.90)
        steps = position_median.index.to_numpy()
        axes[0].plot(steps, position_median, color=colour, lw=1.0, label=f"{initialization}: median")
        axes[0].plot(steps, position_p90, color=colour, lw=0.8, ls=":", label=f"{initialization}: $P_{{90}}$")
        axes[1].plot(steps, heading_median, color=colour, lw=1.0)
        axes[1].plot(steps, heading_p90, color=colour, lw=0.8, ls=":")
    axes[0].set_ylabel("Position error (m)", fontsize=8)
    axes[1].set_ylabel("Heading error (deg)", fontsize=8)
    axes[1].set_xlabel("Prediction index", fontsize=8)
    axes[0].set_xlim(1, INITIALIZATION_EVALUATION_STEPS)
    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.grid(axis="y", color="0.9", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=6.2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output / "initialization_over_live_segments.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_json(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "protocol": "paired non-overlapping nominal-live-recording segments",
            "source_runs": len(source_records),
            "segments": len(segments),
            "raw_segment_steps": INITIALIZATION_RAW_SEGMENT_STEPS,
            "segment_stride_steps": INITIALIZATION_SEGMENT_STRIDE,
            "initialization_fix_index_within_raw_segment": 1,
            "rnn_window_steps": RNN_WINDOW,
            "evaluation_steps": INITIALIZATION_EVALUATION_STEPS,
            "minimum_vlp_heading_baseline_m": MIN_HEADING_BASELINE_M,
            "bootstrap_replicates": INITIALIZATION_BOOTSTRAP_REPLICATES,
            "model_sha256": sha256_file(model_path),
            "sources": source_records,
        },
    )
    (output / "methods_and_scope.md").write_text(
        "# Repeated live-segment initialization sensitivity\n\n"
        "The nine nominal live recordings are divided into 50 non-overlapping "
        "70-sample blocks. For each block, the first VLP fix and the following "
        "fix initialize a two-fix VLP heading. Both conditions begin at the second "
        "fix: the known-pose condition uses Vive position and heading at that same "
        "instant, whereas the deployable condition uses the second VLP position "
        "and heading derived from the two fixes. Encoder absolute-pose features are "
        "then re-integrated from each initial state using identical recorded "
        "increments. Fifty paired predictions are evaluated, starting at the end of "
        "the first complete 20-sample LSTM window. Prediction index 1 denotes the "
        "estimate at the end of that first window. The saved LSTM, normalization, "
        "feature order, and weights "
        "are unchanged. Segments do not overlap, but segments from the same physical "
        "recording remain dependent; paired-difference confidence intervals therefore "
        "resample complete source runs. This is an offline estimator replay and does "
        "not reproduce controller reactions to alternative initialization.\n"
    )
    return summary, paired_summary


@dataclass(frozen=True)
class StressProtocol:
    minimum_gap_steps: int = 21
    post_splice_steps: int = 70
    evaluation_horizon_steps: int = 50
    residual_window_steps: int = 20
    recovery_consecutive_steps: int = 5
    recovery_position_m: float = 0.05
    recovery_heading_deg: float = 10.0
    exclusion_steps: int = 10
    events_per_type: int = 50
    maximum_events_per_run_type: int = 8
    translation_min_m: float = 0.10
    translation_max_m: float = 0.20
    small_translation_max_m: float = 0.05
    small_rotation_max_deg: float = 15.0
    rotation_min_deg: float = 60.0
    rotation_max_deg: float = 120.0


def stress_candidates(frame, run_name, path_name, controller, protocol):
    candidates = []
    last_source = len(frame) - protocol.post_splice_steps
    for pre in range(19, len(frame) - 1):
        for source in range(pre + protocol.minimum_gap_steps, last_source):
            distance = float(
                np.hypot(
                    frame.loc[source, "x_hist"] - frame.loc[pre, "x_hist"],
                    frame.loc[source, "y_hist"] - frame.loc[pre, "y_hist"],
                )
            )
            rotation = float(
                abs(
                    np.degrees(
                        wrap_angle(
                            frame.loc[source, "heading_hist_rad"]
                            - frame.loc[pre, "heading_hist_rad"]
                        )
                    )
                )
            )
            scenario = None
            if protocol.translation_min_m <= distance <= protocol.translation_max_m and rotation < protocol.small_rotation_max_deg:
                scenario = "Translation only"
                score = abs(distance - 0.15) / 0.05 + rotation / 15
            elif distance < protocol.small_translation_max_m and protocol.rotation_min_deg <= rotation <= protocol.rotation_max_deg:
                scenario = "Rotation only"
                score = distance / 0.05 + abs(rotation - 90) / 30
            elif protocol.translation_min_m <= distance <= protocol.translation_max_m and protocol.rotation_min_deg <= rotation <= protocol.rotation_max_deg:
                scenario = "Combined"
                score = abs(distance - 0.15) / 0.05 + abs(rotation - 90) / 30
            if scenario:
                candidates.append(
                    {
                        "run": run_name,
                        "path": path_name,
                        "controller": controller,
                        "scenario": scenario,
                        "pre_splice_index": pre,
                        "source_index": source,
                        "translation_m": distance,
                        "rotation_deg": rotation,
                        "selection_score": score,
                    }
                )
    return candidates


def select_stress_events(candidates, protocol):
    candidate_frame = pd.DataFrame(candidates)
    selected = []
    for scenario in ["Translation only", "Rotation only", "Combined"]:
        per_run = {}
        for run, group in candidate_frame[candidate_frame.scenario == scenario].groupby("run"):
            accepted = []
            for row in group.sort_values(["selection_score", "pre_splice_index", "source_index"]).itertuples(index=False):
                if all(
                    abs(row.pre_splice_index - previous.pre_splice_index) >= protocol.exclusion_steps
                    and abs(row.source_index - previous.source_index) >= protocol.exclusion_steps
                    for previous in accepted
                ):
                    accepted.append(row)
                if len(accepted) == protocol.maximum_events_per_run_type:
                    break
            per_run[run] = accepted
        scenario_selected = []
        while len(scenario_selected) < protocol.events_per_type:
            added = False
            for run in sorted(per_run):
                if per_run[run] and len(scenario_selected) < protocol.events_per_type:
                    scenario_selected.append(per_run[run].pop(0))
                    added = True
            if not added:
                break
        if len(scenario_selected) != protocol.events_per_type:
            raise RuntimeError(
                f"Only {len(scenario_selected)} separated {scenario} events are available"
            )
        selected.extend(row._asdict() for row in scenario_selected)
    result = pd.DataFrame(selected)
    result.insert(0, "event_id", np.arange(1, len(result) + 1))
    return candidate_frame, result


def derive_vlp_angles(frame):
    heading = np.arctan2(frame.vlp_x_hist.diff(), frame.vlp_y_hist.diff())
    frame["vlp_heading_hist_rad"] = heading
    frame["vlp_heading_change_rad"] = np.r_[np.nan, wrap_angle(np.diff(heading))]
    return frame


def build_spliced_live_frame(frame, pre, source, protocol):
    prefix = frame.iloc[: pre + 1].copy()
    suffix = frame.iloc[source : source + protocol.post_splice_steps].copy()
    event = pd.concat([prefix, suffix], ignore_index=True)
    first_post = len(prefix)
    state = frame.loc[
        pre, ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist_rad"]
    ].to_numpy(dtype=float)
    for index in range(first_post, len(event)):
        distance = float(event.loc[index, "encoder_location_change"])
        delta = float(event.loc[index, "encoder_heading_change_rad"])
        state[2] = wrap_angle(state[2] + delta)
        state[0] += distance * np.sin(state[2])
        state[1] += distance * np.cos(state[2])
        event.loc[index, ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist_rad"]] = state
    return derive_vlp_angles(event), first_post


def recovery_result(position, heading_deg, protocol):
    success = (position < protocol.recovery_position_m) & (
        heading_deg < protocol.recovery_heading_deg
    )
    recovery = None
    for offset in range(protocol.evaluation_horizon_steps):
        if offset + protocol.recovery_consecutive_steps <= len(success) and success[
            offset : offset + protocol.recovery_consecutive_steps
        ].all():
            recovery = offset
            break
    if recovery is None:
        return False, np.nan, np.nan, np.nan
    residual = slice(recovery, min(recovery + protocol.residual_window_steps, len(position)))
    return True, recovery, float(np.median(position[residual])), float(np.median(heading_deg[residual]))


def run_live_stress_test(
    robot_root: Path,
    output_root: Path,
    process_covariances=None,
    measurement_covariances=None,
):
    output = Path(output_root) / "WP2_stress_test"
    output.mkdir(parents=True, exist_ok=True)
    protocol = StressProtocol()
    runs = load_navigation_runs(robot_root)
    if process_covariances is None or measurement_covariances is None:
        process_covariances, measurement_covariances = estimate_covariances(
            [runs[index] for index in TRAIN_RUNS]
        )
    live_dir = Path(robot_root) / "notebooks/exploratory"
    live_frames, source_records, candidates = {}, [], []
    for path_name, controllers in LIVE_FILES.items():
        for controller, filename in controllers.items():
            run_name = f"{path_name} / {controller}"
            source_path = live_dir / filename
            frame = live_frame(source_path)
            live_frames[run_name] = frame
            source_records.append(
                {"run": run_name, "file": str(source_path), "sha256": sha256_file(source_path)}
            )
            candidates.extend(stress_candidates(frame, run_name, path_name, controller, protocol))
    candidate_frame, selected = select_stress_events(candidates, protocol)
    candidate_frame.groupby(["scenario", "run"]).size().rename("candidates").reset_index().to_csv(
        output / "candidate_counts.csv", index=False
    )
    selected.to_csv(output / "selected_splices.csv", index=False)
    model_path = Path(robot_root) / "models/navigation_neural_nets/rnn.keras"
    model = load_model(model_path, compile=False)
    event_rows, trace_rows = [], []
    for event in selected.itertuples(index=False):
        spliced, first_post = build_spliced_live_frame(
            live_frames[event.run], int(event.pre_splice_index), int(event.source_index), protocol
        )
        estimates = {
            "LSTM": predict_rnn(spliced, model),
            "Position-only EKF": run_known_pose_ekf(
                spliced,
                POSITION_EKF,
                process_covariances,
                measurement_covariances,
                0,
                np.array([0.0, 0.0, 0.0]),
            ),
        }
        post_indices = np.arange(first_post, first_post + protocol.evaluation_horizon_steps)
        truth = spliced.loc[post_indices]
        for method, estimate in estimates.items():
            predicted = estimate.loc[post_indices]
            position = np.linalg.norm(
                predicted[["x", "y"]].to_numpy() - truth[["x_hist", "y_hist"]].to_numpy(), axis=1
            )
            heading = np.degrees(
                np.abs(wrap_angle(predicted.heading.to_numpy() - truth.heading_hist_rad.to_numpy()))
            )
            recovered, recovery, residual_position, residual_heading = recovery_result(
                position, heading, protocol
            )
            event_rows.append(
                {
                    **event._asdict(),
                    "method": method,
                    "recovered": recovered,
                    "recovery_steps": recovery,
                    "residual_position_median_m": residual_position,
                    "residual_heading_median_deg": residual_heading,
                    "position_error_at_splice_m": float(position[0]),
                    "heading_error_at_splice_deg": float(heading[0]),
                }
            )
            trace_rows.extend(
                {
                    "event_id": event.event_id,
                    "scenario": event.scenario,
                    "run": event.run,
                    "method": method,
                    "steps_after_splice": offset,
                    "position_error_m": float(pos),
                    "absolute_heading_error_deg": float(ang),
                }
                for offset, (pos, ang) in enumerate(zip(position, heading))
            )
    events, traces = pd.DataFrame(event_rows), pd.DataFrame(trace_rows)
    events.to_csv(output / "event_results.csv", index=False)
    traces.to_csv(output / "event_error_traces.csv", index=False)
    summary_rows = []
    for (scenario, method), group in events.groupby(["scenario", "method"], sort=False):
        recovered = group[group.recovered]
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "events": len(group),
                "source_runs": group.run.nunique(),
                "non_recovery_fraction": 1 - group.recovered.mean(),
                "recovery_steps_median": recovered.recovery_steps.median(),
                "recovery_steps_q1": recovered.recovery_steps.quantile(0.25),
                "recovery_steps_q3": recovered.recovery_steps.quantile(0.75),
                "residual_position_median_m": recovered.residual_position_median_m.median(),
                "residual_position_q1_m": recovered.residual_position_median_m.quantile(0.25),
                "residual_position_q3_m": recovered.residual_position_median_m.quantile(0.75),
                "residual_heading_median_deg": recovered.residual_heading_median_deg.median(),
                "residual_heading_q1_deg": recovered.residual_heading_median_deg.quantile(0.25),
                "residual_heading_q3_deg": recovered.residual_heading_median_deg.quantile(0.75),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "stress_test_summary.csv", index=False)
    write_json(output / "scenario_definition.json", asdict(protocol))
    write_json(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "model_sha256": sha256_file(model_path),
            "sources": source_records,
            "candidate_counts": candidate_frame.groupby("scenario").size().to_dict(),
            "selected_counts": selected.groupby("scenario").size().to_dict(),
        },
    )
    (output / "methods_and_scope.md").write_text(
        "# Paired live-recording estimator stress test\n\n"
        "Fifty synthetic events per class are selected across nine nominal live "
        "recordings (three paths under three controllers). Each splice remains within "
        "one physical recording. The saved LSTM and primary position-only EKF replay "
        "the same event without resetting their internal state. Events are balanced "
        "across source runs and separated by at least 10 samples within each run. "
        "They reuse recorded motion and noise and are therefore dependent open-loop "
        "estimator tests, not repeated physical perturbations or controller-recovery trials.\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, required=True)
    args = parser.parse_args()
    tf.config.set_visible_devices([], "GPU")
    output_root = args.revision_root / "revision_round_1/analysis/outputs"
    compact, _, process, measurement = run_wp1_known_pose(args.robot_root, output_root)
    heading = run_heading_baseline_analysis(args.robot_root, output_root)
    initialization, initialization_paired = run_live_initialization_sensitivity(
        args.robot_root, output_root
    )
    stress = run_live_stress_test(args.robot_root, output_root, process, measurement)
    print("\nKnown-pose LSTM--EKF comparison\n", compact.to_string(index=False))
    print("\nHeading error by CNC baseline\n", heading.to_string(index=False))
    print("\nLive-segment initialization summary\n", initialization.to_string(index=False))
    print("\nPaired initialization differences\n", initialization_paired.to_string(index=False))
    print("\nLive-recording stress test\n", stress.to_string(index=False))


if __name__ == "__main__":
    main()
