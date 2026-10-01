"""Reproducible analyses for reviewer work packages R1.4--R1.6 and R2.5.

This module is imported by ``reviewer_revision_analysis.ipynb``.  It deliberately
keeps all thresholds, random seeds, and dataset roles in one place so the held-out
results cannot silently change through notebook state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import circstd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
import tensorflow as tf
from tensorflow.keras.models import load_model


RANDOM_SEED = 20260930
RSS_COLUMNS = ["L1", "L2", "L3", "L4"]
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


def wrap_angle(values):
    values = np.asarray(values, dtype=float)
    return (values + np.pi) % (2 * np.pi) - np.pi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _radial_errors(predicted, truth):
    return np.linalg.norm(np.asarray(predicted) - np.asarray(truth), axis=1)


def _error_metrics(position_error_m, heading_error_rad=None):
    position_error_m = np.asarray(position_error_m, dtype=float)
    row = {
        "samples": int(position_error_m.size),
        "position_median_m": float(np.median(position_error_m)),
        "position_rmse_m": float(np.sqrt(np.mean(position_error_m**2))),
        "position_p90_m": float(np.percentile(position_error_m, 90)),
        "position_max_m": float(np.max(position_error_m)),
    }
    if heading_error_rad is not None:
        heading_deg = np.degrees(np.abs(wrap_angle(heading_error_rad)))
        row.update(
            heading_median_deg=float(np.median(heading_deg)),
            heading_p90_deg=float(np.percentile(heading_deg, 90)),
        )
    return row


def _make_rnn_windows(frame, window=20):
    features = np.nan_to_num(
        frame[RNN_FEATURES].to_numpy(dtype=np.float32), nan=0.0
    )
    return np.stack(
        [features[end - window : end] for end in range(window, len(features) + 1)]
    )


def _predict_rnn(frame, model, window=20):
    windows = _make_rnn_windows(frame, window)
    xy, heading_vector = model.predict(windows, verbose=0)
    heading = np.arctan2(heading_vector[:, 0], heading_vector[:, 1])
    return pd.DataFrame(
        {"x": xy[:, 0], "y": xy[:, 1], "heading": heading},
        index=frame.index[window - 1 :],
    )


def _derive_vlp_angles(frame):
    dx = frame["vlp_x_hist"].diff().to_numpy(dtype=float)
    dy = frame["vlp_y_hist"].diff().to_numpy(dtype=float)
    heading = np.arctan2(dx, dy)
    heading_change = np.r_[np.nan, wrap_angle(np.diff(heading))]
    frame["vlp_heading_hist_rad"] = heading
    frame["vlp_heading_change_rad"] = heading_change
    return frame


def _load_cnc_frames(robot_root: Path):
    sys.path.insert(0, str(robot_root))
    from robot_vlp.data_collection.communication import process_cnc, process_vlp

    raw = robot_root / "data" / "raw" / "experiments" / "CNC"
    frames = []
    for filename in ["cnc_fingerprint_01.csv", "cnc_fingerprint_02.csv"]:
        frame = pd.read_csv(raw / filename, delimiter="|")
        frame = process_vlp(process_cnc(frame))
        frames.append(frame)
    return frames


def _load_vlp_model(robot_root: Path):
    model_dir = robot_root / "models" / "vlp" / "CNC"
    metadata_path = model_dir / "CNC_vlp_models.pkl"
    metadata = pickle.loads(metadata_path.read_bytes())
    scaler = metadata["high_acc"]["scaler"]
    model_path = model_dir / metadata["high_acc"]["model_path"]
    model = load_model(model_path, compile=False)
    return scaler, model, metadata_path, model_path


def _scale_rss(rss, scaler):
    values = np.asarray(rss, dtype=float)
    columns = list(getattr(scaler, "feature_names_in_", RSS_COLUMNS))
    return scaler.transform(pd.DataFrame(values, columns=columns))


def _vlp_predict_m(rss, scaler, model):
    scaled = _scale_rss(rss, scaler)
    return np.asarray(model(scaled, training=False), dtype=float) / 1000.0


def run_heading_uncertainty(robot_root: Path, output_root: Path):
    """Independent per-axis position residuals and empirical derived-heading error."""
    output = Path(output_root) / "heading_uncertainty"
    output.mkdir(parents=True, exist_ok=True)
    _, validation = _load_cnc_frames(Path(robot_root))
    scaler, model, metadata_path, model_path = _load_vlp_model(Path(robot_root))

    predicted_m = _vlp_predict_m(validation[RSS_COLUMNS], scaler, model)
    truth_m = validation[["cnc_x", "cnc_y"]].to_numpy(dtype=float) / 1000.0
    residual = predicted_m - truth_m
    sigma_x, sigma_y = np.std(residual, axis=0, ddof=1)
    sigma_xy = float(np.sqrt((sigma_x**2 + sigma_y**2) / 2.0))

    held_path = (
        Path(robot_root)
        / "data"
        / "interim"
        / "exp_vive_navigated_paths"
        / "exp1_9_high_acc.csv"
    )
    held = pd.read_csv(held_path).reset_index(drop=True)
    displacement = held["vive_location_change"].to_numpy(dtype=float)
    valid_displacement = displacement[np.isfinite(displacement) & (displacement > 0)]
    experimental_d = float(np.median(valid_displacement))
    propagated_sigma = float(np.sqrt(sigma_x**2 + sigma_y**2) / experimental_d)

    direct_heading = np.arctan2(
        held["vlp_x_hist"].diff().to_numpy(dtype=float),
        held["vlp_y_hist"].diff().to_numpy(dtype=float),
    )
    truth_heading = held["heading_hist_rad"].to_numpy(dtype=float)
    heading_residual = wrap_angle(direct_heading - truth_heading)
    finite = np.isfinite(heading_residual) & np.isfinite(displacement)
    empirical = heading_residual[finite]
    empirical_abs_deg = np.degrees(np.abs(empirical))
    empirical_circular_sd = float(circstd(empirical, high=np.pi, low=-np.pi))

    summary = pd.DataFrame(
        [
            {
                "validation_samples": len(validation),
                "sigma_x_m": sigma_x,
                "sigma_y_m": sigma_y,
                "pooled_sigma_xy_m": sigma_xy,
                "experimental_displacement_m": experimental_d,
                "propagated_sigma_heading_deg": np.degrees(propagated_sigma),
                "empirical_circular_sd_deg": np.degrees(empirical_circular_sd),
                "empirical_median_abs_deg": np.median(empirical_abs_deg),
                "empirical_p90_abs_deg": np.percentile(empirical_abs_deg, 90),
                "empirical_samples": empirical.size,
            }
        ]
    )
    summary.to_csv(output / "heading_uncertainty_summary.csv", index=False)
    pd.DataFrame(
        {
            "sample_index": np.flatnonzero(finite),
            "true_displacement_m": displacement[finite],
            "heading_error_rad": empirical,
            "absolute_heading_error_deg": empirical_abs_deg,
        }
    ).to_csv(output / "empirical_heading_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ordered = np.sort(empirical_abs_deg)
    ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), lw=1.8)
    ax.axvline(
        np.degrees(propagated_sigma), color="tab:red", ls="--",
        label="First-order propagated $\\sigma_\\theta$",
    )
    ax.set(xlabel="Absolute VLP-derived heading error (degrees)", ylabel="Empirical CDF")
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "heading_error_distribution.png", dpi=200)
    plt.close(fig)

    methods = f"""# Heading uncertainty analysis (R1.5)

- Position uncertainty uses independent CNC validation data (`cnc_fingerprint_02.csv`, n={len(validation)}), not radial RMSE.
- Per-axis sample standard deviations are $\\sigma_x={sigma_x:.5f}$ m and $\\sigma_y={sigma_y:.5f}$ m; the pooled isotropic value is {sigma_xy:.5f} m.
- At the held-out run's median measured step, $d={experimental_d:.5f}$ m, first-order propagation gives $\\sqrt{{\\sigma_x^2+\\sigma_y^2}}/d={np.degrees(propagated_sigma):.2f}$ degrees. This equals $\\sqrt{{2}}\\sigma_{{xy}}/d$ for the pooled isotropic definition.
- On held-out dataset 10 the empirical circular standard deviation is {np.degrees(empirical_circular_sd):.2f} degrees; median/P90 absolute errors are {np.median(empirical_abs_deg):.2f}/{np.percentile(empirical_abs_deg,90):.2f} degrees.
- The first-order expression assumes independent position errors. Consecutive fingerprint errors can be spatially correlated, so the propagated value is an approximation rather than a calibrated confidence interval.
- A two-fix heading is undefined at zero displacement and becomes poorly conditioned as displacement approaches zero. It is a motion-derived quantity, not an independent photodiode heading measurement.
"""
    (output / "methods_and_scope.md").write_text(methods)
    _json_dump(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "validation_file": str(
                Path(robot_root) / "data/raw/experiments/CNC/cnc_fingerprint_02.csv"
            ),
            "validation_sha256": sha256_file(
                Path(robot_root) / "data/raw/experiments/CNC/cnc_fingerprint_02.csv"
            ),
            "held_out_file": str(held_path),
            "held_out_sha256": sha256_file(held_path),
            "model_metadata_sha256": sha256_file(metadata_path),
            "model_sha256": sha256_file(model_path),
        },
    )
    return summary


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
    maximum_events_per_type: int = 10
    translation_min_m: float = 0.10
    translation_max_m: float = 0.20
    small_translation_max_m: float = 0.05
    small_rotation_max_deg: float = 15.0
    rotation_min_deg: float = 60.0
    rotation_max_deg: float = 120.0


def _stress_candidates(frame, protocol):
    candidates = []
    last_source = len(frame) - protocol.post_splice_steps
    for j in range(19, len(frame) - 1):
        for m in range(j + protocol.minimum_gap_steps, last_source):
            distance = float(
                np.hypot(
                    frame.loc[m, "x_hist"] - frame.loc[j, "x_hist"],
                    frame.loc[m, "y_hist"] - frame.loc[j, "y_hist"],
                )
            )
            rotation = float(
                abs(
                    np.degrees(
                        wrap_angle(
                            frame.loc[m, "heading_hist_rad"]
                            - frame.loc[j, "heading_hist_rad"]
                        )
                    )
                )
            )
            scenario = None
            if (
                protocol.translation_min_m <= distance <= protocol.translation_max_m
                and rotation < protocol.small_rotation_max_deg
            ):
                scenario = "translation only"
                score = abs(distance - 0.15) / 0.05 + rotation / 15.0
            elif (
                distance < protocol.small_translation_max_m
                and protocol.rotation_min_deg <= rotation <= protocol.rotation_max_deg
            ):
                scenario = "rotation only"
                score = distance / 0.05 + abs(rotation - 90.0) / 30.0
            elif (
                protocol.translation_min_m <= distance <= protocol.translation_max_m
                and protocol.rotation_min_deg <= rotation <= protocol.rotation_max_deg
            ):
                scenario = "combined"
                score = abs(distance - 0.15) / 0.05 + abs(rotation - 90.0) / 30.0
            if scenario is not None:
                candidates.append(
                    {
                        "scenario": scenario,
                        "pre_splice_index": j,
                        "source_index": m,
                        "translation_m": distance,
                        "rotation_deg": rotation,
                        "selection_score": score,
                    }
                )
    return pd.DataFrame(candidates)


def _select_stress_events(candidates, protocol):
    selected = []
    for scenario in ["translation only", "rotation only", "combined"]:
        group = candidates[candidates["scenario"] == scenario].sort_values(
            ["selection_score", "pre_splice_index", "source_index"]
        )
        accepted = []
        for row in group.itertuples(index=False):
            if all(
                abs(row.pre_splice_index - previous.pre_splice_index)
                >= protocol.exclusion_steps
                and abs(row.source_index - previous.source_index)
                >= protocol.exclusion_steps
                for previous in accepted
            ):
                accepted.append(row)
            if len(accepted) == protocol.maximum_events_per_type:
                break
        selected.extend(accepted)
    return pd.DataFrame([row._asdict() for row in selected])


def _build_spliced_frame(frame, j, m, protocol):
    prefix = frame.iloc[j - 18 : j + 1].copy()
    suffix = frame.iloc[m : m + protocol.post_splice_steps].copy()
    event = pd.concat([prefix, suffix], ignore_index=True)

    first_post = len(prefix)
    encoder_state = frame.loc[
        j, ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist_rad"]
    ].to_numpy(dtype=float)
    for event_index in range(first_post, len(event)):
        distance = float(event.loc[event_index, "encoder_location_change"])
        delta = float(event.loc[event_index, "encoder_heading_change_rad"])
        if not np.isfinite(distance):
            distance = 0.0
        if not np.isfinite(delta):
            delta = 0.0
        encoder_state[2] = wrap_angle(encoder_state[2] + delta)
        encoder_state[0] += distance * np.sin(encoder_state[2])
        encoder_state[1] += distance * np.cos(encoder_state[2])
        event.loc[
            event_index,
            ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist_rad"],
        ] = encoder_state
    event = _derive_vlp_angles(event)
    return event, first_post


def run_wp2_stress_test(robot_root: Path, output_root: Path, rnn_model=None):
    output = Path(output_root) / "WP2_stress_test"
    output.mkdir(parents=True, exist_ok=True)
    protocol = StressProtocol()
    held_path = (
        Path(robot_root)
        / "data/interim/exp_vive_navigated_paths/exp1_9_high_acc.csv"
    )
    frame = pd.read_csv(held_path).reset_index(drop=True)
    if rnn_model is None:
        rnn_model = load_model(
            Path(robot_root) / "models/navigation_neural_nets/rnn.keras", compile=False
        )

    candidates = _stress_candidates(frame, protocol)
    selected = _select_stress_events(candidates, protocol)
    selected.insert(0, "event_id", np.arange(1, len(selected) + 1))
    selected.to_csv(output / "selected_splices.csv", index=False)
    _json_dump(output / "scenario_definition.json", asdict(protocol))

    event_rows = []
    traces = []
    for event in selected.itertuples(index=False):
        spliced, first_post = _build_spliced_frame(
            frame, int(event.pre_splice_index), int(event.source_index), protocol
        )
        prediction = _predict_rnn(spliced, rnn_model)
        post_indices = np.arange(first_post, len(spliced))
        estimate = prediction.loc[post_indices]
        truth = spliced.loc[post_indices]
        pos = _radial_errors(
            estimate[["x", "y"]], truth[["x_hist", "y_hist"]]
        )
        ang = wrap_angle(
            estimate["heading"].to_numpy()
            - truth["heading_hist_rad"].to_numpy()
        )
        success = (pos < protocol.recovery_position_m) & (
            np.degrees(np.abs(ang)) < protocol.recovery_heading_deg
        )
        recovery_offset = None
        for offset in range(protocol.evaluation_horizon_steps):
            if (
                offset + protocol.recovery_consecutive_steps <= len(success)
                and success[offset : offset + protocol.recovery_consecutive_steps].all()
            ):
                recovery_offset = offset
                break

        recovered = recovery_offset is not None
        if recovered:
            residual_slice = slice(
                recovery_offset,
                min(recovery_offset + protocol.residual_window_steps, len(pos)),
            )
            residual_position = float(np.median(pos[residual_slice]))
            residual_heading = float(
                np.median(np.degrees(np.abs(ang[residual_slice])))
            )
        else:
            residual_position = np.nan
            residual_heading = np.nan
        event_rows.append(
            {
                **event._asdict(),
                "recovered": recovered,
                "recovery_steps": recovery_offset if recovered else np.nan,
                "residual_position_median_m": residual_position,
                "residual_heading_median_deg": residual_heading,
                "position_error_at_splice_m": pos[0],
                "heading_error_at_splice_deg": np.degrees(abs(ang[0])),
            }
        )
        for offset, (p_error, h_error) in enumerate(zip(pos, ang)):
            traces.append(
                {
                    "event_id": event.event_id,
                    "scenario": event.scenario,
                    "steps_after_splice": offset,
                    "position_error_m": p_error,
                    "absolute_heading_error_deg": np.degrees(abs(h_error)),
                }
            )

    event_results = pd.DataFrame(event_rows)
    trace_frame = pd.DataFrame(traces)
    event_results.to_csv(output / "event_results.csv", index=False)
    trace_frame.to_csv(output / "event_error_traces.csv", index=False)

    summary_rows = []
    for scenario, group in event_results.groupby("scenario", sort=False):
        recovered = group[group["recovered"]]
        summary_rows.append(
            {
                "scenario": scenario,
                "events": len(group),
                "non_recovery_fraction": 1.0 - group["recovered"].mean(),
                "recovery_steps_median": recovered["recovery_steps"].median(),
                "recovery_steps_q1": recovered["recovery_steps"].quantile(0.25),
                "recovery_steps_q3": recovered["recovery_steps"].quantile(0.75),
                "residual_position_median_m": recovered[
                    "residual_position_median_m"
                ].median(),
                "residual_position_q1_m": recovered[
                    "residual_position_median_m"
                ].quantile(0.25),
                "residual_position_q3_m": recovered[
                    "residual_position_median_m"
                ].quantile(0.75),
                "residual_heading_median_deg": recovered[
                    "residual_heading_median_deg"
                ].median(),
                "residual_heading_q1_deg": recovered[
                    "residual_heading_median_deg"
                ].quantile(0.25),
                "residual_heading_q3_deg": recovered[
                    "residual_heading_median_deg"
                ].quantile(0.75),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "stress_test_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    colours = {
        "translation only": "tab:blue",
        "rotation only": "tab:orange",
        "combined": "tab:green",
    }
    for scenario, group in trace_frame.groupby("scenario"):
        pivot_p = group.pivot(
            index="steps_after_splice", columns="event_id", values="position_error_m"
        )
        pivot_h = group.pivot(
            index="steps_after_splice",
            columns="event_id",
            values="absolute_heading_error_deg",
        )
        axes[0].plot(pivot_p.index, pivot_p.median(axis=1), label=scenario, color=colours[scenario])
        axes[1].plot(pivot_h.index, pivot_h.median(axis=1), label=scenario, color=colours[scenario])
    axes[0].axhline(protocol.recovery_position_m, color="0.35", ls="--", lw=1)
    axes[1].axhline(protocol.recovery_heading_deg, color="0.35", ls="--", lw=1)
    axes[0].set(ylabel="Median position error (m)", xlabel="Steps after splice")
    axes[1].set(ylabel="Median absolute heading error (degrees)", xlabel="Steps after splice")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlim(0, protocol.evaluation_horizon_steps - 1)
    fig.tight_layout()
    fig.savefig(output / "stress_test_diagnostics.png", dpi=200)
    plt.close(fig)

    counts = summary.set_index("scenario")["events"].to_dict()
    methods = f"""# Synthetic estimator stress test (WP2; R1.4)

This is an offline, open-loop estimator stress test on held-out dataset 10, not a repeated physical trial and not evidence of controller recovery. The recorded run is spliced after a 19-sample history. Post-splice VLP and ground truth start at a later recorded pose, while encoder pose features continue by integrating the post-splice commands from the pre-splice encoder state. VLP heading and heading change are recomputed across the splice.

Scenario thresholds, recovery thresholds, and the 50-step horizon were fixed before model evaluation and are recorded in `scenario_definition.json`. The 0.05 m and 10 degree recovery thresholds are operational analysis thresholds (approximately twice the fingerprint-position RMSE and a coarse navigation-heading tolerance), not safety tolerances. Recovery requires five consecutive successful steps. Residual errors use the first 20 steps after recovery. Events not meeting the criterion within 50 steps are non-recoveries.

Candidate events are ranked only by proximity to the centres of their scenario ranges. Within each scenario type, accepted pre-splice and source indices are separated by at least 10 samples; at most 10 events are retained. This produced {counts}. Different events still reuse portions of one recorded trajectory and are therefore not independent experimental repetitions.
"""
    (output / "methods_and_scope.md").write_text(methods)
    _json_dump(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "held_out_file": str(held_path),
            "held_out_sha256": sha256_file(held_path),
            "rnn_model_sha256": sha256_file(
                Path(robot_root) / "models/navigation_neural_nets/rnn.keras"
            ),
            "candidate_counts": candidates.groupby("scenario").size().to_dict(),
            "selected_counts": counts,
        },
    )
    return summary, event_results


def _fingerprint_metrics(prediction_mm, truth_mm):
    errors_m = _radial_errors(prediction_mm, truth_mm) / 1000.0
    return _error_metrics(errors_m)


def _median_query_ms(callable_predict, values, repeats=25):
    callable_predict(values[: min(16, len(values))])
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_predict(values)
        samples.append((time.perf_counter() - start) * 1000.0 / len(values))
    return float(np.median(samples))


def run_wp3_fingerprint_comparison(robot_root: Path, output_root: Path):
    output = Path(output_root) / "WP3_fingerprint_models"
    output.mkdir(parents=True, exist_ok=True)
    training, validation = _load_cnc_frames(Path(robot_root))
    X_train = training[RSS_COLUMNS].to_numpy(dtype=float)
    y_train = training[["cnc_x", "cnc_y"]].to_numpy(dtype=float)
    X_valid = validation[RSS_COLUMNS].to_numpy(dtype=float)
    y_valid = validation[["cnc_x", "cnc_y"]].to_numpy(dtype=float)
    scaler, network, metadata_path, network_path = _load_vlp_model(Path(robot_root))

    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_rows = []
    for k in range(1, 16):
        fold_errors = []
        for fold, (train_index, test_index) in enumerate(folds.split(X_train), start=1):
            estimator = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=k, weights="distance", p=2),
            )
            estimator.fit(X_train[train_index], y_train[train_index])
            error = _radial_errors(estimator.predict(X_train[test_index]), y_train[test_index])
            fold_errors.extend(error.tolist())
            cv_rows.append(
                {
                    "k": k,
                    "fold": fold,
                    "fold_radial_rmse_mm": np.sqrt(np.mean(error**2)),
                }
            )
        cv_rows.append(
            {
                "k": k,
                "fold": "pooled",
                "fold_radial_rmse_mm": np.sqrt(np.mean(np.asarray(fold_errors) ** 2)),
            }
        )
    cv = pd.DataFrame(cv_rows)
    pooled = cv[cv["fold"] == "pooled"].sort_values(
        ["fold_radial_rmse_mm", "k"]
    )
    selected_k = int(pooled.iloc[0]["k"])
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=selected_k, weights="distance", p=2),
    )
    knn.fit(X_train, y_train)

    gp_input_scaler = StandardScaler().fit(X_train)
    X_train_gp = gp_input_scaler.transform(X_train)
    X_valid_gp = gp_input_scaler.transform(X_valid)
    gps = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for coordinate in range(2):
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(np.ones(4), (1e-2, 1e2))
                + WhiteKernel(1.0, (1e-5, 1e2))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=RANDOM_SEED,
            )
            gp.fit(X_train_gp, y_train[:, coordinate])
            gps.append(gp)

    network_prediction = np.asarray(
        network(_scale_rss(X_valid, scaler), training=False), dtype=float
    )
    knn_prediction = knn.predict(X_valid)
    gp_prediction = np.column_stack([gp.predict(X_valid_gp) for gp in gps])

    scaler_bytes = len(pickle.dumps(scaler, protocol=pickle.HIGHEST_PROTOCOL))
    memory = {
        "Feedforward network": network_path.stat().st_size + scaler_bytes,
        "IDW kNN": len(pickle.dumps(knn, protocol=pickle.HIGHEST_PROTOCOL)),
        "Gaussian process": len(
            pickle.dumps((gp_input_scaler, gps), protocol=pickle.HIGHEST_PROTOCOL)
        ),
    }
    latency = {
        "Feedforward network": _median_query_ms(
            lambda values: np.asarray(
                network(_scale_rss(values, scaler), training=False), dtype=float
            ),
            X_valid,
        ),
        "IDW kNN": _median_query_ms(knn.predict, X_valid),
        "Gaussian process": _median_query_ms(
            lambda values: np.column_stack(
                [gp.predict(gp_input_scaler.transform(values)) for gp in gps]
            ),
            X_valid,
        ),
    }
    predictions = {
        "Feedforward network": network_prediction,
        "IDW kNN": knn_prediction,
        "Gaussian process": gp_prediction,
    }
    result_rows = []
    for method, prediction in predictions.items():
        result_rows.append(
            {
                "method": method,
                **_fingerprint_metrics(prediction, y_valid),
                "serialized_memory_kib": memory[method] / 1024.0,
                "median_batched_query_ms": latency[method],
            }
        )
    results = pd.DataFrame(result_rows)
    results.to_csv(output / "fingerprint_model_comparison.csv", index=False)
    cv.to_csv(output / "knn_cross_validation.csv", index=False)
    pd.DataFrame(
        {
            "sample_index": np.arange(len(validation)),
            **{
                method: _radial_errors(prediction, y_valid) / 1000.0
                for method, prediction in predictions.items()
            },
        }
    ).to_csv(output / "validation_error_traces.csv", index=False)

    configuration = {
        "random_seed": RANDOM_SEED,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "rss_columns": RSS_COLUMNS,
        "knn_candidates": list(range(1, 16)),
        "knn_cv": "five-fold shuffled training-only CV; radial RMSE selection",
        "selected_k": selected_k,
        "gp_models": [str(gp.kernel_) for gp in gps],
        "gp_preprocessing": "training-fitted StandardScaler; separate x/y GPs",
        "gp_kernel": "Constant * anisotropic RBF + WhiteKernel",
        "memory_definition": "serialized estimator and required scaler; Keras file includes saved artifact metadata",
        "latency_definition": "median CPU batch wall time divided by 528 queries, 25 repeats",
    }
    _json_dump(output / "model_configurations.json", configuration)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for method, prediction in predictions.items():
        ordered = np.sort(_radial_errors(prediction, y_valid) / 1000.0)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=method)
    ax.set(xlabel="Radial position error (m)", ylabel="Empirical CDF", xlim=(0, None))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "fingerprint_model_error_cdf.png", dpi=200)
    plt.close(fig)

    methods = f"""# Fingerprint-regressor comparison (WP3; R1.6)

All methods use the paper's four FFT-peak features, the same 1,517-point CNC training grid, and the independent 528-point perturbed-grid validation set. The saved feedforward model is evaluated without retraining. Inverse-distance-weighted kNN selects k={selected_k} by five-fold training-only cross-validation over k=1--15. Separate x/y Gaussian-process regressors use a constant-times-anisotropic-RBF plus white-noise kernel; hyperparameters are fitted by marginal likelihood using training data only.

Memory is the serialized estimator plus required preprocessing object. Keras serialization contains framework metadata as well as weights, so it is an artifact-size comparison rather than bare parameter memory. Query times are CPU measurements for a 528-sample batch divided by batch size; they are host measurements, not embedded-device benchmarks.
"""
    (output / "methods_and_scope.md").write_text(methods)
    _json_dump(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "scikit_learn": sklearn.__version__,
            "tensorflow": tf.__version__,
            "training_sha256": sha256_file(
                Path(robot_root) / "data/raw/experiments/CNC/cnc_fingerprint_01.csv"
            ),
            "validation_sha256": sha256_file(
                Path(robot_root) / "data/raw/experiments/CNC/cnc_fingerprint_02.csv"
            ),
            "model_metadata_sha256": sha256_file(metadata_path),
            "model_sha256": sha256_file(network_path),
        },
    )
    return results, configuration


def _evaluate_vlp_and_rnn(frame, rnn_model, evaluation_indices):
    prediction = _predict_rnn(frame, rnn_model)
    truth_xy = frame.loc[evaluation_indices, ["x_hist", "y_hist"]].to_numpy()
    truth_heading = frame.loc[evaluation_indices, "heading_hist_rad"].to_numpy()
    vlp_position = _radial_errors(
        frame.loc[evaluation_indices, ["vlp_x_hist", "vlp_y_hist"]], truth_xy
    )
    vlp_heading = wrap_angle(
        frame.loc[evaluation_indices, "vlp_heading_hist_rad"].to_numpy()
        - truth_heading
    )
    rnn_position = _radial_errors(
        prediction.loc[evaluation_indices, ["x", "y"]], truth_xy
    )
    rnn_heading = wrap_angle(
        prediction.loc[evaluation_indices, "heading"].to_numpy() - truth_heading
    )
    return {
        "VLP only": _error_metrics(vlp_position, vlp_heading),
        "LSTM fusion": _error_metrics(rnn_position, rnn_heading),
    }


def run_wp4_source_gain_sensitivity(robot_root: Path, output_root: Path, rnn_model=None):
    output = Path(output_root) / "WP4_source_gain_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    held_path = (
        Path(robot_root)
        / "data/interim/exp_vive_navigated_paths/exp1_9_high_acc.csv"
    )
    raw_path = Path(robot_root) / "data/raw/experiments/Robot/exp1_9.csv"
    frame = pd.read_csv(held_path).reset_index(drop=True)
    raw_rss_available = all(column in frame for column in RSS_COLUMNS) and frame[
        RSS_COLUMNS
    ].notna().all().all()
    if not raw_rss_available:
        note = "Raw four-carrier RSS is not available; no perturbation analysis was run.\n"
        (output / "rss_availability.md").write_text(note)
        return pd.DataFrame(), {"raw_rss_available": False}

    scaler, vlp_model, metadata_path, vlp_model_path = _load_vlp_model(Path(robot_root))
    if rnn_model is None:
        rnn_model = load_model(
            Path(robot_root) / "models/navigation_neural_nets/rnn.keras", compile=False
        )
    rss = frame[RSS_COLUMNS].to_numpy(dtype=float)
    evaluation_indices = frame.index[19 : 19 + 300].to_numpy()

    # Reproduce the constant Vive-alignment patch used by the submitted paper's
    # offline LSTM evaluation.  The same fixed encoder features are used for every
    # RSS condition, so the only manipulated quantities are the VLP features.
    alignment_index = 19
    encoder_columns = ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist_rad"]
    truth_columns = ["x_hist", "y_hist", "heading_hist_rad"]
    encoder_patch = (
        frame.loc[alignment_index, truth_columns].to_numpy(dtype=float)
        - frame.loc[alignment_index, encoder_columns].to_numpy(dtype=float)
    )

    def frame_from_rss(perturbed_rss):
        result = frame.copy()
        result[encoder_columns] = (
            result[encoder_columns].to_numpy(dtype=float) + encoder_patch
        )
        result[["vlp_x_hist", "vlp_y_hist"]] = _vlp_predict_m(
            perturbed_rss, scaler, vlp_model
        )
        return _derive_vlp_angles(result)

    reconstructed = frame_from_rss(rss)
    reconstruction_difference = np.max(
        np.abs(
            reconstructed[["vlp_x_hist", "vlp_y_hist"]].to_numpy()
            - frame[["vlp_x_hist", "vlp_y_hist"]].to_numpy()
        )
    )

    deterministic_rows = []
    condition_frames = [("baseline", "none", 1.0, reconstructed)]
    factors = [0.80, 0.90, 0.95, 1.05, 1.10, 1.20]
    for factor in factors:
        altered = rss * factor
        condition_frames.append(
            (f"all LEDs {factor - 1:+.0%}", "all LEDs", factor, frame_from_rss(altered))
        )
        for led_index, led in enumerate(RSS_COLUMNS):
            altered = rss.copy()
            altered[:, led_index] *= factor
            condition_frames.append(
                (f"{led} {factor - 1:+.0%}", led, factor, frame_from_rss(altered))
            )
    for condition, target, factor, altered_frame in condition_frames:
        metrics = _evaluate_vlp_and_rnn(altered_frame, rnn_model, evaluation_indices)
        for method, values in metrics.items():
            deterministic_rows.append(
                {
                    "condition": condition,
                    "gain_target": target,
                    "gain_factor": factor,
                    "method": method,
                    **values,
                }
            )
    deterministic = pd.DataFrame(deterministic_rows)
    deterministic.to_csv(output / "systematic_gain_results.csv", index=False)

    rng = np.random.default_rng(RANDOM_SEED)
    random_rows = []
    for sigma in [0.02, 0.05, 0.10]:
        for replicate in range(30):
            factors_random = np.clip(
                1.0 + rng.normal(0.0, sigma, size=rss.shape), 0.01, None
            )
            altered_frame = frame_from_rss(rss * factors_random)
            metrics = _evaluate_vlp_and_rnn(
                altered_frame, rnn_model, evaluation_indices
            )
            for method, values in metrics.items():
                random_rows.append(
                    {
                        "sigma_fraction": sigma,
                        "replicate": replicate,
                        "method": method,
                        **values,
                    }
                )
    random_results = pd.DataFrame(random_rows)
    random_results.to_csv(output / "random_gain_replicates.csv", index=False)
    random_summary = (
        random_results.groupby(["sigma_fraction", "method"])
        .agg(
            position_median_m=("position_median_m", "median"),
            position_median_q1_m=("position_median_m", lambda x: x.quantile(0.25)),
            position_median_q3_m=("position_median_m", lambda x: x.quantile(0.75)),
            position_p90_m=("position_p90_m", "median"),
            heading_median_deg=("heading_median_deg", "median"),
            heading_p90_deg=("heading_p90_deg", "median"),
        )
        .reset_index()
    )
    random_summary.to_csv(output / "random_gain_summary.csv", index=False)

    # A compact, pre-specified paper table: baseline, all-channel cases, and the
    # worst single-channel position median at each factor (not selected post hoc
    # for significance; "worst" is the stated sensitivity envelope).
    compact = deterministic[
        (deterministic["condition"] == "baseline")
        | (deterministic["gain_target"] == "all LEDs")
    ].copy()
    single = deterministic[deterministic["gain_target"].isin(RSS_COLUMNS)].copy()
    worst_index = single.groupby(["gain_factor", "method"])[
        "position_median_m"
    ].idxmax()
    worst = single.loc[worst_index].copy()
    worst["condition"] = "worst single LED " + worst["gain_factor"].map(
        lambda value: f"{value - 1:+.0%}"
    )
    compact = pd.concat([compact, worst], ignore_index=True)
    compact.to_csv(output / "source_gain_compact_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    all_led = deterministic[deterministic["gain_target"].isin(["none", "all LEDs"])]
    for method, group in all_led.groupby("method"):
        group = group.sort_values("gain_factor")
        axes[0].plot(group["gain_factor"], group["position_median_m"], marker="o", label=method)
        axes[1].plot(group["gain_factor"], group["heading_median_deg"], marker="o", label=method)
    axes[0].set(xlabel="Common RSS gain factor", ylabel="Median position error (m)")
    axes[1].set(xlabel="Common RSS gain factor", ylabel="Median heading error (degrees)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "all_channel_gain_sensitivity.png", dpi=200)
    plt.close(fig)

    audit = {
        "raw_rss_available": True,
        "interim_columns": RSS_COLUMNS,
        "raw_waveform_available": "vlp_data" in frame.columns,
        "navigation_rows": len(frame),
        "evaluation_samples": len(evaluation_indices),
        "reconstruction_max_abs_position_difference_m": float(reconstruction_difference),
        "systematic_factors": factors,
        "systematic_targets": RSS_COLUMNS + ["all LEDs"],
        "random_sigmas": [0.02, 0.05, 0.10],
        "random_replicates": 30,
        "random_seed": RANDOM_SEED,
        "lstm_encoder_initialization": "submitted-paper Vive-aligned offline evaluation",
        "encoder_alignment_index": alignment_index,
        "encoder_alignment_patch": encoder_patch.tolist(),
    }
    _json_dump(output / "perturbation_specification.json", audit)
    note = f"""# Raw RSS availability and scope (WP4; R2.5)

Held-out navigation dataset 10 contains finite FFT-peak amplitudes for all four carriers on all {len(frame)} retained movement cycles, as well as archived ADC waveforms. Re-running the saved fingerprint model from the stored four peaks reproduces the saved VLP positions to a maximum absolute coordinate difference of {reconstruction_difference:.3e} m.

The offline analysis applies common receiver/source gain changes, one-carrier gain changes, and independent random multiplicative changes to the recorded FFT peaks before re-running the saved fingerprint model and LSTM. These perturbations characterize modulated-source/receiver-gain sensitivity only. They are not measurements of ambient illumination.

The receiver subtracts the sample mean, applies a Hann window, and extracts peaks near 15, 17, 20, and 23 kHz. This rejects DC and attenuates low-frequency ambient variation while the analog front end remains linear. It does not establish immunity to increased shot noise, front-end saturation, or interference near a carrier. Controlled daylight, artificial-light, saturation, and near-carrier interference tests were not performed.
"""
    (output / "rss_availability_and_scope.md").write_text(note)
    _json_dump(
        output / "provenance.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "held_out_file": str(held_path),
            "held_out_sha256": sha256_file(held_path),
            "raw_navigation_file": str(raw_path),
            "raw_navigation_sha256": sha256_file(raw_path),
            "vlp_metadata_sha256": sha256_file(metadata_path),
            "vlp_model_sha256": sha256_file(vlp_model_path),
            "rnn_model_sha256": sha256_file(
                Path(robot_root) / "models/navigation_neural_nets/rnn.keras"
            ),
        },
    )
    return compact, random_summary
