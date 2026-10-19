# Round-one reviewer analysis

This directory publishes the machine-readable evidence added during the
round-one revision of *Fusion of Visible Light Positioning and Odometry for Robot
Navigation* (Sensors-103366-2026).

The original live EKF implementation and its experimental outputs remain excluded
because they contained implementation errors. `WP1_ekf_comparison` instead contains
a newly implemented, offline position-only EKF comparator and one explicitly
secondary, one-step VLP-derived-heading sensitivity variant evaluated on Dataset 10.
No closed-loop EKF trial is claimed.

Dataset 10 was excluded from LSTM fitting and EKF covariance estimation, but it
was inspected during exploratory EKF formulation development. The EKF comparison
is therefore descriptive and is not an unbiased test of EKF design selection.

`outputs/` retains configurations, complete error traces, summary tables, figures,
scope notes, and provenance for the revision work packages. The executed source
notebook and its helper modules are included for auditability. The final WP1, WP2,
WP5, and heading-uncertainty protocols are in `revised_main_concerns.py`; older
notebook cells are development history. The analyses were executed against the
archival `robot-vlp` checkout, so rerunning their full upstream preprocessing
still requires that checkout and its environment. The equivalent recorded datasets
and saved models are in this repository. The retained traces and summaries can be
checked directly here with `verify_headline_results.py`.

## Verify headline statistics

The following command independently recomputes the principal WP1, WP2, WP3, WP5,
WP6, and heading-uncertainty statistics from the retained trace CSVs and compares
them with the saved summaries:

```bash
python revision_analysis/verify_headline_results.py \
  revision_analysis/outputs \
  --json revision_analysis/headline_verification.json
```

## Rebuild the revised temporal-error figure

```bash
python revision_analysis/outputs/WP9_figure8/plot_figure8.py \
  --source-csv revision_analysis/outputs/WP9_figure8/figure8_source_data.csv \
  --output-dir revision_analysis/outputs/WP9_figure8
```

The prior-work table source and page-level primary-source audit are under
`outputs/WP8_prior_work/`.
