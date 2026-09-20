"""Build a deterministic provenance manifest for migrated CSV files."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUTPUT = DATA_ROOT / "manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def source_path(relative: Path) -> str:
    parts = relative.parts
    filename = relative.name

    if parts[:2] == ("raw", "cnc"):
        return f"data/raw/experiments/CNC/{filename}"
    if parts[:2] == ("raw", "navigation_training"):
        return f"data/raw/experiments/Robot/{filename}"
    if parts[:2] == ("processed", "navigation_training"):
        return f"data/interim/exp_vive_navigated_paths/{filename}"
    if parts[:2] == ("processed", "vlp_calibration"):
        return f"data/external/{filename}"
    if parts[:2] == ("raw", "live_navigation"):
        return f"notebooks/exploratory/{filename}"
    if parts[:2] == ("raw", "capture_experiments"):
        return f"notebooks/exploratory/{filename}"
    raise ValueError(f"No source mapping for {relative}")


def group_name(relative: Path) -> str:
    if relative.parts[:2] == ("raw", "live_navigation"):
        return f"live_navigation/{relative.parts[2]}"
    if relative.parts[:2] == ("raw", "capture_experiments"):
        return f"capture_experiments/{relative.parts[2]}"
    return "/".join(relative.parts[:2])


def main() -> None:
    paths = sorted(
        path for path in DATA_ROOT.rglob("*.csv") if path.resolve() != OUTPUT.resolve()
    )
    fieldnames = [
        "repository_path",
        "source_repository_path",
        "group",
        "bytes",
        "data_rows",
        "sha256",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for path in paths:
            relative = path.relative_to(DATA_ROOT)
            writer.writerow(
                {
                    "repository_path": path.relative_to(ROOT).as_posix(),
                    "source_repository_path": source_path(relative),
                    "group": group_name(relative),
                    "bytes": path.stat().st_size,
                    "data_rows": data_rows(path),
                    "sha256": sha256(path),
                }
            )

    print(f"Wrote {len(paths)} entries to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
