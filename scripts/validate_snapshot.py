"""Validate the structure and integrity of the migrated research snapshot."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        entries = list(csv.DictReader(stream))

    if len(entries) != 41:
        raise SystemExit(f"Expected 41 CSV entries, found {len(entries)}")

    missing: list[str] = []
    changed: list[str] = []
    forbidden: list[str] = []

    for entry in entries:
        relative = entry["repository_path"]
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256(path) != entry["sha256"]:
            changed.append(relative)
        if "ekf" in relative.lower():
            forbidden.append(relative)

    for path in (ROOT / "data").rglob("*"):
        if path.is_file() and "ekf" in path.name.lower():
            forbidden.append(path.relative_to(ROOT).as_posix())

    if missing or changed or forbidden:
        if missing:
            print("Missing files:", *missing, sep="\n  ")
        if changed:
            print("Checksum mismatches:", *changed, sep="\n  ")
        if forbidden:
            print("Forbidden EKF data files:", *sorted(set(forbidden)), sep="\n  ")
        raise SystemExit(1)

    print(f"Validated {len(entries)} CSV files; no EKF data files were found.")


if __name__ == "__main__":
    main()
