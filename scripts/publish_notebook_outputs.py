"""Publish compact, GitHub-friendly outputs from executed notebook copies."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXECUTED = ROOT / "results" / "executed_notebooks"
PUBLISH = ("01_vlp_dataset", "02_navigation_dataset", "04_results")
STREAM_LIMIT = 2_000
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_outputs(outputs: list[dict], notebook: str, cell_index: int) -> list[dict]:
    cleaned: list[dict] = []
    stream_chars = 0

    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "error":
            raise RuntimeError(f"Execution error in {notebook}, cell {cell_index}")
        if output_type == "stream":
            if output.get("name") == "stderr":
                continue
            value = output.get("text", "")
            text = "".join(value) if isinstance(value, list) else value
            if any(character in text for character in ("\r", "\b", "\x1b")):
                continue
            text = ANSI.sub("", text)
            remaining = max(STREAM_LIMIT - stream_chars, 0)
            if not remaining:
                continue
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "\n… output truncated …\n"
            stream_chars += len(text)
            if not text.strip():
                continue
            output = {**output, "text": text}
        cleaned.append(output)

    return cleaned


def publish(name: str) -> None:
    source_path = NOTEBOOKS / f"{name}.ipynb"
    executed_path = EXECUTED / f"{name}.executed.ipynb"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    executed = json.loads(executed_path.read_text(encoding="utf-8"))

    if len(source["cells"]) != len(executed["cells"]):
        raise RuntimeError(f"Cell count differs for {name}")

    for index, (target, completed) in enumerate(zip(source["cells"], executed["cells"])):
        if target.get("source") != completed.get("source"):
            raise RuntimeError(f"Cell {index} source differs for {name}")
        if target.get("cell_type") != "code":
            continue
        target["execution_count"] = completed.get("execution_count")
        target["outputs"] = clean_outputs(completed.get("outputs", []), name, index)
        stream_size = sum(
            len(output.get("text", ""))
            for output in target["outputs"]
            if output.get("output_type") == "stream"
        )
        if stream_size > 1_200:
            target.setdefault("metadata", {})["scrolled"] = True

    source_path.write_text(
        json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Published sanitized outputs: {source_path.relative_to(ROOT)}")


def main() -> None:
    for name in PUBLISH:
        publish(name)


if __name__ == "__main__":
    main()
