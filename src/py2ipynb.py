"""Convert a jupytext-style percent script (`# %%` / `# %% [markdown]`) to .ipynb.

Stdlib only, so it runs before Jupyter is installed. Usage:
    python src/py2ipynb.py notebooks/_src/01_intent.py notebooks/01_intent.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def split_cells(text: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    kind, buf = "code", []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if buf:
                cells.append((kind, "\n".join(buf).strip("\n")))
                buf = []
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        buf.append(line)
    if buf:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


def strip_md(src: str) -> str:
    out = []
    for line in src.splitlines():
        out.append(line[2:] if line.startswith("# ") else line.lstrip("#").lstrip())
    return "\n".join(out)


def to_notebook(path_in: Path, path_out: Path) -> None:
    cells = []
    for kind, src in split_cells(path_in.read_text(encoding="utf-8")):
        body = strip_md(src) if kind == "markdown" else src
        lines = [l + "\n" for l in body.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            # Pin the project's own kernel by name. A generic "python3" here is resolved by
            # the editor against whatever it calls python3, which on this machine was the v1
            # project's 3.12 venv: notebooks 02-04 were executed against it and came back
            # stamped `.venv (3.12.11)`. The name must match `jupyter kernelspec list`, and is
            # what src/run_notebooks.py already passes to nbclient.
            "kernelspec": {
                "display_name": "Python (model testing)",
                "language": "python",
                "name": "dsep22-modeltesting",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{path_in.name} -> {path_out} ({len(cells)} cells)")


if __name__ == "__main__":
    to_notebook(Path(sys.argv[1]), Path(sys.argv[2]))
