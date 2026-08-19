"""Execute the benchmark notebooks headlessly and write the outputs back in place.

    python src/run_notebooks.py                 # all four, in order
    python src/run_notebooks.py 01 03           # only the matching notebooks

Each notebook keeps its executed outputs (tables, figures, logs), so the committed
.ipynb is the record of the run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
TIMEOUT = 60 * 90  # per cell; fine-tuning cells are the long ones


def run(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name="dsep22-modeltesting",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    start = time.perf_counter()
    print(f"\n=== executing {path.name} ===", flush=True)
    try:
        client.execute()
        ok = True
    except Exception as exc:  # keep the partial outputs for debugging
        print(f"!!! {path.name} failed: {type(exc).__name__}: {exc}", flush=True)
        ok = False
    nbformat.write(nb, path)
    print(f"=== {path.name} {'finished' if ok else 'FAILED'} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ===", flush=True)
    return ok


if __name__ == "__main__":
    wanted = sys.argv[1:]
    paths = sorted(NB_DIR.glob("*.ipynb"))
    if wanted:
        paths = [p for p in paths if any(w in p.name for w in wanted)]
    results = {p.name: run(p) for p in paths}
    print("\n--- summary ---")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(results.values()) else 1)
