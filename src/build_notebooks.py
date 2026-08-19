"""Rebuild every notebook in notebooks/ from its percent-script source."""
from pathlib import Path

from py2ipynb import to_notebook

SRC = Path(__file__).resolve().parents[1] / "notebooks" / "_src"
OUT = Path(__file__).resolve().parents[1] / "notebooks"

if __name__ == "__main__":
    for py in sorted(SRC.glob("*.py")):
        to_notebook(py, OUT / f"{py.stem}.ipynb")
