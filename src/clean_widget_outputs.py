"""Strip dead ipywidget outputs from executed notebooks.

tqdm (via sentence-transformers, transformers and whisper) emits progress bars as
`application/vnd.jupyter.widget-view+json` outputs. nbclient does not persist the
matching widget *state* into notebook metadata, so a saved run shows

    Could not render content for 'application/vnd.jupyter.widget-view+json'

instead of a progress bar. The bars carry no information once a run has finished, so
this removes them (and the stray carriage-return progress spam in stream outputs)
while leaving every real output — tables, figures, logs — untouched.

    python src/clean_widget_outputs.py                # all notebooks
    python src/clean_widget_outputs.py 01 03          # only matching ones
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
# tqdm redraws one line with \r; keep only the final state of each such line.
PROGRESS_RE = re.compile(r"^.*\r(?!\n)")


def clean_output(out: dict) -> dict | None:
    """Return the cleaned output, or None if it should be dropped entirely."""
    if out.get("output_type") == "display_data":
        data = out.get("data", {})
        if WIDGET_MIME in data:
            # Keep the plain-text fallback if tqdm left one, otherwise drop the output.
            text = data.get("text/plain")
            if not text or "".join(text).strip() in ("", "…"):
                return None
            out["data"] = {"text/plain": text}
    if out.get("output_type") == "stream":
        text = "".join(out.get("text", []))
        collapsed = "\n".join(PROGRESS_RE.sub("", line) for line in text.split("\n"))
        if not collapsed.strip():
            return None
        out["text"] = collapsed
    return out


def clean_notebook(path: Path) -> tuple[int, int]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    dropped = kept = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        new_outputs = []
        for out in cell.get("outputs", []):
            cleaned = clean_output(out)
            if cleaned is None:
                dropped += 1
            else:
                kept += 1
                new_outputs.append(cleaned)
        cell["outputs"] = new_outputs
    # The widget state block references the same dead widgets; drop it too.
    nb.get("metadata", {}).pop("widgets", None)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return dropped, kept


if __name__ == "__main__":
    wanted = sys.argv[1:]
    paths = sorted(NB_DIR.glob("*.ipynb"))
    if wanted:
        paths = [p for p in paths if any(w in p.name for w in wanted)]
    for p in paths:
        dropped, kept = clean_notebook(p)
        print(f"{p.name}: dropped {dropped} widget/progress outputs, kept {kept}")
