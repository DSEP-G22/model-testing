"""Disk cache for ASR hypotheses, keyed by (model, task, call).

Transcribing the 106-minute call corpus takes tens of minutes per model, so a notebook
re-run — after a crash, a added model, or an edit further down the notebook — should not
have to redo work it has already done. Each entry stores the hypothesis, the measured
processing time and the detected language, so cached rows reproduce the original
benchmark exactly.

Timings in a cached entry come from the run that produced it. Delete the cache (or pass
`force=True`) before quoting real-time factors measured on different hardware.
"""
from __future__ import annotations

import json
from pathlib import Path

from common import CACHE

CACHE_PATH = CACHE / "asr_hypotheses.json"


def _load() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def key(model: str, task: str, call_id: str) -> str:
    return f"{model}|{task}|{call_id}"


def get(model: str, task: str, call_id: str) -> dict | None:
    return _load().get(key(model, task, call_id))


def put(model: str, task: str, call_id: str, text: str, proc_s: float,
        language: str | None) -> None:
    data = _load()
    data[key(model, task, call_id)] = {
        "text": text,
        "proc_s": float(proc_s),
        "language": language,
    }
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def stats() -> dict:
    data = _load()
    models: dict[str, int] = {}
    for k in data:
        model, task, _ = k.split("|", 2)
        models[f"{model}/{task}"] = models.get(f"{model}/{task}", 0) + 1
    return models


def clear(model: str | None = None) -> int:
    """Drop the whole cache, or just the entries for one model. Returns rows removed."""
    data = _load()
    if model is None:
        CACHE_PATH.unlink(missing_ok=True)
        return len(data)
    keep = {k: v for k, v in data.items() if not k.startswith(f"{model}|")}
    removed = len(data) - len(keep)
    CACHE_PATH.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    return removed
