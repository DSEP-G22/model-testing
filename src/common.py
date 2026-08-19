"""Shared helpers for the DSEP22 model-testing benchmarks.

Every notebook imports from here so that paths, seeding, metric computation and
results serialisation stay identical across experiments.
"""
from __future__ import annotations

import json
import os
import platform
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]           # d:/DSEP22
BENCH_ROOT = Path(__file__).resolve().parents[1]     # d:/DSEP22/model testing
DATA = ROOT / "data"

BITEXT_CSV = (
    DATA
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
ROUTER_DIR = DATA / "router detection.v38-data_video.coco"
CALLS_DIR = DATA / "archive" / "Call center data samples"

ARTIFACTS = BENCH_ROOT / "artifacts"
RESULTS = BENCH_ROOT / "results"
CACHE = BENCH_ROOT / "data_cache"

for _p in (ARTIFACTS, RESULTS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42


# --------------------------------------------------------------------------- #
# Interpreter guard
# --------------------------------------------------------------------------- #
# Fail loudly when a notebook is run on the wrong interpreter. VS Code remembers its own kernel
# choice per notebook in workspace state, which overrides the `kernelspec` the .ipynb asks for,
# so a notebook can silently execute against another project's venv. That happened here: the
# v1 project's venv sits next door with a CPU-only torch wheel, and benchmarks run against it
# report `cuda_available: false` and time every model on the CPU. Those numbers look plausible
# and are wrong, which is worse than a crash.
def _check_interpreter() -> None:
    import sys

    expected = (BENCH_ROOT / ".venv").resolve()
    running = Path(sys.executable).resolve()
    try:
        running.relative_to(expected)
    except ValueError:
        raise RuntimeError(
            f"Wrong interpreter: {running}\n"
            f"Expected the model-testing venv at: {expected}\n"
            "In VS Code use 'Select Kernel' -> 'Python (model testing)'. Headless runs should go "
            "through `python src/run_notebooks.py`, which pins the kernel by name."
        ) from None


if os.environ.get("DSEP22_SKIP_INTERPRETER_CHECK") != "1":
    _check_interpreter()


# --------------------------------------------------------------------------- #
# CUDA runtime shim for CTranslate2 (faster-whisper)
# --------------------------------------------------------------------------- #
# torch here is built against CUDA 13.2 and ships cublas64_13.dll, but ctranslate2 4.x is built
# against CUDA 12 and dlopens cublas64_12.dll by name. Without this, faster-whisper dies with
# `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded` the moment it is asked
# for a CUDA device, while torch itself reports CUDA working perfectly.
#
# The CUDA 12 cuBLAS is supplied by the `nvidia-cublas-cu12` wheel, which unpacks to a directory
# Windows does not search. Registering it here means every notebook that imports common gets it,
# and nothing has to be copied next to the ctranslate2 package.
#
# `os.add_dll_directory` alone is NOT enough: it only affects loads that go through
# LoadLibraryEx with the LOAD_LIBRARY_SEARCH_* flags, and ctranslate2 resolves cuBLAS by
# bare name, which searches PATH. Both are set here — PATH is the one that actually works.
def _register_cuda12_cublas() -> None:
    import sys

    dll_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
    if not dll_dir.is_dir():
        return
    # PATH, not os.add_dll_directory(): CTranslate2 resolves the library by bare name, which
    # searches PATH but not the directories add_dll_directory registers. The difference is
    # invisible until the first decode — constructing a WhisperModel on cuda succeeds either
    # way, so a construction-only smoke test reports success and the run still dies later.
    entry = str(dll_dir)
    if entry not in os.environ.get("PATH", ""):
        os.environ["PATH"] = entry + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(entry)


_register_cuda12_cublas()


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if present) torch, including CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@contextmanager
def timer(label: str = "", verbose: bool = True):
    """Wall-clock timer: with timer('fit') as t: ...  then t.seconds."""

    class _T:
        seconds = 0.0

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start
        if verbose and label:
            print(f"[{label}] {t.seconds:.2f}s")


def measure_latency(fn, samples, n: int = 200, warmup: int = 10) -> float:
    """Median single-item inference latency in milliseconds."""
    pool = list(samples)[: n + warmup]
    for x in pool[:warmup]:
        fn(x)
    times = []
    for x in pool[warmup:]:
        s = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - s) * 1000.0)
    return float(np.median(times)) if times else float("nan")


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    """One benchmark row; `task` groups rows into a comparison table."""

    task: str
    model: str
    family: str  # traditional | embedding | deep | unsupervised | asr
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    train_seconds: float | None = None
    infer_ms_per_item: float | None = None
    model_size_mb: float | None = None
    n_params: int | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ResultStore:
    """Append-only JSON store, one file per task, keyed by model name."""

    def __init__(self, task: str):
        self.task = task
        self.path = RESULTS / f"{task}.json"
        self.rows: dict = {}
        if self.path.exists():
            self.rows = {r["model"]: r for r in json.loads(self.path.read_text())}

    def add(self, result: Result) -> Result:
        self.rows[result.model] = result.to_dict()
        self.flush()
        print(f"  -> stored {result.model}: {fmt_metrics(result.metrics)}")
        return result

    def flush(self) -> None:
        self.path.write_text(json.dumps(list(self.rows.values()), indent=2, default=float))

    def frame(self, sort_by: str | None = None):
        import pandas as pd

        recs = []
        for r in self.rows.values():
            rec = {"model": r["model"], "family": r["family"]}
            rec.update(r["metrics"])
            rec["train_s"] = r.get("train_seconds")
            rec["infer_ms"] = r.get("infer_ms_per_item")
            rec["size_mb"] = r.get("model_size_mb")
            rec["notes"] = r.get("notes", "")
            recs.append(rec)
        df = pd.DataFrame(recs)
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        return df.reset_index(drop=True)


def fmt_metrics(m: dict) -> str:
    return ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, proba=None, labels=None) -> dict:
    """Accuracy / balanced acc / macro + weighted F1 (+ top-3 when proba given)."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None and labels is not None:
        out["top3_accuracy"] = float(top_k_accuracy(y_true, proba, labels, k=3))
    return out


def top_k_accuracy(y_true, proba, labels, k: int = 3) -> float:
    proba = np.asarray(proba)
    labels = np.asarray(labels)
    topk = labels[np.argsort(-proba, axis=1)[:, :k]]
    return float(np.mean([t in row for t, row in zip(y_true, topk)]))


def clustering_metrics(y_true, cluster_ids, X=None) -> dict:
    """External (ARI/NMI/V/purity/Hungarian acc) + internal (silhouette) scores."""
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        v_measure_score,
        silhouette_score,
    )

    y_true = np.asarray(y_true)
    cluster_ids = np.asarray(cluster_ids)
    out = {
        "ARI": float(adjusted_rand_score(y_true, cluster_ids)),
        "NMI": float(normalized_mutual_info_score(y_true, cluster_ids)),
        "V_measure": float(v_measure_score(y_true, cluster_ids)),
        "purity": float(cluster_purity(y_true, cluster_ids)),
        "hungarian_acc": float(hungarian_accuracy(y_true, cluster_ids)),
        "n_clusters": int(len(set(cluster_ids[cluster_ids >= 0]))),
        "noise_frac": float(np.mean(cluster_ids < 0)),
    }
    if X is not None:
        mask = cluster_ids >= 0
        if len(set(cluster_ids[mask])) > 1:
            out["silhouette"] = float(silhouette_score(np.asarray(X)[mask], cluster_ids[mask]))
    return out


def cluster_purity(y_true, cluster_ids) -> float:
    """Fraction of points falling in the majority true class of their cluster."""
    y_true = np.asarray(y_true)
    cluster_ids = np.asarray(cluster_ids)
    total = 0
    for c in set(cluster_ids.tolist()):
        members = y_true[cluster_ids == c]
        if len(members):
            _, counts = np.unique(members, return_counts=True)
            total += counts.max()
    return total / len(y_true)


def hungarian_accuracy(y_true, cluster_ids) -> float:
    """Best one-to-one cluster->class assignment accuracy."""
    from scipy.optimize import linear_sum_assignment

    y_true = np.asarray(y_true)
    cluster_ids = np.asarray(cluster_ids)
    classes = {c: i for i, c in enumerate(np.unique(y_true))}
    clusters = {c: i for i, c in enumerate(np.unique(cluster_ids))}
    cm = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for t, c in zip(y_true, cluster_ids):
        cm[clusters[c], classes[t]] += 1
    row, col = linear_sum_assignment(-cm)
    return cm[row, col].sum() / len(y_true)


# --------------------------------------------------------------------------- #
# Model artefacts
# --------------------------------------------------------------------------- #
def save_sklearn(model, name: str) -> Path:
    import joblib

    path = ARTIFACTS / f"{name}.joblib"
    joblib.dump(model, path, compress=3)
    return path


def save_torch(model, name: str, extra: dict | None = None) -> Path:
    import torch

    path = ARTIFACTS / f"{name}.pt"
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def file_size_mb(path) -> float:
    return round(Path(path).stat().st_size / 1024**2, 2)


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
FAMILY_COLORS = {
    "traditional": "#4C72B0",
    "embedding": "#DD8452",
    "deep": "#55A868",
    "unsupervised": "#C44E52",
    "asr": "#8172B3",
}


def plot_confusion(y_true, y_pred, labels, title: str, figsize=(10, 8), normalize=True):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_benchmark(df, metric: str = "macro_f1", title: str = "", figsize=(9, 5), higher_better=True):
    import matplotlib.pyplot as plt

    d = df.dropna(subset=[metric]).sort_values(metric, ascending=higher_better)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(d["model"], d[metric], color=[FAMILY_COLORS.get(f, "#888") for f in d["family"]])
    span = max(d[metric].max(), 1e-9)
    for i, v in enumerate(d[metric]):
        ax.text(v + span * 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel(metric)
    ax.set_title(title or metric)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
