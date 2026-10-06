"""Fetch every translation candidate used by notebook 09 into data_cache/ (idempotent).

Pre-built CTranslate2 int8 builds are pulled as-is; HF checkpoints without one are pulled and
converted to CTranslate2 int8 (as the mt-benchmark workflow does for 600M), so every
encoder-decoder runs on the production runtime. The local link is ~70 KB/s from the Hub, so
this runs on the `mt-validation` workflow.

    python src/mt_fetch.py [name ...]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
WEIGHTS = ("pytorch_model.bin", "model.safetensors")

# name -> (repo, pinned revision, needs CT2 int8 conversion)
REPOS = {
    "nllb-600M-ct2": ("facebook/nllb-200-distilled-600M", "f8d333a", True),
    "nllb-1.3B-ct2": ("OpenNMT/nllb-200-distilled-1.3B-ct2-int8", "70f572a", False),
    "nllb-3.3B-ct2": ("OpenNMT/nllb-200-3.3B-ct2-int8", "28d998c", False),
    "madlad-3B-ct2": ("Nextcloud-AI/madlad400-3b-mt-ct2-int8", "aa32bbd", False),
    "m2m100-418M-ct2": ("facebook/m2m100_418M", "55c2e61", True),
    "mbart50-ct2": ("facebook/mbart-large-50-many-to-many-mmt", "e30b6cb", True),
    "opus-mul-en-ct2": ("Helsinki-NLP/opus-mt-mul-en", "848eae0", True),
    "opus-en-mul-ct2": ("Helsinki-NLP/opus-mt-en-mul", "07ab277", True),
    "translategemma-4b": ("Infomaniak-AI/vllm-translategemma-4b-it", "cb3e0b2", False),
}


def fetch(name: str) -> Path:
    repo, rev, convert = REPOS[name]
    out = CACHE / name
    if not convert:  # snapshot_download resumes and skips complete files
        snapshot_download(repo, revision=rev, local_dir=out, ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*.onnx"])
        return out
    if (out / "model.bin").exists():
        return out
    import ctranslate2
    raw = CACHE / "_hf" / name
    snapshot_download(repo, revision=rev, local_dir=raw,
                      allow_patterns=["*.json", "*.model", "*.spm", "*.txt", "pytorch_model.bin", "model.safetensors"])
    keep = [f.name for f in raw.iterdir() if f.is_file() and f.name not in WEIGHTS and f.name != "config.json"]
    ctranslate2.converters.TransformersConverter(str(raw), copy_files=keep).convert(str(out), quantization="int8", force=True)
    shutil.rmtree(raw)
    return out


def flores() -> Path:
    return Path(hf_hub_download("mteb/flores", "devtest.parquet", repo_type="dataset", local_dir=CACHE / "flores"))


if __name__ == "__main__":
    flores()
    for n in sys.argv[1:] or REPOS:
        print(n, "->", fetch(n), flush=True)
