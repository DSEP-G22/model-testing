"""Router-detection COCO dataset -> object-crop classification dataset.

The Roboflow export is a *detection* dataset (bounding boxes over routers, cables
and connectors). To benchmark classifiers we turn every annotated box into a
cropped image with its category as the label, which is the standard way to reuse a
detection corpus for a classification study.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import CACHE, ROUTER_DIR

SPLITS = ("train", "valid", "test")
MIN_BOX_PX = 24  # drop boxes whose shorter side is below this: unusable after resize


def load_split(split: str) -> tuple[list[dict], dict]:
    """Return (records, categories) for one COCO split.

    Each record: {path, category_id, name, bbox (xywh), area, image_wh}.
    """
    ann_path = ROUTER_DIR / split / "_annotations.coco.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    recs = []
    for a in data["annotations"]:
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        if min(w, h) < MIN_BOX_PX:
            continue
        recs.append(
            {
                "path": str(ROUTER_DIR / split / im["file_name"]),
                "category_id": a["category_id"],
                "name": cats[a["category_id"]],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "image_wh": (im["width"], im["height"]),
                "split": split,
            }
        )
    return recs, cats


def load_all(min_class_count: int = 25) -> tuple[dict, dict]:
    """Load every split, dropping classes too rare to evaluate meaningfully."""
    per_split, cats = {}, {}
    for s in SPLITS:
        per_split[s], cats = load_split(s)

    counts: dict[str, int] = {}
    for recs in per_split.values():
        for r in recs:
            counts[r["name"]] = counts.get(r["name"], 0) + 1
    keep = {n for n, c in counts.items() if c >= min_class_count}
    dropped = {n: c for n, c in counts.items() if n not in keep}
    if dropped:
        print(f"dropping rare classes (<{min_class_count} boxes): {dropped}")
    for s in SPLITS:
        per_split[s] = [r for r in per_split[s] if r["name"] in keep]
    return per_split, cats


def crop_array(rec: dict, size: int = 96, pad: float = 0.08) -> np.ndarray:
    """Crop the annotated box (with a small context margin) to a square RGB array."""
    from PIL import Image

    x, y, w, h = rec["bbox"]
    px, py = w * pad, h * pad
    with Image.open(rec["path"]) as im:
        im = im.convert("RGB")
        W, H = im.size
        box = (
            max(0, int(x - px)),
            max(0, int(y - py)),
            min(W, int(x + w + px)),
            min(H, int(y + h + py)),
        )
        crop = im.crop(box).resize((size, size), Image.BILINEAR)
        return np.asarray(crop, dtype=np.uint8)


def build_crop_cache(per_split: dict, size: int = 96, force: bool = False) -> dict:
    """Materialise all crops once into .npz caches; later runs just memory-map them."""
    out = {}
    for split, recs in per_split.items():
        path = CACHE / f"router_crops_{split}_{size}.npz"
        if path.exists() and not force:
            z = np.load(path, allow_pickle=True)
            out[split] = (z["X"], z["y"])
            continue
        X = np.zeros((len(recs), size, size, 3), dtype=np.uint8)
        y = []
        for i, r in enumerate(recs):
            X[i] = crop_array(r, size=size)
            y.append(r["name"])
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{len(recs)} crops")
        y = np.array(y)
        np.savez_compressed(path, X=X, y=y)
        out[split] = (X, y)
        print(f"cached {split}: {X.shape} -> {path.name}")
    return out
