# %% [markdown]
# # 11 — Router crops: whole-model size and speed for the frozen CNNs
#
# Notebook 02 reported the frozen-CNN rows with the size of the LogReg head only
# (0.04–0.09 MB) and no latency, because the backbone was run once for the whole split.
# In service the backbone runs on every photo, so the fair cost is **backbone + head**.
# This notebook loads the saved notebook-02 artifacts (nothing is retrained), checks they
# reproduce notebook 02's test scores, then measures, with notebook 02's protocol:
#
# * **size**: fp32 backbone weights (classifier removed) + the saved head, as files on disk;
# * **latency**: one crop end to end (resize, backbone, head), median of 60 after 10 warm-up,
#   on the same GPU as notebook 02 and on the laptop CPU. The hand-crafted rows are re-timed on CPU
#   with feature extraction included (section 2b).
#
# The three fine-tuned CNNs are re-timed in the same session so every row is comparable.
# A *crop* is one router-component image cut out of a photo by its labelled bounding box
# (an LED panel, a port, a button); every model classifies one crop at a time.

# %%
import sys, json, warnings, tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from common import set_seed, device, env_info, measure_latency, classification_metrics, file_size_mb, ARTIFACTS, RESULTS
import vision_data as vd

set_seed()
DEV = device()
print(json.dumps(env_info(), indent=2))

per_split, _ = vd.load_all(min_class_count=25)
cache = vd.build_crop_cache(per_split, size=96)
Xte, yte = cache["test"]
CLASSES = sorted(set(cache["train"][1]) | set(cache["valid"][1]) | set(yte))
print(f"test crops: {len(Xte)}, classes: {len(CLASSES)}")

SAVED = {r["model"]: r["metrics"] for r in json.loads((RESULTS / "router_image_classification.json").read_text())}

# %% [markdown]
# ## 1. Models, exactly as notebook 02 built them

# %%
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True),
                              transforms.Normalize(MEAN, STD)])
BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build(key, num_classes=None):
    fn, weights = BACKBONES[key]
    model = fn(weights=weights)
    if key == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes) if num_classes else nn.Identity()
    else:
        model.classifier[-1] = (nn.Linear(model.classifier[-1].in_features, num_classes)
                                if num_classes else nn.Identity())
    return model.eval()


def frozen(key, dev):
    backbone, head = build(key).to(dev), joblib.load(ARTIFACTS / f"router_{key}_frozen_logreg.joblib")

    @torch.no_grad()
    def one(img):
        feat = backbone(eval_tf(img).unsqueeze(0).to(dev)).float().cpu().numpy()
        return head.predict(feat)[0]
    return backbone, head, one


def finetuned(key, dev):
    ck = torch.load(ARTIFACTS / f"router_{key}_finetuned.pt", map_location="cpu", weights_only=False)
    model = build(key, len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    model = model.eval().to(dev)
    classes = ck["classes"]

    @torch.no_grad()
    def one(img):
        return classes[model(eval_tf(img).unsqueeze(0).to(dev)).argmax(1).item()]
    return model, one


def disk_mb(state_dict):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "w.pt"
        torch.save(state_dict, p)
        return file_size_mb(p)

# %% [markdown]
# ## 2. Check, score and time every CNN row

# %%
rows = []
for key in BACKBONES:
    backbone, head, one = frozen(key, DEV)
    pred = np.array([one(x) for x in Xte])
    m = classification_metrics(yte, pred)
    saved = SAVED[f"{key}_frozen+logreg"]
    head_mb = file_size_mb(ARTIFACTS / f"router_{key}_frozen_logreg.joblib")
    back_mb = disk_mb(backbone.state_dict())
    gpu_ms = measure_latency(one, list(Xte), n=60)
    _, _, one_cpu = frozen(key, "cpu")
    cpu_ms = measure_latency(one_cpu, list(Xte), n=60)
    rows.append({"model": f"{key}_frozen+logreg", "family": "frozen CNN",
                 "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                 "macro_f1_nb02": saved["macro_f1"],
                 "head_mb": head_mb, "backbone_mb": back_mb, "whole_mb": round(back_mb + head_mb, 2),
                 "gpu_ms": gpu_ms, "cpu_ms": cpu_ms,
                 "n_params": int(sum(p.numel() for p in backbone.parameters())) + int(head[-1].coef_.size + head[-1].intercept_.size)})

    model, one = finetuned(key, DEV)
    pred = np.array([one(x) for x in Xte])
    m = classification_metrics(yte, pred)
    saved = SAVED[f"{key}_finetuned"]
    gpu_ms = measure_latency(one, list(Xte), n=60)
    _, one_cpu = finetuned(key, "cpu")
    cpu_ms = measure_latency(one_cpu, list(Xte), n=60)
    rows.append({"model": f"{key}_finetuned", "family": "fine-tuned",
                 "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                 "macro_f1_nb02": saved["macro_f1"],
                 "head_mb": None, "backbone_mb": None,
                 "whole_mb": file_size_mb(ARTIFACTS / f"router_{key}_finetuned.pt"),
                 "gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "n_params": int(sum(p.numel() for p in model.parameters()))})
    del backbone, model
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
assert (df["macro_f1"] - df["macro_f1_nb02"]).abs().max() < 0.002, "artifacts do not reproduce notebook 02"
print("all six artifacts reproduce notebook 02's test macro-F1")
df.round({"macro_f1": 4, "accuracy": 4, "gpu_ms": 1, "cpu_ms": 1}).sort_values("macro_f1", ascending=False)

# %% [markdown]
# ## 2b. The hand-crafted rows, end to end
# Notebook 02 timed these on precomputed features. Here feature extraction is included, on CPU,
# so every row on the slide answers the same question: one crop in, one label out.

# %%
from skimage.feature import hog
from skimage.color import rgb2gray
from PIL import Image


def hog_f(im):
    return hog(rgb2gray(im), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


def colour_f(im, bins=16):
    f = [np.histogram(im[..., c], bins=bins, range=(0, 255), density=True)[0] for c in range(3)]
    arr = im.reshape(-1, 3).astype(np.float32)
    return np.concatenate(f + [arr.mean(0) / 255, arr.std(0) / 255])


def pixel_f(im, size=32):
    return np.asarray(Image.fromarray(im).resize((size, size))).ravel() / 255.0


FEAT = {"hog+linsvc": hog_f, "hog+logreg": hog_f, "colourhist+randomforest": colour_f,
        "pixels32+pca128+logreg": pixel_f, "hog+colour+pca200+histgb": lambda im: np.concatenate([hog_f(im), colour_f(im)])}
for name, fn in FEAT.items():
    path = ARTIFACTS / f"router_{name.replace('+', '_')}.joblib"
    clf = joblib.load(path)
    if hasattr(clf, "n_jobs"):
        clf.n_jobs = 1  # one crop at a time; a process pool per call would time the pool, not the model
    one = lambda im, clf=clf, fn=fn: clf.predict(np.asarray(fn(im), dtype=np.float32)[None])[0]
    pred = np.array([one(x) for x in Xte])
    m = classification_metrics(yte, pred)
    rows.append({"model": name, "family": "classic", "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                 "macro_f1_nb02": SAVED[name]["macro_f1"], "head_mb": None, "backbone_mb": None,
                 "whole_mb": file_size_mb(path), "gpu_ms": None, "cpu_ms": measure_latency(one, list(Xte), n=60),
                 "n_params": None})

df = pd.DataFrame(rows)
assert (df["macro_f1"] - df["macro_f1_nb02"]).abs().max() < 0.002, "artifacts do not reproduce notebook 02"
df.round({"macro_f1": 4, "accuracy": 4, "gpu_ms": 1, "cpu_ms": 2}).sort_values("macro_f1", ascending=False)

# %% [markdown]
# ## 3. How close is the top of the table?
# The test split is small, so count crops, not only F1.

# %%
n = len(yte)
for _, r in df.sort_values("macro_f1", ascending=False).head(3).iterrows():
    print(f"{r['model']:32s} correct {round(r['accuracy'] * n)} of {n}  macro-F1 {r['macro_f1']:.4f}")

# %%
out = RESULTS / "router_full_cost.json"
out.write_text(json.dumps({"device": DEV, "n_test": n, "protocol": "one crop end to end, median of 60 after 10 warm-up",
                           "rows": rows}, indent=1))
df.to_csv(RESULTS / "summary_router_full_cost.csv", index=False)
print("wrote", out)
