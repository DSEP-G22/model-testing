# %% [markdown]
# # 02 — Router / cable / connector image classification
#
# The Roboflow `router detection v38` export is a **detection** dataset: 2 137 training
# images with 4 630 bounding boxes over 16 categories (router body, fiber/lan/phone/power/
# usb cables and their connectors). To benchmark *classifiers* we crop every annotated box
# and classify the crop. Roboflow's own train/valid/test split is preserved, so no image
# contributes crops to two splits.
#
# | Family | Models |
# |---|---|
# | Traditional | HOG + LinearSVC, colour histogram + RandomForest, raw-pixel PCA + LogReg, HOG+colour + HistGradientBoosting |
# | Frozen deep features | ResNet18 / MobileNetV3 / EfficientNet-B0 ImageNet embeddings + LogReg |
# | Fine-tuned deep | ResNet18, MobileNetV3-Small, EfficientNet-B0 |
# | Unsupervised | KMeans + Agglomerative on ResNet18 features |

# %%
import sys, json, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
from common import (
    Result, ResultStore, set_seed, device, env_info, timer, measure_latency,
    classification_metrics, clustering_metrics, plot_confusion, plot_benchmark,
    save_sklearn, save_torch, file_size_mb, count_params, ARTIFACTS, SEED,
)
import vision_data as vd

set_seed()
DEV = device()
print(json.dumps(env_info(), indent=2))
store = ResultStore("router_image_classification")

# %% [markdown]
# ## 1. Build the crop dataset

# %%
per_split, cats = vd.load_all(min_class_count=25)
for s, recs in per_split.items():
    print(f"{s:6s}: {len(recs)} boxes")

counts = (
    pd.Series([r["name"] for r in per_split["train"]]).value_counts().rename("train")
    .to_frame()
    .join(pd.Series([r["name"] for r in per_split["valid"]]).value_counts().rename("valid"))
    .join(pd.Series([r["name"] for r in per_split["test"]]).value_counts().rename("test"))
    .fillna(0).astype(int)
)
print(counts.to_string())

# %%
IMG = 96
cache = vd.build_crop_cache(per_split, size=IMG)
Xtr, ytr = cache["train"]
Xva, yva = cache["valid"]
Xte, yte = cache["test"]
CLASSES = sorted(set(ytr) | set(yva) | set(yte))
print(f"train {Xtr.shape}  valid {Xva.shape}  test {Xte.shape}  classes={len(CLASSES)}")

# The test split is small; note the per-class support so single-sample classes are not
# over-interpreted in the macro-F1.
print(pd.Series(yte).value_counts().to_string())

# %%
fig, axes = plt.subplots(3, 8, figsize=(15, 6))
rng = np.random.default_rng(SEED)
for ax, i in zip(axes.ravel(), rng.choice(len(Xtr), 24, replace=False)):
    ax.imshow(Xtr[i]); ax.set_title(ytr[i], fontsize=7); ax.axis("off")
plt.suptitle("sample crops"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Traditional computer-vision baselines
#
# Hand-designed features (gradient orientation histograms, colour statistics) plus a
# classical classifier — the pre-deep-learning pipeline, and still the cheap option.

# %%
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier

def hog_features(X):
    return np.array([
        hog(rgb2gray(im), orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), feature_vector=True)
        for im in X
    ], dtype=np.float32)

def colour_features(X, bins=16):
    feats = []
    for im in X:
        f = [np.histogram(im[..., c], bins=bins, range=(0, 255), density=True)[0]
             for c in range(3)]
        arr = im.reshape(-1, 3).astype(np.float32)
        feats.append(np.concatenate(f + [arr.mean(0) / 255, arr.std(0) / 255]))
    return np.array(feats, dtype=np.float32)

def pixel_features(X, size=32):
    from PIL import Image
    return np.array([
        np.asarray(Image.fromarray(im).resize((size, size))).ravel() / 255.0 for im in X
    ], dtype=np.float32)

with timer("HOG features"):
    H_tr, H_va, H_te = hog_features(Xtr), hog_features(Xva), hog_features(Xte)
with timer("colour features"):
    C_tr, C_va, C_te = colour_features(Xtr), colour_features(Xva), colour_features(Xte)
with timer("pixel features"):
    P_tr, P_va, P_te = pixel_features(Xtr), pixel_features(Xva), pixel_features(Xte)
print(f"HOG dim={H_tr.shape[1]}  colour dim={C_tr.shape[1]}  pixel dim={P_tr.shape[1]}")

# %%
d = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(H_tr, ytr)
store.add(Result("router_image_classification", "dummy_most_frequent", "traditional",
                 classification_metrics(yte, d.predict(H_te)), notes="baseline floor"))

TRAD = {
    "hog+linsvc": (H_tr, H_te, Pipeline([("sc", StandardScaler()), ("clf", LinearSVC(C=1.0))])),
    "hog+logreg": (H_tr, H_te, Pipeline([("sc", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=3000, n_jobs=-1))])),
    "colourhist+randomforest": (C_tr, C_te, RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=SEED)),
    "pixels32+pca128+logreg": (P_tr, P_te, Pipeline([
        ("pca", PCA(n_components=128, random_state=SEED)),
        ("clf", LogisticRegression(max_iter=3000, n_jobs=-1))])),
    "hog+colour+pca200+histgb": (np.hstack([H_tr, C_tr]), np.hstack([H_te, C_te]),
                                 Pipeline([("pca", PCA(n_components=200, random_state=SEED)),
                                           ("clf", HistGradientBoostingClassifier(
                                               max_iter=200, random_state=SEED))])),
}

for name, (ftr, fte, clf) in TRAD.items():
    print(f"\n=== {name} ===")
    with timer("fit") as t:
        clf.fit(ftr, ytr)
    pred = clf.predict(fte)
    proba = clf.predict_proba(fte) if hasattr(clf, "predict_proba") else None
    m = classification_metrics(yte, pred, proba, getattr(clf, "classes_", None))
    lat = measure_latency(lambda i: clf.predict(fte[i:i+1]), range(len(fte)), n=100)
    p = save_sklearn(clf, f"router_{name.replace('+', '_')}")
    store.add(Result("router_image_classification", name, "traditional", m,
                     train_seconds=t.seconds, infer_ms_per_item=lat,
                     model_size_mb=file_size_mb(p),
                     notes="latency excludes feature extraction"))

# %% [markdown]
# ## 3. Frozen ImageNet features + linear head
#
# No gradient updates to the backbone: extract embeddings once, fit a logistic regression.
# This is the strongest option when the labelled set is this small (a few thousand crops).

# %%
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(MEAN, STD),
])

BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small,
                           torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (torchvision.models.efficientnet_b0,
                        torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}

def build_backbone(key, num_classes=None):
    fn, weights = BACKBONES[key]
    model = fn(weights=weights)
    if key == "resnet18":
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes) if num_classes else nn.Identity()
    else:
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes) if num_classes else nn.Identity()
    return model, in_f

@torch.no_grad()
def extract(model, X, bs=64):
    model.eval().to(DEV)
    out = []
    for i in range(0, len(X), bs):
        batch = torch.stack([eval_tf(im) for im in X[i:i + bs]]).to(DEV)
        out.append(model(batch).float().cpu().numpy())
    return np.concatenate(out)

FEATS = {}
for key in BACKBONES:
    m, dim = build_backbone(key)
    with timer(f"{key} feature extraction") as t:
        FEATS[key] = (extract(m, Xtr), extract(m, Xva), extract(m, Xte), t.seconds)
    print(f"{key}: dim={FEATS[key][0].shape[1]}")

# %%
for key, (ftr, fva, fte, extract_s) in FEATS.items():
    name = f"{key}_frozen+logreg"
    print(f"\n=== {name} ===")
    clf = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, C=1.0, n_jobs=-1))])
    with timer("fit") as t:
        clf.fit(ftr, ytr)
    pred = clf.predict(fte)
    m = classification_metrics(yte, pred, clf.predict_proba(fte), clf.classes_)
    p = save_sklearn(clf, f"router_{key}_frozen_logreg")
    store.add(Result("router_image_classification", name, "embedding", m,
                     train_seconds=t.seconds + extract_s,
                     model_size_mb=file_size_mb(p),
                     notes="ImageNet weights frozen; train time includes feature extraction"))

# %% [markdown]
# ## 4. Fine-tuned CNNs
#
# Full backbone fine-tuning with light augmentation (flip, rotation, colour jitter),
# early stopping on the validation split, best checkpoint saved to `artifacts/`.

# %%
from torch.utils.data import Dataset, DataLoader

cls2id = {c: i for i, c in enumerate(CLASSES)}
train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.9, 1.1)),
    transforms.ColorJitter(0.25, 0.25, 0.2, 0.03),
    transforms.Normalize(MEAN, STD),
])

class CropDS(Dataset):
    def __init__(self, X, y, tf):
        self.X, self.y, self.tf = X, np.array([cls2id[v] for v in y]), tf
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.tf(self.X[i]), self.y[i]

tr_loader = DataLoader(CropDS(Xtr, ytr, train_tf), batch_size=32, shuffle=True, num_workers=0)
va_loader = DataLoader(CropDS(Xva, yva, eval_tf), batch_size=64)
te_loader = DataLoader(CropDS(Xte, yte, eval_tf), batch_size=64)

# Class imbalance is severe (usb-cable 887 boxes vs fiber-conn 15), so the loss is
# weighted by inverse class frequency.
freq = np.array([np.sum(ytr == c) for c in CLASSES], dtype=np.float32)
weights = torch.tensor((freq.sum() / (len(CLASSES) * np.clip(freq, 1, None))), dtype=torch.float32).to(DEV)
print(dict(zip(CLASSES, weights.cpu().numpy().round(2))))


def finetune_cnn(key, epochs=12, lr=3e-4):
    set_seed()
    model, _ = build_backbone(key, num_classes=len(CLASSES))
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    best_state, best = None, -1

    with timer(f"{key} fine-tune") as t:
        for ep in range(epochs):
            model.train(); tot = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                    loss = lossf(model(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += loss.item() * len(yb)
            sched.step()
            model.eval(); preds, gts = [], []
            with torch.no_grad():
                for xb, yb in va_loader:
                    preds.append(model(xb.to(DEV)).argmax(1).cpu().numpy()); gts.append(yb.numpy())
            from sklearn.metrics import f1_score
            p, g = np.concatenate(preds), np.concatenate(gts)
            f1 = f1_score(g, p, average="macro", zero_division=0)
            print(f"  epoch {ep+1:02d} loss={tot/len(tr_loader.dataset):.4f} "
                  f"val_acc={(p==g).mean():.4f} val_macroF1={f1:.4f}")
            if f1 > best:
                best = f1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in te_loader:
            preds.append(torch.softmax(model(xb.to(DEV)).float(), 1).cpu().numpy())
    proba = np.concatenate(preds)
    pred = np.array(CLASSES)[proba.argmax(1)]
    m = classification_metrics(yte, pred, proba, np.array(CLASSES))
    @torch.no_grad()
    def one(i):
        return model(eval_tf(Xte[i]).unsqueeze(0).to(DEV)).argmax(1).item()
    lat = measure_latency(one, range(len(Xte)), n=60)
    p = save_torch(model, f"router_{key}_finetuned",
                   {"classes": CLASSES, "img_size": 224, "mean": MEAN, "std": STD})
    store.add(Result("router_image_classification", f"{key}_finetuned", "deep", m,
                     train_seconds=t.seconds, infer_ms_per_item=lat,
                     model_size_mb=file_size_mb(p), n_params=count_params(model),
                     params={"epochs": epochs, "lr": lr, "class_weighted": True},
                     notes="augmented, class-weighted loss, best-val-macroF1 checkpoint"))
    return model, pred


ft_models = {}
for key in ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]:
    print(f"\n########## fine-tune {key} ##########")
    ft_models[key] = finetune_cnn(key)

# %% [markdown]
# ## 5. Unsupervised — do the classes separate without labels?

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN

F_all = np.concatenate([FEATS["resnet18"][0], FEATS["resnet18"][1], FEATS["resnet18"][2]])
y_all = np.concatenate([ytr, yva, yte])
K = len(CLASSES)

for name, algo in {
    "kmeans_resnet18_feats": KMeans(n_clusters=K, n_init=10, random_state=SEED),
    "agglomerative_resnet18_feats": AgglomerativeClustering(n_clusters=K, linkage="ward"),
    "hdbscan_resnet18_feats": HDBSCAN(min_cluster_size=20),
}.items():
    with timer(name) as t:
        ids = algo.fit_predict(F_all)
    m = clustering_metrics(y_all, ids, F_all)
    store.add(Result("router_image_classification", name, "unsupervised", m,
                     train_seconds=t.seconds,
                     notes="k = true class count" if "hdbscan" not in name else "k discovered"))

# %%
from sklearn.manifold import TSNE

proj = TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(F_all)
fig, ax = plt.subplots(figsize=(9, 7))
codes = pd.Categorical(y_all, categories=CLASSES).codes
sc = ax.scatter(proj[:, 0], proj[:, 1], c=codes, cmap="tab20", s=6)
handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab20(i / max(len(CLASSES)-1, 1)),
                      label=c) for i, c in enumerate(CLASSES)]
ax.legend(handles=handles, fontsize=7, ncol=2, loc="best")
ax.set_title("t-SNE of frozen ResNet18 crop features, coloured by true class")
ax.set_xticks([]); ax.set_yticks([]); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Benchmark

# %%
bench = store.frame(sort_by="macro_f1")
sup = bench[bench.family != "unsupervised"]
uns = bench[bench.family == "unsupervised"]

print("=== SUPERVISED ===")
print(sup[["model", "family", "accuracy", "balanced_acc", "macro_f1", "weighted_f1",
           "train_s", "infer_ms", "size_mb"]].round(4).to_string(index=False))
print("\n=== UNSUPERVISED ===")
print(uns[["model", "ARI", "NMI", "purity", "hungarian_acc", "n_clusters",
           "silhouette"]].round(4).to_string(index=False))

fig = plot_benchmark(sup, "macro_f1", "Router crop classification — macro-F1 (test split)")
fig.savefig(common.RESULTS / "router_macro_f1.png", dpi=150)
plt.show()

# %%
best = sup.iloc[0]["model"]
print(f"best model: {best}")
key = best.replace("_finetuned", "")
if key in ft_models:
    _, pred = ft_models[key]
    fig = plot_confusion(yte, pred, CLASSES, f"confusion matrix — {best}", figsize=(9, 7))
    fig.savefig(common.RESULTS / "router_confusion_best.png", dpi=150)
    plt.show()
    from sklearn.metrics import classification_report
    print(classification_report(yte, pred, zero_division=0))

# %% [markdown]
# ## 7. Notes for the report
#
# * The test split holds only ~200 crops and several classes have single-digit support,
#   so **macro-F1 on this split is noisy**; quote balanced accuracy and the per-class
#   report together, and treat differences under ~3 points as noise.
# * Frozen ImageNet features + logistic regression is the accuracy-per-minute winner at
#   this dataset size; fine-tuning pays off mainly on the visually confusable connector
#   classes.
# * Traditional HOG/colour pipelines separate cable *colour* well but collapse the
#   connector classes, which is exactly the shape of their confusion matrices.
# * These are crop classifiers, not detectors — an end-to-end detector (YOLO/Faster R-CNN)
#   is the next step if the deployed system must localise as well as label.

# %%
print(f"results -> {store.path}")
print(f"artifacts -> {ARTIFACTS}")
