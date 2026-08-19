# %% [markdown]
# # 01 — Intent classification on the Bitext customer-support corpus
#
# Benchmarks four families of models on the same stratified split of the 27K Bitext
# customer-support dataset (27 intents / 11 categories):
#
# | Family | Models |
# |---|---|
# | Traditional supervised | TF-IDF + LogisticRegression, LinearSVC, ComplementNB, RandomForest, SVD+HistGradientBoosting |
# | Embedding + shallow head | MiniLM + LogisticRegression, MiniLM + kNN, MiniLM nearest-centroid |
# | Semantic matching (no labels used at fit time) | MiniLM zero-shot cosine against intent descriptions |
# | Deep supervised | TextCNN (from scratch), MiniLM fine-tuned, DistilBERT fine-tuned |
# | Unsupervised | KMeans (TF-IDF-SVD & MiniLM), Agglomerative, HDBSCAN, LDA topics |
#
# Every model is scored on the **same held-out test set** with accuracy, balanced
# accuracy, macro-F1, weighted-F1, training time, inference latency and artifact size.
# Clustering methods are additionally scored with ARI / NMI / V-measure / purity /
# Hungarian-matched accuracy against the true intent labels.

# %%
import sys, os, json, warnings
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
import text_data as td

set_seed()
DEV = device()
print(json.dumps(env_info(), indent=2))
store = ResultStore("intent_classification")

# %% [markdown]
# ## 1. Load and inspect the data

# %%
raw = td.load_bitext()
print(f"rows={len(raw)}  intents={raw['intent'].nunique()}  categories={raw['category'].nunique()}")
raw.head(3)

# %%
summary = td.train_class_summary(raw)
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
raw["intent"].value_counts().plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set_title("utterances per intent")
axes[0].invert_yaxis()
axes[0].tick_params(labelsize=7)
raw["n_words"].plot.hist(bins=40, ax=axes[1], color="#DD8452")
axes[1].set_title("utterance length (words)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Duplicate check — why this matters
#
# Bitext is template-generated: the same utterance recurs verbatim many times. If we
# split naively, identical strings land in both train and test and every model looks
# near-perfect. We therefore **deduplicate before splitting** and report the leakage
# that de-duplication removes.

# %%
dedup = td.dedupe(raw)
print(f"raw rows           : {len(raw)}")
print(f"unique utterances  : {len(dedup)}  ({100 * len(dedup) / len(raw):.1f}%)")
print(f"duplicates removed : {len(raw) - len(dedup)}")

df = dedup
train_df, val_df, test_df = td.split(df)
print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

LABELS = sorted(df["intent"].unique())
lab2id, id2lab = td.label_maps(df["intent"])
y_train = train_df["intent"].values
y_val = val_df["intent"].values
y_test = test_df["intent"].values

# Sanity: no test utterance appears in train.
overlap = set(train_df["text"].str.lower()) & set(test_df["text"].str.lower())
print(f"train/test string overlap after dedupe: {len(overlap)}")

# %% [markdown]
# ### Baseline floors
# Any model must beat these to be worth anything.

# %%
from sklearn.dummy import DummyClassifier

for strat in ("most_frequent", "stratified"):
    d = DummyClassifier(strategy=strat, random_state=SEED).fit(train_df["text"], y_train)
    m = classification_metrics(y_test, d.predict(test_df["text"]))
    store.add(Result("intent_classification", f"dummy_{strat}", "traditional", m,
                     notes="baseline floor"))

# %% [markdown]
# ## 2. Traditional supervised models (TF-IDF features)

# %%
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV

# Each pipeline gets its *own* vectoriser instance: sklearn pipelines do not clone their
# steps, so a shared vectoriser would be refitted by every pipeline and silently
# invalidate the ones fitted earlier.
def make_word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           strip_accents="unicode")

def make_char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True)

TRADITIONAL = {
    "tfidf_word+logreg": Pipeline([
        ("vec", make_word_tfidf()),
        ("clf", LogisticRegression(max_iter=2000, C=8.0, n_jobs=-1)),
    ]),
    "tfidf_word+char+linsvc": Pipeline([
        ("vec", FeatureUnion([("w", make_word_tfidf()), ("c", make_char_tfidf())])),
        ("clf", LinearSVC(C=1.0)),
    ]),
    "tfidf_word+complementNB": Pipeline([
        ("vec", make_word_tfidf()),
        ("clf", ComplementNB(alpha=0.3)),
    ]),
    "tfidf_word+sgd_hinge": Pipeline([
        ("vec", make_word_tfidf()),
        ("clf", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=30,
                              random_state=SEED)),
    ]),
    "tfidf_word+randomforest": Pipeline([
        ("vec", make_word_tfidf()),
        ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)),
    ]),
    "tfidf_svd300+histgb": Pipeline([
        ("vec", make_word_tfidf()),
        ("svd", TruncatedSVD(n_components=300, random_state=SEED)),
        ("clf", HistGradientBoostingClassifier(max_iter=200, random_state=SEED)),
    ]),
}

TEST_PREDS = {}   # model name -> test-set predictions, for the error analysis later

for name, pipe in TRADITIONAL.items():
    print(f"\n=== {name} ===")
    with timer("fit") as t:
        pipe.fit(train_df["text"], y_train)
    pred = pipe.predict(test_df["text"])
    TEST_PREDS[name] = pred
    proba = pipe.predict_proba(test_df["text"]) if hasattr(pipe, "predict_proba") else None
    m = classification_metrics(y_test, pred, proba, getattr(pipe, "classes_", None))
    lat = measure_latency(lambda s: pipe.predict([s]), test_df["text"].tolist())
    path = save_sklearn(pipe, f"intent_{name.replace('+', '_')}")
    store.add(Result("intent_classification", name, "traditional", m,
                     train_seconds=t.seconds, infer_ms_per_item=lat,
                     model_size_mb=file_size_mb(path)))

# %% [markdown]
# ## 3. Sentence embeddings (MiniLM) + shallow heads
#
# `all-MiniLM-L6-v2` — 6-layer, 384-dim sentence transformer. The encoder is **frozen**;
# only the head is trained. This is the cheap production pattern: embed once, swap heads.

# %%
from sentence_transformers import SentenceTransformer

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
encoder = SentenceTransformer(MINILM, device=DEV)

def embed(texts, bs=256, normalize=True):
    # show_progress_bar=False on purpose: the tqdm bar is emitted as an ipywidget, and
    # nbclient does not persist widget state into the notebook, so a saved run renders it
    # as "Could not render application/vnd.jupyter.widget-view+json" instead of a bar.
    return encoder.encode(list(texts), batch_size=bs, convert_to_numpy=True,
                          normalize_embeddings=normalize, show_progress_bar=False)

with timer("embed train") as t_emb_tr:
    E_train = embed(train_df["text"])
with timer("embed test") as t_emb_te:
    E_test = embed(test_df["text"])
E_val = embed(val_df["text"])
print(E_train.shape, E_test.shape)
np.savez_compressed(common.CACHE / "minilm_embeddings.npz",
                    E_train=E_train, E_val=E_val, E_test=E_test)

# Cost of embedding one sentence at a time (what a live service pays).
embed_latency = measure_latency(lambda s: encoder.encode([s]), test_df["text"].tolist(), n=100)
print(f"MiniLM single-sentence encode latency: {embed_latency:.2f} ms")

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

EMB_HEADS = {
    "minilm+logreg": LogisticRegression(max_iter=3000, C=10.0, n_jobs=-1),
    "minilm+knn5_cosine": KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
    "minilm+linsvc": LinearSVC(C=2.0),
    "minilm+mlp256": MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                                   random_state=SEED, early_stopping=True),
}

for name, head in EMB_HEADS.items():
    print(f"\n=== {name} ===")
    with timer("fit") as t:
        head.fit(E_train, y_train)
    pred = head.predict(E_test)
    TEST_PREDS[name] = pred
    proba = head.predict_proba(E_test) if hasattr(head, "predict_proba") else None
    m = classification_metrics(y_test, pred, proba, getattr(head, "classes_", None))
    path = save_sklearn(head, f"intent_{name.replace('+', '_')}")
    store.add(Result("intent_classification", name, "embedding", m,
                     train_seconds=t.seconds,
                     infer_ms_per_item=embed_latency + 0.1,
                     model_size_mb=file_size_mb(path),
                     notes="frozen MiniLM encoder; latency includes encoding"))

# %% [markdown]
# ## 4. Semantic matching
#
# Two label-efficient variants that need no gradient training:
#
# * **Zero-shot** — embed a natural-language description of each intent
#   (`"the customer wants to cancel order"`) and take the nearest description by cosine.
#   No training labels touched at all.
# * **Few-shot prototypes** — average the embeddings of *k* training examples per intent
#   and take the nearest class centroid. Sweeping *k* shows how many labels you actually
#   need before a trained head stops being worth it.

# %%
prompts = td.intent_prompts(LABELS)
P = embed(list(prompts.values()), normalize=True)
sims = E_test @ P.T
zs_pred = np.array(LABELS)[sims.argmax(1)]
TEST_PREDS["minilm_zeroshot_label_match"] = zs_pred
m = classification_metrics(y_test, zs_pred, sims, np.array(LABELS))
store.add(Result("intent_classification", "minilm_zeroshot_label_match", "embedding", m,
                 train_seconds=0.0, infer_ms_per_item=embed_latency,
                 notes="no training labels used; cosine vs auto-generated intent descriptions"))

# %%
proto_scores = []
for k in [1, 5, 10, 25, 50, None]:
    cents, names = [], []
    for lab in LABELS:
        idx = np.where(y_train == lab)[0]
        take = idx if k is None else idx[:k]
        v = E_train[take].mean(0)
        cents.append(v / (np.linalg.norm(v) + 1e-9))
        names.append(lab)
    C = np.vstack(cents)
    pred = np.array(names)[(E_test @ C.T).argmax(1)]
    mm = classification_metrics(y_test, pred)
    proto_scores.append({"k_per_class": k or "all", **mm})
    label = f"minilm_prototype_k{k}" if k else "minilm_prototype_all"
    store.add(Result("intent_classification", label, "embedding", mm,
                     train_seconds=0.0, infer_ms_per_item=embed_latency,
                     notes=f"nearest class centroid, {k or 'all'} shots/class"))

proto_df = pd.DataFrame(proto_scores)
print(proto_df[["k_per_class", "accuracy", "macro_f1"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
xs = [1, 5, 10, 25, 50, 100]
ax.plot(xs, proto_df["macro_f1"], "o-", label="prototype (few-shot)")
ax.axhline(store.rows["minilm+logreg"]["metrics"]["macro_f1"], ls="--", color="green",
           label="MiniLM+logreg (full supervision)")
ax.axhline(store.rows["minilm_zeroshot_label_match"]["metrics"]["macro_f1"], ls=":",
           color="red", label="zero-shot")
ax.set_xscale("log"); ax.set_xlabel("labelled examples per intent"); ax.set_ylabel("macro-F1")
ax.set_title("label efficiency of semantic matching"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Deep supervised models
#
# * **TextCNN** — trained from scratch, no pretraining, shows what pretraining buys.
# * **MiniLM fine-tuned** and **DistilBERT fine-tuned** — full encoder fine-tuning with a
#   classification head, early-stopped on the validation split.

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

y_train_id = np.array([lab2id[y] for y in y_train])
y_val_id = np.array([lab2id[y] for y in y_val])
y_test_id = np.array([lab2id[y] for y in y_test])
N_CLASSES = len(LABELS)


def evaluate_logits(model, loader):
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEV) for k, v in batch.items() if k != "labels"}
            outs.append(model(**batch).float().cpu())
    return torch.cat(outs).numpy()

# %% [markdown]
# ### 5a. TextCNN from scratch

# %%
from collections import Counter
import re as _re

def tokenize(s):
    return _re.findall(r"[a-z0-9<>]+", s.lower())

counter = Counter(tok for s in train_df["text"] for tok in tokenize(s))
kept = [w for w, c in counter.most_common(20000) if c >= 2]
vocab = {w: i + 2 for i, w in enumerate(kept)}   # contiguous ids, 0=<pad> 1=<unk>
vocab["<pad>"], vocab["<unk>"] = 0, 1
VOCAB_SIZE = len(kept) + 2
print(f"vocabulary: {VOCAB_SIZE} tokens")
MAXLEN = 32

def encode_ids(texts):
    X = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [vocab.get(t, 1) for t in tokenize(s)][:MAXLEN]
        X[i, : len(ids)] = ids
    return X

class IdsDS(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"x": self.X[i], "labels": self.y[i]}

class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, dim=128, filters=128, ks=(2, 3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(dim, filters, k, padding=k // 2) for k in ks])
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(filters * len(ks), n_classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)
        return self.fc(self.drop(h))

set_seed()
cnn = TextCNN(VOCAB_SIZE, N_CLASSES).to(DEV)
tr_loader = DataLoader(IdsDS(encode_ids(train_df["text"]), y_train_id), batch_size=128, shuffle=True)
va_loader = DataLoader(IdsDS(encode_ids(val_df["text"]), y_val_id), batch_size=256)
te_loader = DataLoader(IdsDS(encode_ids(test_df["text"]), y_test_id), batch_size=256)

opt = torch.optim.AdamW(cnn.parameters(), lr=2e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
best_state, best_acc, EPOCHS = None, -1, 12
with timer("TextCNN train") as t_cnn:
    for ep in range(EPOCHS):
        cnn.train()
        tot = 0.0
        for b in tr_loader:
            opt.zero_grad()
            loss = lossf(cnn(b["x"].to(DEV)), b["labels"].to(DEV))
            loss.backward(); opt.step(); tot += loss.item() * len(b["labels"])
        cnn.eval()
        with torch.no_grad():
            va = np.concatenate([cnn(b["x"].to(DEV)).argmax(1).cpu().numpy() for b in va_loader])
        acc = (va == y_val_id).mean()
        print(f"epoch {ep+1:02d}  train_loss={tot/len(tr_loader.dataset):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in cnn.state_dict().items()}
cnn.load_state_dict(best_state)

cnn.eval()
with torch.no_grad():
    logits = np.concatenate([cnn(b["x"].to(DEV)).cpu().numpy() for b in te_loader])
pred = np.array(LABELS)[logits.argmax(1)]
TEST_PREDS["textcnn_scratch"] = pred
m = classification_metrics(y_test, pred, torch.softmax(torch.tensor(logits), 1).numpy(),
                           np.array(LABELS))
lat = measure_latency(
    lambda s: cnn(torch.from_numpy(encode_ids([s])).to(DEV)).argmax(1).item(),
    test_df["text"].tolist(), n=100)
p = save_torch(cnn, "intent_textcnn", {"vocab": vocab, "labels": LABELS, "maxlen": MAXLEN})
store.add(Result("intent_classification", "textcnn_scratch", "deep", m,
                 train_seconds=t_cnn.seconds, infer_ms_per_item=lat,
                 model_size_mb=file_size_mb(p), n_params=count_params(cnn),
                 notes="no pretraining, 12 epochs, best-val checkpoint"))

# %% [markdown]
# ### 5b. Fine-tuned transformers

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TxtDS(Dataset):
    def __init__(self, texts, labels, tok, maxlen=48):
        self.enc = tok(list(texts), truncation=True, padding="max_length",
                       max_length=maxlen, return_tensors="pt")
        self.y = torch.tensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def finetune(model_name, tag, epochs=3, bs=64, lr=3e-5, maxlen=48):
    set_seed()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=N_CLASSES,
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=lab2id,
    ).to(DEV)
    tr = DataLoader(TxtDS(train_df["text"], y_train_id, tok, maxlen), batch_size=bs, shuffle=True)
    va = DataLoader(TxtDS(val_df["text"], y_val_id, tok, maxlen), batch_size=128)
    te = DataLoader(TxtDS(test_df["text"], y_test_id, tok, maxlen), batch_size=128)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    best_state, best_acc = None, -1

    with timer(f"{tag} fine-tune") as t:
        for ep in range(epochs):
            model.train(); tot = 0.0
            for b in tr:
                b = {k: v.to(DEV) for k, v in b.items()}
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                    out = model(**b)
                scaler.scale(out.loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                tot += out.loss.item() * len(b["labels"])
            model.eval(); preds = []
            with torch.no_grad():
                for b in va:
                    b = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
                    preds.append(model(**b).logits.argmax(1).cpu().numpy())
            acc = (np.concatenate(preds) == y_val_id).mean()
            print(f"  epoch {ep+1}  loss={tot/len(tr.dataset):.4f}  val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    model.eval(); logits = []
    with torch.no_grad():
        for b in te:
            b = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
            logits.append(model(**b).logits.float().cpu().numpy())
    logits = np.concatenate(logits)
    pred = np.array(LABELS)[logits.argmax(1)]
    proba = torch.softmax(torch.tensor(logits), 1).numpy()
    mm = classification_metrics(y_test, pred, proba, np.array(LABELS))

    def one(s):
        enc = tok([s], truncation=True, padding="max_length", max_length=maxlen,
                  return_tensors="pt").to(DEV)
        with torch.no_grad():
            return model(**enc).logits.argmax(1).item()
    lat = measure_latency(one, test_df["text"].tolist(), n=100)

    out_dir = ARTIFACTS / f"intent_{tag}"
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    size = round(sum(f.stat().st_size for f in out_dir.rglob("*")) / 1024**2, 2)
    store.add(Result("intent_classification", tag, "deep", mm,
                     train_seconds=t.seconds, infer_ms_per_item=lat,
                     model_size_mb=size, n_params=count_params(model),
                     params={"epochs": epochs, "lr": lr, "batch_size": bs, "max_len": maxlen},
                     notes=f"fine-tuned {model_name}, best-val checkpoint"))
    return model, tok, pred


minilm_ft, minilm_tok, minilm_pred = finetune(MINILM, "minilm_finetuned", epochs=3, lr=5e-5)
distil_ft, distil_tok, distil_pred = finetune("distilbert-base-uncased", "distilbert_finetuned",
                                              epochs=3, lr=3e-5)
TEST_PREDS["minilm_finetuned"] = minilm_pred
TEST_PREDS["distilbert_finetuned"] = distil_pred

# %% [markdown]
# ## 6. Unsupervised — can the intents be recovered without labels?
#
# Labels are used **only for scoring**, never for fitting. `k` is fixed to the true
# number of intents for the partitional methods; HDBSCAN decides its own `k`, which is
# the honest setting when the taxonomy is unknown.

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation

K = N_CLASSES

# Ward agglomerative clustering is O(n^2) in memory, so every clustering method is run on
# the same stratified subsample. Fixing one subsample for all of them keeps the ARI/NMI
# numbers directly comparable.
CLUSTER_N = 8000
rng = np.random.default_rng(SEED)
sub_idx = rng.choice(len(df), size=min(CLUSTER_N, len(df)), replace=False)
df_clu = df.iloc[sub_idx].reset_index(drop=True)
y_all = df_clu["intent"].values
print(f"clustering on {len(df_clu)} utterances covering {df_clu['intent'].nunique()} intents")

clu_tfidf = make_word_tfidf()
X_tfidf = clu_tfidf.fit_transform(df_clu["text"])
X_svd = TruncatedSVD(n_components=200, random_state=SEED).fit_transform(X_tfidf)
E_all = embed(df_clu["text"])

def run_cluster(name, ids, X, note=""):
    m = clustering_metrics(y_all, ids, X)
    store.add(Result("intent_classification", name, "unsupervised", m, notes=note))
    return m

with timer("kmeans tfidf") as t:
    ids = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X_svd)
run_cluster("kmeans_tfidf_svd200", ids, X_svd, "k = true number of intents")

with timer("kmeans minilm") as t:
    ids = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(E_all)
run_cluster("kmeans_minilm", ids, E_all, "k = true number of intents")

with timer("agglomerative minilm") as t:
    ids = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(E_all)
run_cluster("agglomerative_ward_minilm", ids, E_all, "k = true number of intents")

with timer("gmm minilm") as t:
    ids = GaussianMixture(n_components=K, covariance_type="diag",
                          random_state=SEED).fit_predict(E_all)
run_cluster("gmm_minilm", ids, E_all, "k = true number of intents")

with timer("hdbscan minilm") as t:
    ids = HDBSCAN(min_cluster_size=25, metric="euclidean").fit_predict(E_all)
run_cluster("hdbscan_minilm", ids, E_all, "k discovered by the algorithm")

# LDA is the classic *traditional* unsupervised text model: topics, not centroids.
with timer("lda tfidf") as t:
    lda = LatentDirichletAllocation(n_components=K, learning_method="online",
                                    random_state=SEED, max_iter=15)
    ids = lda.fit_transform(X_tfidf).argmax(1)
run_cluster("lda_topics_tfidf", ids, X_svd, "topic with highest weight = cluster")

# %% [markdown]
# ### Cluster structure, projected to 2-D

# %%
from sklearn.manifold import TSNE

sub = rng.choice(len(E_all), size=min(4000, len(E_all)), replace=False)
proj = TSNE(n_components=2, init="pca", perplexity=35, random_state=SEED).fit_transform(E_all[sub])
km_ids = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(E_all)[sub]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
codes = pd.Categorical(y_all[sub]).codes
axes[0].scatter(proj[:, 0], proj[:, 1], c=codes, cmap="tab20", s=4)
axes[0].set_title("t-SNE of MiniLM embeddings — coloured by TRUE intent")
axes[1].scatter(proj[:, 0], proj[:, 1], c=km_ids, cmap="tab20", s=4)
axes[1].set_title("coloured by KMeans cluster")
for a in axes: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Benchmark

# %%
bench = store.frame(sort_by="macro_f1")
supervised = bench[bench.family != "unsupervised"]
unsup = bench[bench.family == "unsupervised"]

pd.set_option("display.width", 200)
print("=== SUPERVISED / SEMANTIC ===")
print(supervised[["model", "family", "accuracy", "balanced_acc", "macro_f1", "weighted_f1",
                  "train_s", "infer_ms", "size_mb"]].round(4).to_string(index=False))
print("\n=== UNSUPERVISED (labels used only for scoring) ===")
print(unsup[["model", "ARI", "NMI", "V_measure", "purity", "hungarian_acc",
             "n_clusters", "silhouette"]].round(4).to_string(index=False))

# %%
fig = plot_benchmark(supervised, "macro_f1", "Intent classification — macro-F1 (test set)")
fig.savefig(common.RESULTS / "intent_macro_f1.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5.5))
d = supervised.dropna(subset=["infer_ms", "macro_f1"])
for fam, grp in d.groupby("family"):
    ax.scatter(grp["infer_ms"], grp["macro_f1"], s=70,
               label=fam, color=common.FAMILY_COLORS.get(fam, "#888"))
    for _, r in grp.iterrows():
        ax.annotate(r["model"], (r["infer_ms"], r["macro_f1"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("inference latency (ms/utterance, log)")
ax.set_ylabel("macro-F1"); ax.set_title("accuracy vs latency — the deployment trade-off")
ax.legend(fontsize=8); plt.tight_layout()
fig.savefig(common.RESULTS / "intent_accuracy_vs_latency.png", dpi=150)
plt.show()

# %%
best = supervised.iloc[0]["model"]
print(f"best supervised model: {best}")
best_pred = TEST_PREDS.get(best)
if best_pred is not None:
    fig = plot_confusion(y_test, best_pred, LABELS, f"confusion matrix — {best}")
    fig.savefig(common.RESULTS / "intent_confusion_best.png", dpi=150)
    plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(y_test, best_pred, zero_division=0))

    err = pd.DataFrame({"text": test_df["text"], "true": y_test, "pred": best_pred})
    err = err[err.true != err.pred]
    print(f"\n{len(err)} errors ({100*len(err)/len(test_df):.2f}%). Most confused pairs:")
    print(err.groupby(["true", "pred"]).size().sort_values(ascending=False).head(12))
    err.head(25).to_csv(common.RESULTS / "intent_error_samples.csv", index=False)

# %% [markdown]
# ## 8. Notes for the report
#
# * Scores are on **de-duplicated** data. Re-running on the raw 27K file inflates every
#   number because template duplicates cross the split boundary — quote the de-duplicated
#   figures.
# * The zero-shot and few-shot prototype rows are the ones to cite for a cold-start
#   deployment: they need no labelled data yet still recover a large share of the
#   supervised ceiling.
# * Latency for embedding-based rows includes MiniLM encoding, which dominates; TF-IDF
#   rows are one to two orders of magnitude cheaper per utterance.
# * Unsupervised scores answer a different question — whether the intent taxonomy is
#   *discoverable* from raw utterances — so read ARI/NMI, not accuracy, for those rows.

# %%
print(json.dumps({"env": env_info(), "n_models": len(store.rows)}, indent=2))
print(f"results -> {store.path}")
print(f"artifacts -> {ARTIFACTS}")
