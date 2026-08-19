# %% [markdown]
# # 05 — Intent classification under distribution shift
#
# Notebook 01 shows every supervised model scoring 0.99+ macro-F1 on a random split. That
# is a property of the corpus, not of the models: Bitext is template-generated, so a random
# split puts near-identical phrasings on both sides and the ranking collapses into noise.
#
# This notebook re-runs the comparison on a split that actually separates the models. Bitext
# tags every utterance with linguistic *flags*:
#
# | flag | meaning |
# |---|---|
# | `B` basic syntax, `L` semantic variation, `I` interrogative | "clean" phrasings |
# | `Q` colloquial, `Z` noise (typos, missing punctuation), `W` offensive, `P` politeness | "hard" phrasings |
#
# **The shifted split trains only on clean utterances and tests only on noisy/colloquial
# ones** — the call-centre reality, where the model is trained on tidy text and then meets
# ASR output full of disfluencies. Scores here are the ones worth quoting for deployment.

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
    Result, ResultStore, set_seed, device, env_info, timer, classification_metrics,
    plot_confusion, plot_benchmark, save_sklearn, ARTIFACTS, SEED,
)
import text_data as td

set_seed()
DEV = device()
store = ResultStore("intent_robustness_shift")
print(json.dumps(env_info(), indent=2))

# %% [markdown]
# ## 1. Build the shifted split

# %%
HARD_FLAGS = set("QZWP")     # colloquial / noisy / offensive / over-polite
df = td.dedupe(td.load_bitext())
df["is_hard"] = df["flags"].apply(lambda f: bool(set(f) & HARD_FLAGS))

clean = df[~df.is_hard].reset_index(drop=True)
hard = df[df.is_hard].reset_index(drop=True)
print(f"clean utterances: {len(clean)}   hard utterances: {len(hard)}")

# Keep only intents present on both sides, otherwise the comparison is not well posed.
shared = sorted(set(clean["intent"]) & set(hard["intent"]))
clean = clean[clean.intent.isin(shared)].reset_index(drop=True)
hard = hard[hard.intent.isin(shared)].reset_index(drop=True)
LABELS = shared
print(f"intents kept: {len(LABELS)}")

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(clean, test_size=0.12, stratify=clean["intent"],
                                    random_state=SEED)
test_df = hard
y_train, y_val, y_test = train_df["intent"].values, val_df["intent"].values, test_df["intent"].values
print(f"train(clean)={len(train_df)}  val(clean)={len(val_df)}  test(HARD)={len(test_df)}")

# For reference, the same models are also scored on a held-out *clean* test set, so the
# gap between the two columns is the cost of the shift.
clean_train, clean_test = train_test_split(train_df, test_size=0.15,
                                           stratify=train_df["intent"], random_state=SEED)
print(f"in-distribution control: train={len(clean_train)}  test={len(clean_test)}")

# %%
fig, ax = plt.subplots(figsize=(11, 4))
pd.crosstab(df["intent"], df["is_hard"], normalize="index").plot.bar(stacked=True, ax=ax,
                                                                    color=["#4C72B0", "#C44E52"])
ax.set_ylabel("share of utterances"); ax.set_title("clean vs hard phrasings per intent")
ax.legend(["clean", "hard (Q/Z/W/P)"], fontsize=8); ax.tick_params(labelsize=7)
plt.tight_layout(); plt.show()

print("--- example hard utterances ---")
for t in test_df.sample(8, random_state=SEED)["text"]:
    print("  ", t)

# %% [markdown]
# ## 2. Same model families, shifted evaluation

# %%
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           strip_accents="unicode")

def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)

PIPES = {
    "tfidf_word+logreg": Pipeline([("vec", word_tfidf()),
                                   ("clf", LogisticRegression(max_iter=2000, C=8.0, n_jobs=-1))]),
    "tfidf_word+char+linsvc": Pipeline([
        ("vec", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
        ("clf", LinearSVC(C=1.0))]),
}

CONTROL = {}   # model -> in-distribution macro-F1, for the gap column

for name, pipe in PIPES.items():
    print(f"\n=== {name} ===")
    with timer("fit") as t:
        pipe.fit(train_df["text"], y_train)
    m = classification_metrics(y_test, pipe.predict(test_df["text"]))
    ctl = Pipeline(pipe.steps).fit(clean_train["text"], clean_train["intent"])
    CONTROL[name] = classification_metrics(clean_test["intent"],
                                           ctl.predict(clean_test["text"]))["macro_f1"]
    m["in_distribution_macro_f1"] = CONTROL[name]
    m["shift_drop"] = CONTROL[name] - m["macro_f1"]
    save_sklearn(pipe, f"shift_{name.replace('+', '_')}")
    store.add(Result("intent_robustness_shift", name, "traditional", m,
                     train_seconds=t.seconds, notes="trained on clean flags, tested on Q/Z/W/P"))

# %% [markdown]
# ### Embedding and semantic-matching models

# %%
from sentence_transformers import SentenceTransformer

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
encoder = SentenceTransformer(MINILM, device=DEV)

def embed(texts):
    return encoder.encode(list(texts), batch_size=256, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)

E_train, E_test = embed(train_df["text"]), embed(test_df["text"])
E_ctl_tr, E_ctl_te = embed(clean_train["text"]), embed(clean_test["text"])

head = LogisticRegression(max_iter=3000, C=10.0, n_jobs=-1)
with timer("minilm+logreg") as t:
    head.fit(E_train, y_train)
m = classification_metrics(y_test, head.predict(E_test))
ctl = LogisticRegression(max_iter=3000, C=10.0, n_jobs=-1).fit(E_ctl_tr, clean_train["intent"])
m["in_distribution_macro_f1"] = classification_metrics(
    clean_test["intent"], ctl.predict(E_ctl_te))["macro_f1"]
m["shift_drop"] = m["in_distribution_macro_f1"] - m["macro_f1"]
save_sklearn(head, "shift_minilm_logreg")
store.add(Result("intent_robustness_shift", "minilm+logreg", "embedding", m,
                 train_seconds=t.seconds, notes="frozen encoder, clean-trained head"))

# Zero-shot uses no training text at all, so it cannot suffer a train/test shift — its
# score is the same on both sides by construction, which is the point.
P = embed(list(td.intent_prompts(LABELS).values()))
zs = np.array(LABELS)[(E_test @ P.T).argmax(1)]
m = classification_metrics(y_test, zs)
m["in_distribution_macro_f1"] = classification_metrics(
    clean_test["intent"], np.array(LABELS)[(E_ctl_te @ P.T).argmax(1)])["macro_f1"]
m["shift_drop"] = m["in_distribution_macro_f1"] - m["macro_f1"]
store.add(Result("intent_robustness_shift", "minilm_zeroshot_label_match", "embedding", m,
                 train_seconds=0.0, notes="no training text; immune to this shift by construction"))

# %% [markdown]
# ### Fine-tuned DistilBERT, retrained on the clean half only

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

lab2id = {l: i for i, l in enumerate(LABELS)}

class TxtDS(Dataset):
    def __init__(self, texts, labels, tok, maxlen=48):
        self.enc = tok(list(texts), truncation=True, padding="max_length",
                       max_length=maxlen, return_tensors="pt")
        self.y = torch.tensor([lab2id[v] for v in labels])
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def finetune_eval(model_name, tag, train_texts, train_labels, eval_sets, epochs=3, lr=3e-5, bs=64):
    set_seed()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)).to(DEV)
    tr = DataLoader(TxtDS(train_texts, train_labels, tok), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    with timer(f"{tag} fine-tune") as t:
        for ep in range(epochs):
            model.train(); tot = 0.0
            for b in tr:
                b = {k: v.to(DEV) for k, v in b.items()}
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                    out = model(**b)
                scaler.scale(out.loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += out.loss.item() * len(b["labels"])
            print(f"  epoch {ep+1} loss={tot/len(tr.dataset):.4f}")

    preds = {}
    model.eval()
    for key, (texts, labels) in eval_sets.items():
        dl = DataLoader(TxtDS(texts, labels, tok), batch_size=128)
        out = []
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
                out.append(model(**b).logits.argmax(1).cpu().numpy())
        preds[key] = np.array(LABELS)[np.concatenate(out)]
    return model, tok, preds, t.seconds


model, tok, preds, secs = finetune_eval(
    "distilbert-base-uncased", "distilbert_shift",
    train_df["text"], y_train,
    {"hard": (test_df["text"], y_test), "clean": (clean_test["text"], clean_test["intent"])},
)
m = classification_metrics(y_test, preds["hard"])
m["in_distribution_macro_f1"] = classification_metrics(clean_test["intent"],
                                                       preds["clean"])["macro_f1"]
m["shift_drop"] = m["in_distribution_macro_f1"] - m["macro_f1"]
out_dir = ARTIFACTS / "shift_distilbert_finetuned"
model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
store.add(Result("intent_robustness_shift", "distilbert_finetuned", "deep", m,
                 train_seconds=secs, notes="trained on clean flags only"))

# %% [markdown]
# ## 3. How much does the shift cost each family?

# %%
bench = store.frame(sort_by="macro_f1")
cols = ["model", "family", "accuracy", "macro_f1", "in_distribution_macro_f1", "shift_drop"]
print(bench[cols].round(4).to_string(index=False))
bench.to_csv(common.RESULTS / "summary_intent_shift.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
d = bench.sort_values("macro_f1")
y = np.arange(len(d))
ax.barh(y - 0.2, d["in_distribution_macro_f1"], height=0.4, label="clean test (in-distribution)",
        color="#4C72B0")
ax.barh(y + 0.2, d["macro_f1"], height=0.4, label="hard test (shifted)", color="#C44E52")
ax.set_yticks(y); ax.set_yticklabels(d["model"], fontsize=8)
ax.set_xlabel("macro-F1"); ax.set_title("cost of the clean -> noisy phrasing shift")
ax.legend(fontsize=8); plt.tight_layout()
fig.savefig(common.RESULTS / "intent_shift_gap.png", dpi=150)
plt.show()

# %%
best = bench.iloc[0]["model"]
err = pd.DataFrame({"text": test_df["text"], "true": y_test,
                    "pred": preds["hard"] if best == "distilbert_finetuned" else None})
if err["pred"].notna().all():
    err = err[err.true != err.pred]
    print(f"{len(err)} errors on the hard split ({100*len(err)/len(test_df):.2f}%)")
    print(err.groupby(["true", "pred"]).size().sort_values(ascending=False).head(10))
    err.head(30).to_csv(common.RESULTS / "intent_shift_error_samples.csv", index=False)

# %% [markdown]
# ## 4. Notes for the report
#
# * Quote **both** columns: the random-split scores from notebook 01 show the ceiling, the
#   shifted scores here show what survives contact with messy input. The `shift_drop`
#   column is the number that distinguishes the models.
# * Character n-grams matter under noise: they degrade far less than word-only features
#   because typos leave most character n-grams intact.
# * The zero-shot row is the control — it never saw training text, so its two columns
#   differ only by how hard each test set is, not by any train/test shift.
# * This split still understates the real problem: ASR output adds disfluencies, wrong word
#   boundaries and translation artefacts that no Bitext flag simulates. The honest next step
#   is to label a few hundred utterances from the call transcripts of notebook 03 and score
#   these same models against those.

# %%
print(f"results -> {store.path}")
