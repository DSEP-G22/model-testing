# %% [markdown]
# # 10 — Does the TF-IDF SVM classify on word triggers? (replicating `generalization.py`)
#
# Re-runs `v3/docs/final/experiments/generalization.py` step by step and checks every number
# against the report's saved `v3/docs/final/data/generalization.json`. Then it opens the SVM's
# decisions: a LinearSVC is linear, so each intent's score is an exact sum of
# `tf-idf value × learned weight` over the n-grams in the sentence, plus a bias. That sum shows
# which words decided each wrong answer.
#
# | Step | What | Size |
# |---|---|---:|
# | in distribution | Bitext test split (sanity: must match notebook 01) | 3,604 |
# | hand written | sentences written after training, 4 per intent (`content/generalization_sets.yaml`) | 108 |
# | padded | test split with 1–2 courtesy sentences around each message | 3,604 |
# | noise | test split with character typos at 0–20% of letters | 3,604 × 6 |
# | unknown | internet provider messages no Bitext intent covers | 30 |
# | evidence | per sentence: the n-grams pushing towards the predicted and the true intent | 108 |
# | vocabulary | share of words the training split never contained | — |
#
# Models are the saved notebook-01 artifacts, never retrained. Same seed (42), sets, perturbations
# and scoring as the script; the report's own data file is read, not written.

# %%
import sys, json, random, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import common  # noqa: F401  (interpreter guard, CUDA shim)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import latest_generalization  # noqa: F401  (puts v3/docs/final/experiments on sys.path)
import generalization as g    # the script under test: sets, split, perturbations, models

SAVED = json.loads((g.FINAL / "data/generalization.json").read_text(encoding="utf-8"))
SEED, NAMES = g.SEED, g.NAMES
test, train = g.test, g.train
y_test = test["intent"].tolist()
print(f"train={len(train)} test={len(test)} intents={len(g.LABELS)} models={NAMES}")
print(f"saved report file: n_train={SAVED['n_train']} n_test={SAVED['n_test']} seed={SAVED['seed']}")

# %% [markdown]
# ## 1. Load the six saved intent models

# %%
m = g.Models()
print("device:", m.dev)
print("SVM pipeline:", m.svc)

# %% [markdown]
# ## 2. In distribution (Bitext test split)

# %%
out = {}
base_pred, base_conf = {}, {}
for n in NAMES:
    base_pred[n], base_conf[n] = m.predict(n, test["text"])
out["in_distribution"] = {n: g.scores(y_test, base_pred[n]) for n in NAMES}

# %% [markdown]
# ## 3. Hand written sentences (new wording)

# %%
para = [(i, t) for i, ts in g.sets["paraphrases"].items() for t in ts]
y_p, x_p = [i for i, _ in para], [t for _, t in para]
pp = {n: m.predict(n, x_p) for n in NAMES}
out["paraphrase"] = {n: {**g.scores(y_p, pp[n][0]), "lenient_accuracy": float(g.lenient(y_p, pp[n][0]))}
                     for n in NAMES}

# Selective answering: answer only when confidence clears the 5th percentile of in-distribution confidence.
out["paraphrase_selective"] = {}
for n in NAMES:
    thr = float(np.percentile(base_conf[n], 5))
    keep = pp[n][1] >= thr
    correct = np.array(pp[n][0]) == np.array(y_p)
    out["paraphrase_selective"][n] = {
        "coverage": float(keep.mean()),
        "accuracy_when_answering": float(correct[keep].mean()) if keep.any() else None,
        "confidently_wrong": float((keep & ~correct).mean())}
print(f"{len(para)} sentences")

# %% [markdown]
# ## 4. Padding and typing noise

# %%
rng = random.Random(SEED)  # the script draws padding from this generator first, so order matters
x_pad = [g.padded(t, rng) for t in test["text"]]
out["padded"] = {n: g.scores(y_test, m.predict(n, x_pad)[0]) for n in NAMES}
print("padded example:", x_pad[0])

out["noise"] = {}
for rate in (0.0, 0.03, 0.06, 0.10, 0.15, 0.20):
    x_n = [g.noisy(t, rate, random.Random(SEED + k)) for k, t in enumerate(test["text"])]
    out["noise"][str(rate)] = {n: g.scores(y_test, m.predict(n, x_n)[0]) for n in NAMES}
print("10% noise example:", g.noisy(test["text"][0], 0.10, random.Random(SEED)))

# %% [markdown]
# ## 5. Unknown topics: does the model know it is lost?

# %%
isp = g.sets["isp"]
out["unknown"] = {}
for n in NAMES:
    _, iconf = m.predict(n, isp)
    auroc = float(roc_auc_score([1] * len(base_conf[n]) + [0] * len(isp), np.concatenate([base_conf[n], iconf])))
    thr = float(np.percentile(base_conf[n], 5))
    out["unknown"][n] = {"auroc": auroc, "accepted": float((iconf >= thr).mean())}
print(f"{len(isp)} unknown-topic messages")

# %% [markdown]
# ## 6. Results side by side (macro-F1)

# %%
table = pd.DataFrame({
    "in distribution": {n: out["in_distribution"][n]["macro_f1"] for n in NAMES},
    "hand written": {n: out["paraphrase"][n]["macro_f1"] for n in NAMES},
    "hand written, lenient acc": {n: out["paraphrase"][n]["lenient_accuracy"] for n in NAMES},
    "padded": {n: out["padded"][n]["macro_f1"] for n in NAMES},
    "10% typos": {n: out["noise"]["0.1"][n]["macro_f1"] for n in NAMES},
    "unknown AUROC": {n: out["unknown"][n]["auroc"] for n in NAMES},
}).round(3)
table

# %% [markdown]
# ## 7. Opening the SVM: the decision is an exact sum over n-grams
#
# The pipeline is `FeatureUnion(word tf-idf 1–2 grams, char_wb tf-idf 3–5 grams) → LinearSVC`.
# For intent *c*: `score_c = Σ_i x_i · w_c,i + b_c`, where `x_i` is the tf-idf value of n-gram *i* in
# the sentence (zero if absent) and `w_c,i` is `coef_[c, i]`. So `x_i · w_c,i` is exactly how hard
# n-gram *i* pushed towards *c*. First, prove the decomposition reproduces `decision_function`.

# %%
union = m.svc.steps[0][1]
vec_w, vec_c = union.transformer_list[0][1], union.transformer_list[1][1]
clf = m.svc.named_steps["clf"]
coef, bias, classes = clf.coef_, clf.intercept_, list(m.svc.classes_)
names_w = vec_w.get_feature_names_out()
n_word = len(names_w)
print(f"features: {n_word} word n-grams + {len(vec_c.get_feature_names_out())} char n-grams, {len(classes)} intents")

X = union.transform(x_p)
manual = np.asarray(X @ coef.T) + bias
assert np.allclose(manual, m.svc.decision_function(x_p)), "decomposition does not reproduce the SVM"
print("Σ tf-idf × weight + bias == decision_function for all", len(x_p), "sentences")

# %%
def explain(text, intent, top=8):
    """Every n-gram's push towards `intent`, split into word part, char part and bias."""
    k = classes.index(intent)
    row = union.transform([text]).tocsr()
    contrib = {i: v * coef[k, i] for i, v in zip(row.indices, row.data)}
    word = sorted(((c, names_w[i]) for i, c in contrib.items() if i < n_word), reverse=True)
    char_total = sum(c for i, c in contrib.items() if i >= n_word)
    return {"intent": intent, "score": sum(contrib.values()) + bias[k],
            "word_total": sum(c for c, _ in word), "char_total": char_total, "bias": bias[k],
            "top_words": [(w, round(float(c), 3)) for c, w in word[:top]]}


text = "Will it get here before the weekend?"
pred = str(m.svc.predict([text])[0])
for intent in (pred, "delivery_period"):
    e = explain(text, intent)
    print(f"\n{intent:16s} score {e['score']:+.3f} = words {e['word_total']:+.3f} "
          f"+ chars {e['char_total']:+.3f} + bias {e['bias']:+.3f}")
    print("  top word n-grams:", e["top_words"])

# %% [markdown]
# ## 8. Word evidence for every hand written sentence (the script's `svc_evidence`)
#
# Same rule as the script: word n-grams only, `tf-idf × coef` towards the predicted and the true
# intent, top 3 each, plus how many word n-grams of the sentence exist in the vocabulary at all.

# %%
evidence = []
for k, t in enumerate(x_p):
    row = vec_w.transform([t])
    pred_i = classes.index(str(pp["tfidf_svc"][0][k]))
    true_i = classes.index(y_p[k])
    idx, val = row.indices, row.data
    toward_pred = sorted(((float(v * coef[pred_i, i]), names_w[i]) for i, v in zip(idx, val)), reverse=True)[:3]
    toward_true = sorted(((float(v * coef[true_i, i]), names_w[i]) for i, v in zip(idx, val)), reverse=True)[:3]
    evidence.append({"fired": int(len(idx)), "tokens": len(g.tokens(t)),
                     "toward_pred": [[w, round(c, 3)] for c, w in toward_pred],
                     "toward_true": [[w, round(c, 3)] for c, w in toward_true]})
rows = [{"text": t, "true": y, "svc_evidence": evidence[k],
         **{n: {"pred": str(pp[n][0][k]), "conf": float(pp[n][1][k])} for n in NAMES}}
        for k, (y, t) in enumerate(para)]

# %% [markdown]
# ### The failure-analysis slide rows
# SVM wrong, frozen MiniLM right, most confident SVM answer first, one row per wrong intent
# (`lib/facts.py: generalization()`). The deciding word is the top `toward_pred` n-gram.

# %%
fails = sorted((r for r in rows if r["tfidf_svc"]["pred"] != r["true"] and r["minilm_logreg"]["pred"] == r["true"]),
               key=lambda r: -r["tfidf_svc"]["conf"])
seen, slide = set(), []
for r in fails:
    if r["tfidf_svc"]["pred"] in seen:
        continue
    seen.add(r["tfidf_svc"]["pred"])
    ev = r["svc_evidence"]
    slide.append({"sentence": r["text"], "true": r["true"], "SVM said": r["tfidf_svc"]["pred"],
                  "SVM margin": round(r["tfidf_svc"]["conf"], 2), "MiniLM said": r["minilm_logreg"]["pred"],
                  "word n-grams known": f'{ev["fired"]} of {ev["tokens"]} words',
                  "deciding word": f'{ev["toward_pred"][0][0]} ({ev["toward_pred"][0][1]})',
                  "best push to truth": f'{ev["toward_true"][0][0]} ({ev["toward_true"][0][1]})'})
pd.set_option("display.max_colwidth", 70)
pd.DataFrame(slide).head(4)

# %% [markdown]
# ### How concentrated is the evidence across all SVM errors?
# Share of the positive word push towards the wrong intent that comes from the single top n-gram.

# %%
wrong = [r for r in rows if r["tfidf_svc"]["pred"] != r["true"]]
share = []
for r in wrong:
    pushes = [c for _, c in explain(r["text"], r["tfidf_svc"]["pred"], top=10_000)["top_words"] if c > 0]
    if pushes:
        share.append(pushes[0] / sum(pushes))
share = np.array(share)
print(f"SVM wrong on {len(wrong)} of {len(rows)} hand written sentences")
print(f"top n-gram's share of the push to the wrong intent: median {np.median(share):.0%}, "
      f"over half in {np.mean(share > 0.5):.0%} of errors")

# %% [markdown]
# ## 9. Vocabulary: words the training split never contained

# %%
seen_words = {w for t in train["text"] for w in g.tokens(t)}


def oov(texts):
    ws = [w for t in texts for w in g.tokens(t)]
    return float(np.mean([w not in seen_words for w in ws]))


out["oov"] = {"test": oov(test["text"]), "padded": oov(x_pad), "paraphrase": oov(x_p), "unknown": oov(isp),
              "noise_10": oov([g.noisy(t, 0.10, random.Random(SEED + k)) for k, t in enumerate(test["text"])])}
pd.Series(out["oov"]).map("{:.1%}".format)

# %% [markdown]
# ## 10. Replication check against the report's `generalization.json`
# Every score above is compared with the saved file. sklearn models are deterministic; the two
# fine-tuned transformers may move in the last digit between GPU and CPU, so the tolerance is 0.005.

# %%
def flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and v is not None:
            yield f"{prefix}{k}", float(v)


ours = dict(flat({k: out[k] for k in ("in_distribution", "paraphrase", "paraphrase_selective", "padded", "noise", "oov")}))
ours.update({f"unknown.{n}.{k}": v for n in NAMES for k, v in out["unknown"][n].items()})
saved = dict(flat({k: SAVED[k] for k in ("in_distribution", "paraphrase", "paraphrase_selective", "padded", "noise", "oov", "unknown")}))
diff = pd.DataFrame([(k, saved[k], v, abs(saved[k] - v)) for k, v in ours.items() if k in saved],
                    columns=["metric", "report", "this run", "abs diff"])
print(f"compared {len(diff)} numbers; max abs diff {diff['abs diff'].max():.4f}")
evid_same = sum(r["svc_evidence"] == s["svc_evidence"] for r, s in zip(rows, SAVED["paraphrase_rows"]))
print(f"word evidence identical for {evid_same} of {len(rows)} sentences")
diff.sort_values("abs diff", ascending=False).head(10)

# %%
assert diff["abs diff"].max() <= 0.005, "replication drifted from the report"
assert evid_same == len(rows), "word evidence differs from the report"
print("REPLICATED: all scores within 0.005 and identical word evidence")
