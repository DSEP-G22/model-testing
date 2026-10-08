# %% [markdown]
# # 06 — Latest models: XLNet and Laya on intent and priority
#
# Two newer model families against the incumbents, on the **same data and splits** the
# earlier benchmarks use:
#
# | Task | Data | Split | Incumbent numbers from |
# |---|---|---|---|
# | Intent (27-way) | Bitext, de-duplicated | notebook 01's stratified 70/15/15, seed 42 | `results/intent_classification.json` |
# | Priority (urgency band, 4-way) | `TriageModel/data/train.jsonl` (LLM-labelled) | scored on `gold.jsonl` | `TriageModel/artifacts/score_table.json` |
#
# Models:
#
# * **XLNet-base-cased** (`xlnet/xlnet-base-cased`, 117M) — permutation-LM transformer,
#   fine-tuned with notebook 01's recipe. DistilBERT is re-run on priority with the same
#   recipe so XLNet has a like-for-like fine-tuned baseline there.
# * **Laya** (`convaiinnovations/laya`, ModernBERT-large, 421M, Sep 2026) — a "System 1"
#   decision model: it scores typed options against a state in one forward pass, no text
#   generation. Tested three ways:
#   1. **zero-shot** — every label written as a criterion, no training labels at all;
#   2. **zero-shot + shortlist** (intent only) — 27 options exceed the ~20 its head budget
#      handles well, so MiniLM shortlists the top 8 first (still label-free);
#   3. **frozen encoder + LogReg** — Laya's encoder as a feature extractor, the same
#      protocol as `minilm+logreg`.
# * **SetFit** on MiniLM-L6 — contrastive few-shot fine-tuning plus a LogReg head
#   (section 7): 8 and 64 examples per intent, and all of `train.jsonl` for priority.
#
# Full RLCD fine-tuning of Laya (421M) is out of reach on a 6 GB laptop GPU, so (3) is the
# trained Laya row.

# %%
import sys, json, warnings, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from common import (
    Result, ResultStore, set_seed, device, env_info, timer, classification_metrics,
    measure_latency, count_params, ARTIFACTS, ROOT, SEED,
)
import text_data as td

set_seed()
DEV = device()
print(json.dumps(env_info(), indent=2))

XLNET = "xlnet/xlnet-base-cased"
DISTILBERT = "distilbert-base-uncased"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LAYA = "convaiinnovations/laya"

# %% [markdown]
# ## 1. Data — identical to notebook 01 and TriageModel

# %%
df = td.dedupe(td.load_bitext())
train_df, val_df, test_df = td.split(df)
INTENTS = sorted(df["intent"].unique())
print(f"intent  train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={len(INTENTS)}")

TRIAGE = ROOT / "TriageModel" / "data"


def load_triage(name):
    rows = [json.loads(l) for l in (TRIAGE / f"{name}.jsonl").open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("triage")]
    return pd.DataFrame({"text": [r["fused_text"] for r in rows],
                         "urgency": [r["triage"]["urgency"] for r in rows]})


from sklearn.model_selection import train_test_split

p_all = load_triage("train")
p_gold = load_triage("gold")
# The trainer in TriageModel fits on all of train.jsonl; we hold 15% out only to pick the
# best epoch, and score on gold.jsonl exactly as score_table.json does.
p_train, p_val = train_test_split(p_all, test_size=0.15, stratify=p_all["urgency"], random_state=SEED)
URGENCIES = ["critical", "high", "normal", "low"]
print(f"priority train={len(p_train)} val={len(p_val)} gold={len(p_gold)}")
print("gold urgency:", p_gold["urgency"].value_counts().to_dict())

intent_store = ResultStore("intent_latest_models")
priority_store = ResultStore("priority_latest_models")

# %% [markdown]
# ## 2. Fine-tuning — notebook 01's recipe, generalised over task

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TxtDS(Dataset):
    def __init__(self, texts, labels, tok, maxlen):
        self.enc = tok(list(texts), truncation=True, padding="max_length",
                       max_length=maxlen, return_tensors="pt")
        self.y = torch.tensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def finetune(model_name, labels, tr_x, tr_y, va_x, va_y, te_x,
             epochs=3, bs=64, lr=3e-5, maxlen=48, seed=SEED, save_to=None):
    """Train, keep the best-val-accuracy epoch, return (test logits, seconds, latency, model)."""
    set_seed(seed)
    lab2id = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=lab2id,
    ).to(DEV)
    ytr = np.array([lab2id[v] for v in tr_y]); yva = np.array([lab2id[v] for v in va_y])
    tr = DataLoader(TxtDS(tr_x, ytr, tok, maxlen), batch_size=bs, shuffle=True)
    va = DataLoader(TxtDS(va_x, yva, tok, maxlen), batch_size=128)
    te = DataLoader(TxtDS(te_x, np.zeros(len(te_x), int), tok, maxlen), batch_size=128)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    best_state, best_acc = None, -1

    def logits_of(loader):
        model.eval(); out = []
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
                with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                    out.append(model(**b).logits.float().cpu().numpy())
        return np.concatenate(out)

    started = time.perf_counter()
    for ep in range(epochs):
        model.train(); tot = 0.0
        for b in tr:
            b = {k: v.to(DEV) for k, v in b.items()}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                out = model(**b)
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += out.loss.item() * len(b["labels"])
        acc = (logits_of(va).argmax(1) == yva).mean()
        print(f"  {model_name} seed={seed} epoch {ep+1}  loss={tot/len(tr.dataset):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    logits = logits_of(te)

    def one(s):
        enc = tok([s], truncation=True, padding="max_length", max_length=maxlen,
                  return_tensors="pt").to(DEV)
        with torch.no_grad():
            return model(**enc).logits.argmax(1).item()
    lat = measure_latency(one, list(te_x), n=100)

    if save_to is not None:
        model.save_pretrained(save_to); tok.save_pretrained(save_to)
    return logits, seconds, lat, model


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def dir_mb(p):
    return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1024**2, 2)

# %% [markdown]
# ## 3. Intent — XLNet fine-tuned

# %%
out_dir = ARTIFACTS / "intent_xlnet_finetuned"
logits, secs, lat, xl = finetune(
    XLNET, INTENTS, train_df["text"], train_df["intent"], val_df["text"], val_df["intent"],
    test_df["text"], epochs=3, bs=32, lr=3e-5, maxlen=48, save_to=out_dir)
pred = np.array(INTENTS)[logits.argmax(1)]
m = classification_metrics(test_df["intent"], pred, softmax(logits), np.array(INTENTS))
intent_store.add(Result("intent_latest_models", "xlnet_finetuned", "deep", m,
                        train_seconds=secs, infer_ms_per_item=lat, model_size_mb=dir_mb(out_dir),
                        n_params=count_params(xl),
                        params={"epochs": 3, "lr": 3e-5, "batch_size": 32, "max_len": 48},
                        notes=f"fine-tuned {XLNET}, best-val checkpoint, notebook-01 recipe"))
del xl; torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Laya

# %%
import laya
from laya.shortlist import predict_shortlist, embed_fn_from_agent

agent = laya.load(LAYA, device=DEV)
# 27 intents share one option budget; widen it from the shipped 192 tokens so each label
# keeps its words (the model card's own advice for >20 options).
agent.cfg["head_max_len"] = 512
laya_params = count_params(agent.model)
print("laya params:", laya_params, "head_max_len:", agent.cfg["head_max_len"])

INTENT_Q = {"intent": {
    "type": "choice",
    "instructions": "What does the customer want? Pick the single best intent.",
    "criteria": td.intent_prompts(INTENTS),
}}


def laya_choices(states, questions, qid, batch_size=16):
    res = agent.predict_batch(list(states), questions, batch_size=batch_size)
    keys = list(questions[qid]["criteria"])
    pred = np.array([r["answers"][qid]["choice"] for r in res])
    proba = np.array([[r["answers"][qid]["probabilities"].get(k, 0.0) for k in keys] for r in res])
    return pred, proba

# %% [markdown]
# ### 4a. Intent — zero-shot (27 options, no labels)

# %%
started = time.perf_counter()
pred, proba = laya_choices(test_df["text"], INTENT_Q, "intent")
batch_s = time.perf_counter() - started
m = classification_metrics(test_df["intent"], pred, proba, np.array(INTENTS))
lat = measure_latency(lambda s: agent.predict(s, INTENT_Q), test_df["text"].tolist(), n=100)
intent_store.add(Result("intent_latest_models", "laya_zeroshot", "embedding", m,
                        train_seconds=0, infer_ms_per_item=lat, n_params=laya_params,
                        params={"head_max_len": 512, "criteria": "td.intent_prompts"},
                        notes=f"no training labels; 27-way choice; test pass {batch_s:.0f}s batched"))

# %% [markdown]
# ### 4b. Intent — zero-shot with a MiniLM shortlist (top 8)

# %%
from sentence_transformers import SentenceTransformer

minilm = SentenceTransformer(MINILM, device=DEV)
mini_embed = lambda xs: minilm.encode(list(xs), normalize_embeddings=True, convert_to_numpy=True)


def shortlist_one(s):
    return predict_shortlist(agent, s, INTENT_Q, mini_embed, k=8)["answers"]["intent"]["choice"]


started = time.perf_counter()
pred = np.array([shortlist_one(s) for s in test_df["text"]])
loop_s = time.perf_counter() - started
m = classification_metrics(test_df["intent"], pred)
lat = measure_latency(shortlist_one, test_df["text"].tolist(), n=100)
intent_store.add(Result("intent_latest_models", "laya_zeroshot_shortlist8", "embedding", m,
                        train_seconds=0, infer_ms_per_item=lat, n_params=laya_params,
                        params={"k": 8, "shortlist_encoder": MINILM},
                        notes=f"no training labels; MiniLM keeps 8 of 27 labels, Laya picks; {loop_s:.0f}s"))

# %% [markdown]
# ### 4c. Laya frozen encoder + LogReg (trained head)

# %%
from sklearn.linear_model import LogisticRegression

laya_embed = embed_fn_from_agent(agent, max_length=128, batch_size=64)


def frozen_head(store, task, name, tr_x, tr_y, te_x, te_y, labels):
    with timer(f"{name} encode") as t_enc:
        Xtr = laya_embed(list(tr_x)); Xte = laya_embed(list(te_x))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=1.0)
    with timer(f"{name} fit") as t_fit:
        clf.fit((Xtr - mu) / sd, list(tr_y))
    proba = clf.predict_proba((Xte - mu) / sd)
    order = [list(clf.classes_).index(l) for l in labels]
    pred = clf.predict((Xte - mu) / sd)
    m = classification_metrics(list(te_y), pred, proba[:, order], np.array(labels))
    lat = measure_latency(lambda s: clf.predict((laya_embed([s]) - mu) / sd), list(te_x), n=100)
    store.add(Result(task, name, "embedding", m,
                     train_seconds=t_fit.seconds, infer_ms_per_item=lat, n_params=laya_params,
                     params={"pool": "mean", "max_len": 128, "C": 1.0, "standardised": True},
                     notes=f"frozen Laya encoder; encoding took {t_enc.seconds:.0f}s"))
    return pred


frozen_head(intent_store, "intent_latest_models", "laya_encoder+logreg",
            train_df["text"], train_df["intent"], test_df["text"], test_df["intent"], INTENTS)

# %% [markdown]
# ## 5. Priority (urgency band) — Laya
#
# The criteria are the urgency rubric the LLM teacher labelled with
# (`TriageModel/label/prompt.py`), word for word, so Laya is asked the same question.

# %%
URGENCY_Q = {"urgency": {
    "type": "choice",
    "instructions": "How urgent is this support ticket? Anger alone is not urgency: a furious "
                    "customer asking a routine question is normal. An outage reported politely "
                    "is still high or critical.",
    "criteria": {
        "critical": "service is down for many people, or there is a safety or legal risk",
        "high": "this customer has no working service, or has been failed repeatedly",
        "normal": "a real problem with a working service, or a question needing a real answer",
        "low": "routine questions, information requests, praise",
    },
}}

pred, proba = laya_choices(p_gold["text"], URGENCY_Q, "urgency", batch_size=8)
m = classification_metrics(p_gold["urgency"], pred, proba, np.array(URGENCIES))
lat = measure_latency(lambda s: agent.predict(s, URGENCY_Q), p_gold["text"].tolist(), n=100)
priority_store.add(Result("priority_latest_models", "laya_zeroshot", "embedding", m,
                          train_seconds=0, infer_ms_per_item=lat, n_params=laya_params,
                          params={"criteria": "teacher rubric"},
                          notes="no training labels; 4-way choice on fused_text"))
pd.crosstab(p_gold["urgency"], pred, rownames=["true"], colnames=["laya"])

# %%
frozen_head(priority_store, "priority_latest_models", "laya_encoder+logreg",
            p_all["text"], p_all["urgency"], p_gold["text"], p_gold["urgency"], URGENCIES)

del agent; torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Priority — XLNet vs DistilBERT fine-tuned, 3 seeds
#
# 584 training tickets is small enough that one seed is noise, so each model is trained
# three times and the row reports the mean (std alongside).

# %%
for name, ckpt in [("xlnet_finetuned", XLNET), ("distilbert_finetuned", DISTILBERT)]:
    runs, secs_all, lats, n = [], [], [], None
    for seed in (42, 43, 44):
        logits, secs, lat, mdl = finetune(
            ckpt, URGENCIES, p_train["text"], p_train["urgency"], p_val["text"], p_val["urgency"],
            p_gold["text"], epochs=10, bs=16, lr=3e-5, maxlen=128, seed=seed)
        pred = np.array(URGENCIES)[logits.argmax(1)]
        runs.append(classification_metrics(p_gold["urgency"], pred, softmax(logits), np.array(URGENCIES)))
        secs_all.append(secs); lats.append(lat); n = count_params(mdl)
        del mdl; torch.cuda.empty_cache()
    m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    m.update({f"{k}_std": float(np.std([r[k] for r in runs])) for k in ("accuracy", "macro_f1")})
    priority_store.add(Result("priority_latest_models", name, "deep", m,
                              train_seconds=float(np.mean(secs_all)), infer_ms_per_item=float(np.median(lats)),
                              n_params=n,
                              params={"epochs": 10, "lr": 3e-5, "batch_size": 16, "max_len": 128,
                                      "seeds": [42, 43, 44]},
                              notes=f"fine-tuned {ckpt} on train.jsonl (85%), best-val epoch, mean of 3 seeds"))

# %% [markdown]
# ## 7. SetFit — contrastive few-shot fine-tuning
#
# SetFit (Tunstall et al., 2022) fine-tunes a sentence-transformer on pairs built from the
# labelled examples (same label → pull together, different → push apart), then fits a
# LogReg head on the tuned embeddings. Its pitch is label efficiency, so intent runs at
# 8 and 64 examples per intent. The body is MiniLM-L6, the same encoder as the
# `minilm+logreg` and prototype rows, so any gain is the contrastive step alone.

# %%
from datasets import Dataset as HFDataset
from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitArgs

SETFIT_TMP = ARTIFACTS / "_setfit_tmp"


def setfit_run(labels, tr_x, tr_y, te_x, iterations, seed=SEED, save_to=None):
    """Train SetFit; return (test pred, test proba in `labels` order, seconds, latency, model)."""
    set_seed(seed)
    model = SetFitModel.from_pretrained(MINILM, labels=list(labels)).to(DEV)
    args = SetFitArgs(output_dir=str(SETFIT_TMP), batch_size=32, num_epochs=1,
                      num_iterations=iterations, seed=seed, save_strategy="no", report_to="none")
    trainer = SetFitTrainer(model=model, args=args,
                            train_dataset=HFDataset.from_dict({"text": list(tr_x), "label": list(tr_y)}))
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started
    pred = np.array(model.predict(list(te_x), batch_size=128))
    # the head's columns follow its sorted classes_, not `labels`; reorder to `labels`
    classes = [str(c) for c in model.model_head.classes_]
    proba = model.predict_proba(list(te_x), batch_size=128).cpu().numpy()[:, [classes.index(l) for l in labels]]
    lat = measure_latency(lambda s: model.predict([s]), list(te_x), n=100)
    if save_to is not None:
        model.save_pretrained(str(save_to))
    return pred, proba, seconds, lat, model

# %% [markdown]
# ### 7a. Intent — 8 and 64 examples per intent

# %%
for k, iterations in ((8, 20), (64, 5)):
    shots = train_df.groupby("intent").sample(n=k, random_state=SEED)
    out_dir = ARTIFACTS / f"intent_setfit_minilm_k{k}"
    pred, proba, secs, lat, sf = setfit_run(INTENTS, shots["text"], shots["intent"], test_df["text"],
                                            iterations, save_to=out_dir)
    m = classification_metrics(test_df["intent"], pred, proba, np.array(INTENTS))
    intent_store.add(Result("intent_latest_models", f"setfit_minilm_k{k}", "deep", m,
                            train_seconds=secs, infer_ms_per_item=lat, model_size_mb=dir_mb(out_dir),
                            n_params=count_params(sf.model_body),
                            params={"shots_per_intent": k, "num_iterations": iterations, "epochs": 1,
                                    "batch_size": 32, "body": MINILM},
                            notes=f"SetFit, {len(shots)} labelled examples ({k}/intent)"))
    del sf; torch.cuda.empty_cache()

# %% [markdown]
# ### 7b. Priority — all 584 training tickets, 3 seeds
#
# Fitted on the whole of `train.jsonl` like the `score_table.json` baselines (SetFit picks no
# epoch, so it needs no validation slice).

# %%
runs, secs_all, lats = [], [], []
for seed in (42, 43, 44):
    pred, proba, secs, lat, sf = setfit_run(URGENCIES, p_all["text"], p_all["urgency"], p_gold["text"],
                                            iterations=20, seed=seed)
    runs.append(classification_metrics(p_gold["urgency"], pred, proba, np.array(URGENCIES)))
    secs_all.append(secs); lats.append(lat); n = count_params(sf.model_body)
    del sf; torch.cuda.empty_cache()
m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
m.update({f"{k}_std": float(np.std([r[k] for r in runs])) for k in ("accuracy", "macro_f1")})
priority_store.add(Result("priority_latest_models", "setfit_minilm", "deep", m,
                          train_seconds=float(np.mean(secs_all)), infer_ms_per_item=float(np.median(lats)),
                          n_params=n,
                          params={"num_iterations": 20, "epochs": 1, "batch_size": 32, "body": MINILM,
                                  "seeds": [42, 43, 44]},
                          notes="SetFit on train.jsonl (584), mean of 3 seeds"))

import shutil
shutil.rmtree(SETFIT_TMP, ignore_errors=True)

# %% [markdown]
# ## 8. Side by side with the incumbents

# %%
base_intent = {r["model"]: r for r in json.loads((Path.cwd().parent / "results" / "intent_classification.json").read_text())}
rows = [{"model": k, "macro_f1": base_intent[k]["metrics"]["macro_f1"],
         "accuracy": base_intent[k]["metrics"]["accuracy"],
         "infer_ms": base_intent[k].get("infer_ms_per_item"), "source": "notebook 01"}
        for k in ("tfidf_word+char+linsvc", "minilm+logreg", "minilm_finetuned", "minilm_prototype_k10",
                  "distilbert_finetuned", "minilm_zeroshot_label_match") if k in base_intent]
rows += [{"model": r["model"], "macro_f1": r["metrics"]["macro_f1"], "accuracy": r["metrics"]["accuracy"],
          "infer_ms": r["infer_ms_per_item"], "source": "this notebook"} for r in intent_store.rows.values()]
intent_cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
intent_cmp

# %%
score_table = json.loads((ROOT / "TriageModel" / "artifacts" / "score_table.json").read_text())
rows = [{"model": r["model"], "macro_f1": r["f1_urgency"], "accuracy": r["acc_urgency"],
         "infer_ms": r["latency_ms"], "source": "score_table.json"} for r in score_table]
rows += [{"model": r["model"], "macro_f1": r["metrics"]["macro_f1"], "accuracy": r["metrics"]["accuracy"],
          "infer_ms": r["infer_ms_per_item"], "source": "this notebook"} for r in priority_store.rows.values()]
priority_cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
priority_cmp

# %%
intent_cmp.to_csv(Path.cwd().parent / "results" / "summary_latest_intent.csv", index=False)
priority_cmp.to_csv(Path.cwd().parent / "results" / "summary_latest_priority.csv", index=False)
print(intent_cmp.to_string(index=False)); print(); print(priority_cmp.to_string(index=False))
