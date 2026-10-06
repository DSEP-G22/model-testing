# %% [markdown]
# # 08 — XLNet priority, exported for production
#
# Notebook 06 found XLNet-base fine-tuned the best customer-side priority model (urgency band,
# macro-F1 0.756 on `gold.jsonl`, mean of 3 seeds). This notebook makes that result deployable
# in the v3 triage service, which carries no torch:
#
# 1. fine-tune XLNet with notebook 06's exact recipe (3 seeds), keep the seed with the best
#    validation accuracy (gold is never used to choose);
# 2. export it to ONNX (dynamic length) and quantize it to int8 (per-channel, reduced range);
# 3. score the int8 ONNX on `gold.jsonl` through the **production path**: the `tokenizers`
#    library reading `tokenizer.json`, onnxruntime on CPU, one text at a time;
# 4. check the int8 model agrees with torch, and time it at 1 and 2 CPU threads.
#
# Output: `artifacts/priority_xlnet_onnx/` (model.int8.onnx, tokenizer.json, labels.json),
# released as an asset of `DSEP-G22/v3` and baked into the triage image.

# %%
import sys, json, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common import set_seed, device, env_info, classification_metrics, ARTIFACTS, RESULTS, ROOT, SEED

set_seed()
DEV = device()
print(json.dumps(env_info(), indent=2))
XLNET = "xlnet/xlnet-base-cased"
URGENCIES = ["critical", "high", "normal", "low"]
OUT = ARTIFACTS / "priority_xlnet_onnx"

# %% [markdown]
# ## 1. Data — identical to notebook 06

# %%
TRIAGE = ROOT / "TriageModel" / "data"


def load_triage(name):
    rows = [json.loads(l) for l in (TRIAGE / f"{name}.jsonl").open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("triage")]
    return pd.DataFrame({"text": [r["fused_text"] for r in rows],
                         "urgency": [r["triage"]["urgency"] for r in rows]})


p_all, p_gold = load_triage("train"), load_triage("gold")
p_train, p_val = train_test_split(p_all, test_size=0.15, stratify=p_all["urgency"], random_state=SEED)
print(f"priority train={len(p_train)} val={len(p_val)} gold={len(p_gold)}")

# %% [markdown]
# ## 2. Fine-tune, notebook 06 recipe: 10 epochs, batch 16, lr 3e-5, max_len 128, best-val epoch

# %%
lab2id = {l: i for i, l in enumerate(URGENCIES)}
tok = AutoTokenizer.from_pretrained(XLNET)


class TxtDS(Dataset):
    def __init__(self, texts, labels):
        self.enc = tok(list(texts), truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        self.y = torch.tensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.y[i]}


ytr = np.array([lab2id[v] for v in p_train["urgency"]])
yva = np.array([lab2id[v] for v in p_val["urgency"]])
ygo = np.array([lab2id[v] for v in p_gold["urgency"]])


def logits_of(model, texts):
    dl = DataLoader(TxtDS(texts, np.zeros(len(texts), int)), batch_size=64)
    model.eval(); out = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
            out.append(model(**b).logits.float().cpu().numpy())
    return np.concatenate(out)


def finetune(seed):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        XLNET, num_labels=4, id2label=dict(enumerate(URGENCIES)), label2id=lab2id).to(DEV)
    tr = DataLoader(TxtDS(p_train["text"], ytr), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=10 * len(tr), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    best_state, best_acc = None, -1
    for ep in range(10):
        model.train()
        for b in tr:
            b = {k: v.to(DEV) for k, v in b.items()}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                loss = model(**b).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        acc = (logits_of(model, p_val["text"]).argmax(1) == yva).mean()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, float(best_acc)


runs = []
for seed in (42, 43, 44):
    model, val_acc = finetune(seed)
    gold_pred = logits_of(model, p_gold["text"]).argmax(1)
    m = classification_metrics(ygo, gold_pred, labels=np.arange(4))
    runs.append({"seed": seed, "val_acc": val_acc, "gold_macro_f1": m["macro_f1"], "gold_accuracy": m["accuracy"]})
    print(runs[-1])
    if val_acc >= max(r["val_acc"] for r in runs):
        chosen = seed
        model.cpu().eval().save_pretrained(OUT / "torch"); tok.save_pretrained(OUT / "torch")
    del model; torch.cuda.empty_cache()
runs_df = pd.DataFrame(runs)
print(runs_df.to_string(index=False))
print(f"mean gold macro-F1 {runs_df.gold_macro_f1.mean():.4f} (std {runs_df.gold_macro_f1.std(ddof=0):.4f}); "
      f"chosen by val accuracy: seed {chosen}")

# %% [markdown]
# ## 3. Export to ONNX (dynamic batch and length), then int8 dynamic quantization
#
# The dynamo exporter traces XLNet's relative-position arithmetic correctly at every length;
# the legacy tracer froze it to the example's length. The mask goes in as int64 and is cast
# inside, so the service feeds the tokenizer's output unchanged.

# %%
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

best = AutoModelForSequenceClassification.from_pretrained(OUT / "torch").eval()


class Logits(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask.float(), use_mems=False).logits


ex = tok(["my fibre is down since morning"], return_tensors="pt")
fp32 = OUT / "model.fp32.onnx"
torch.onnx.export(Logits(best), (ex["input_ids"], ex["attention_mask"]), fp32, dynamo=True,
                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_shapes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}})
graph = onnx.load(fp32)
del graph.graph.value_info[:]  # stale intermediate shapes from the exporter trip the quantizer's shape inference
onnx.save(graph, fp32, save_as_external_data=False)  # one self-contained file
(OUT / "model.fp32.onnx.data").unlink(missing_ok=True)
# Plain S8 weights saturate the int16 accumulators of non-VNNI x86 kernels: on gold that copy
# fell from 0.789 to 0.526 macro-F1 and never predicted `critical`. Per-channel scales with
# 7-bit range (reduce_range) avoid the overflow on any CPU. Both are scored below.
quantize_dynamic(str(fp32), str(OUT / "model.int8-naive.onnx"), weight_type=QuantType.QInt8)
quantize_dynamic(str(fp32), str(OUT / "model.int8.onnx"), weight_type=QuantType.QInt8,
                 per_channel=True, reduce_range=True)
tok.backend_tokenizer.save(str(OUT / "tokenizer.json"))
(OUT / "labels.json").write_text(json.dumps({"labels": URGENCIES, "max_len": 128, "source": XLNET,
                                             "seed": chosen}, indent=1))
for f in ("model.fp32.onnx", "model.int8.onnx", "tokenizer.json"):
    print(f, round((OUT / f).stat().st_size / 2**20, 1), "MB")

# %% [markdown]
# ## 4. Score the int8 ONNX the way the triage service runs it

# %%
from tokenizers import Tokenizer

prod_tok = Tokenizer.from_file(str(OUT / "tokenizer.json"))
prod_tok.no_padding()
prod_tok.enable_truncation(128)
# The production tokenizer must produce exactly the ids the model was trained on.
for t in p_gold["text"]:
    assert prod_tok.encode(t).ids == tok(t, truncation=True, max_length=128)["input_ids"], t


def run_onnx(sess, text):
    e = prod_tok.encode(text)
    feed = {"input_ids": np.array([e.ids], np.int64), "attention_mask": np.array([e.attention_mask], np.int64)}
    return sess.run(None, feed)[0][0]


rows, preds = [], {}
torch_logits = logits_of(best.to(DEV), p_gold["text"])
preds["torch fp32"] = torch_logits.argmax(1)
for name in ("model.fp32.onnx", "model.int8-naive.onnx", "model.int8.onnx"):
    for threads in (1, 2):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(OUT / name), so, providers=["CPUExecutionProvider"])
        for t in p_gold["text"][:5]: run_onnx(sess, t)  # warm
        lat, logits = [], []
        for t in p_gold["text"]:
            t0 = time.perf_counter(); logits.append(run_onnx(sess, t)); lat.append((time.perf_counter() - t0) * 1000)
        pred = np.array(logits).argmax(1)
        preds[name] = pred
        m = classification_metrics(ygo, pred, labels=np.arange(4))
        rows.append({"model": name, "threads": threads, "gold_macro_f1": round(m["macro_f1"], 4),
                     "gold_accuracy": round(m["accuracy"], 4), "agree_with_torch": round(float((pred == preds["torch fp32"]).mean()), 4),
                     "p50_ms": round(float(np.median(lat)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1)})
m = classification_metrics(ygo, preds["torch fp32"], labels=np.arange(4))
rows.insert(0, {"model": "torch fp32 (GPU)", "threads": None, "gold_macro_f1": round(m["macro_f1"], 4),
                "gold_accuracy": round(m["accuracy"], 4), "agree_with_torch": 1.0, "p50_ms": None, "p95_ms": None})
export_df = pd.DataFrame(rows)
print(export_df.to_string(index=False))

# %% [markdown]
# ## 5. Save the record

# %%
from sklearn.metrics import confusion_matrix

int8_pred = preds["model.int8.onnx"]
cm = pd.DataFrame(confusion_matrix(ygo, int8_pred, labels=range(4)), index=URGENCIES, columns=URGENCIES)
print("int8 ONNX confusion on gold (rows true, columns predicted):\n", cm)
(RESULTS / "priority_xlnet_export.json").write_text(json.dumps(
    {"seeds": runs, "chosen_seed": chosen, "export": rows, "confusion_int8": cm.to_dict(),
     "sizes_mb": {f: round((OUT / f).stat().st_size / 2**20, 1) for f in ("model.fp32.onnx", "model.int8.onnx", "tokenizer.json")}},
    indent=1))
shutil.rmtree(OUT / "torch")
(OUT / "model.int8-naive.onnx").unlink()
print("wrote", RESULTS / "priority_xlnet_export.json")
