# %% [markdown]
# # 07 — Drift check on hand written sentences (all intent models)
#
# Every intent model scores 0.99+ on the Bitext test split because the corpus is
# template-generated. This notebook asks the drift question instead: what happens on
# wording the templates never produced? It runs the four stress tests from
# `v3/docs/final/experiments/generalization.py`, using that script's sets, perturbations,
# seed and scoring. It re-measures the report's six models live and adds XLNet, Laya and SetFit
# from notebook 06.
#
# | Test | What | Size |
# |---|---|---:|
# | hand written | sentences the team wrote after training, 4 per intent (`content/generalization_sets.yaml`) | 108 |
# | padded | test split with 1–2 polite courtesy sentences around each message | 3,604 |
# | noise | test split with character typos at 3–20% of letters | 3,604 × 5 |
# | unknown | internet provider messages no Bitext intent covers | 30 |
#
# The report's own `data/generalization.json` is not written here.

# %%
import sys, json, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import common  # noqa: F401  (interpreter guard, CUDA shim)
import pandas as pd
import torch

import latest_generalization as lg
g = lg.g
print(f"hand written sentences: {sum(len(v) for v in g.sets['paraphrases'].values())}, "
      f"unknown: {len(g.sets['isp'])}, test split: {len(g.test)}")

# %% [markdown]
# ## 1. The report's six models (saved notebook-01 artifacts, never retrained)

# %%
holder = g.Models()
base = lg.evaluate(holder, g.NAMES)
del holder; torch.cuda.empty_cache()

# %% [markdown]
# ## 2. XLNet fine-tuned, Laya and SetFit (notebook 06)

# %%
holder = lg.Latest()
latest = lg.evaluate(holder, lg.NAMES)
del holder; torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Results

# %%
rows = []
for n, d in lg.table([base, latest]):
    p, u = d["paraphrase"][n], d["unknown"][n]
    rows.append({
        "model": n,
        "test_f1": d["in_distribution"][n]["macro_f1"],
        "hand_written_f1": p["macro_f1"],
        "hand_written_lenient_acc": p["lenient_accuracy"],
        "drop": d["in_distribution"][n]["macro_f1"] - p["macro_f1"],
        "padded_f1": d["padded"][n]["macro_f1"],
        "noise10_f1": d["noise"]["0.1"][n]["macro_f1"],
        "noise20_f1": d["noise"]["0.2"][n]["macro_f1"],
        "unknown_auroc": u["auroc"],
        "unknown_accepted": u["accepted"],
    })
summary = pd.DataFrame(rows).round(3)
print(summary.to_string(index=False))

# %% [markdown]
# ### Typo sweep (macro-F1 by character error rate)

# %%
noise = pd.DataFrame({rate: {**{n: v["macro_f1"] for n, v in base["noise"][rate].items()},
                             **{n: v["macro_f1"] for n, v in latest["noise"][rate].items()}}
                      for rate in base["noise"]}).round(3)
print(noise.to_string())

# %% [markdown]
# ### Selective answering on the hand written set
# Answer only when confidence clears the 5th percentile of in-distribution confidence.
# `confidently_wrong` is the share of all 108 sentences answered confidently and wrongly.

# %%
sel = pd.DataFrame({**base["paraphrase_selective"], **latest["paraphrase_selective"]}).T.round(3)
print(sel.to_string())

# %% [markdown]
# ### Where each model goes wrong (hand written set)

# %%
models = base["models"] + latest["models"]
pred = pd.DataFrame([{"text": b["text"], "true": b["true"],
                      **{n: b[n] for n in base["models"]},
                      **{n: l[n] for n in latest["models"]}}
                     for b, l in zip(base["paraphrase_rows"], latest["paraphrase_rows"])])
per_intent = pd.DataFrame({n: (pred[n] == pred["true"]).groupby(pred["true"]).mean() for n in models}).round(2)
print(per_intent.to_string())

# %%
wrong_everywhere = pred[[all(pred.loc[i, n] != pred.loc[i, "true"] for n in models) for i in pred.index]]
print(f"{len(wrong_everywhere)} sentences no model gets right:")
print(wrong_everywhere[["true", "text"]].to_string(index=False))

# %%
out = Path.cwd().parent / "results"
summary.to_csv(out / "summary_generalization_all.csv", index=False)
(out / "latest_generalization.json").write_text(json.dumps(latest, indent=1), encoding="utf-8")
lg.write_report([base, latest])
