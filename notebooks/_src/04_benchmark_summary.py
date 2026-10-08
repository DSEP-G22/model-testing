# %% [markdown]
# # 04 — Cross-task benchmark summary
#
# Reads every `results/*.json` written by notebooks 01–03 and produces the consolidated
# tables and figures for the report. Run the three benchmark notebooks first; this one
# does no training of its own.

# %%
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
from common import RESULTS, FAMILY_COLORS, env_info

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

tasks = {}
for p in sorted(RESULTS.glob("*.json")):
    rows = json.loads(p.read_text())
    if not isinstance(rows, list):  # e.g. latest_generalization.json, not a ResultStore table
        continue
    recs = []
    for r in rows:
        rec = {"task": r["task"], "model": r["model"], "family": r["family"]}
        rec.update(r["metrics"])
        rec["train_s"] = r.get("train_seconds")
        rec["infer_ms"] = r.get("infer_ms_per_item")
        rec["size_mb"] = r.get("model_size_mb")
        rec["n_params"] = r.get("n_params")
        rec["notes"] = r.get("notes", "")
        recs.append(rec)
    tasks[p.stem] = pd.DataFrame(recs)
    print(f"{p.stem}: {len(recs)} models")

# %% [markdown]
# ## 1. Intent classification (text)

# %%
t = tasks.get("intent_classification")
if t is not None:
    sup = t[t.family != "unsupervised"].sort_values("macro_f1", ascending=False)
    print(sup[["model", "family", "accuracy", "balanced_acc", "macro_f1", "weighted_f1",
               "top3_accuracy", "train_s", "infer_ms", "size_mb"]].round(4).to_string(index=False))
    print("\nunsupervised:")
    uns = t[t.family == "unsupervised"].sort_values("ARI", ascending=False)
    print(uns[["model", "ARI", "NMI", "V_measure", "purity", "hungarian_acc",
               "n_clusters"]].round(4).to_string(index=False))
    sup.to_csv(RESULTS / "summary_intent_supervised.csv", index=False)
    uns.to_csv(RESULTS / "summary_intent_unsupervised.csv", index=False)

# %% [markdown]
# ## 2. Router crop classification (vision)

# %%
t = tasks.get("router_image_classification")
if t is not None:
    sup = t[t.family != "unsupervised"].sort_values("macro_f1", ascending=False)
    print(sup[["model", "family", "accuracy", "balanced_acc", "macro_f1", "weighted_f1",
               "train_s", "infer_ms", "size_mb"]].round(4).to_string(index=False))
    print("\nunsupervised:")
    uns = t[t.family == "unsupervised"].sort_values("ARI", ascending=False)
    print(uns[["model", "ARI", "NMI", "purity", "hungarian_acc", "n_clusters"]]
          .round(4).to_string(index=False))
    sup.to_csv(RESULTS / "summary_router_supervised.csv", index=False)

# %% [markdown]
# ## 3. Voice / ASR

# %%
t = tasks.get("voice_asr")
if t is not None:
    cols = [c for c in ["model", "wer", "cer", "embed_cos", "rtf", "lang_id_acc",
                        "chrf2", "bleu", "n_calls", "notes"] if c in t.columns]
    print(t[cols].round(4).sort_values("wer").to_string(index=False))
    t.to_csv(RESULTS / "summary_voice.csv", index=False)

# %% [markdown]
# ## 4. Winner per task and per family

# %%
rows = []
for name, df in tasks.items():
    metric = "wer" if name == "voice_asr" else "macro_f1"
    if metric not in df.columns:
        continue
    d = df.dropna(subset=[metric])
    for fam, grp in d.groupby("family"):
        best = grp.loc[grp[metric].idxmin() if metric == "wer" else grp[metric].idxmax()]
        rows.append({"task": name, "family": fam, "best_model": best["model"],
                     metric: round(float(best[metric]), 4),
                     "train_s": best.get("train_s"), "infer_ms": best.get("infer_ms")})
winners = pd.DataFrame(rows)
print(winners.to_string(index=False))
winners.to_csv(RESULTS / "summary_winners.csv", index=False)

# %% [markdown]
# ## 5. Consolidated figure

# %%
plot_tasks = [(k, "macro_f1") for k in ("intent_classification", "router_image_classification")
              if k in tasks] + ([("voice_asr", "wer")] if "voice_asr" in tasks else [])

fig, axes = plt.subplots(1, len(plot_tasks), figsize=(7 * len(plot_tasks), 6))
axes = np.atleast_1d(axes)
for ax, (name, metric) in zip(axes, plot_tasks):
    d = tasks[name].dropna(subset=[metric])
    d = d[d.family != "unsupervised"] if metric == "macro_f1" else d
    d = d.sort_values(metric, ascending=(metric != "wer"))
    ax.barh(d["model"], d[metric],
            color=[FAMILY_COLORS.get(f, "#888") for f in d["family"]])
    for i, v in enumerate(d[metric]):
        ax.text(v + d[metric].max() * 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_title(f"{name} — {metric}" + (" (lower better)" if metric == "wer" else ""))
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "summary_all_tasks.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Cost/benefit view — accuracy against training time

# %%
fig, ax = plt.subplots(figsize=(10, 6))
for name, df in tasks.items():
    if "macro_f1" not in df.columns:
        continue
    d = df.dropna(subset=["macro_f1", "train_s"])
    d = d[d["train_s"] > 0]
    marker = "o" if "intent" in name else "s"
    for fam, grp in d.groupby("family"):
        ax.scatter(grp["train_s"], grp["macro_f1"], marker=marker, s=70,
                   color=FAMILY_COLORS.get(fam, "#888"),
                   label=f"{name.split('_')[0]} / {fam}")
        for _, r in grp.iterrows():
            ax.annotate(r["model"], (r["train_s"], r["macro_f1"]), fontsize=6,
                        xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("training seconds (log)"); ax.set_ylabel("macro-F1")
ax.set_title("what each extra minute of training buys")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(RESULTS / "summary_cost_benefit.png", dpi=150); plt.show()

# %% [markdown]
# ## 7. Markdown report block
#
# Paste the output below straight into the project documentation.

# %%
lines = ["# Model benchmark summary", "",
         f"Environment: `{json.dumps(env_info())}`", ""]
for name, df in tasks.items():
    metric = "wer" if name == "voice_asr" else "macro_f1"
    if metric not in df.columns:
        continue
    d = df.dropna(subset=[metric]).sort_values(metric, ascending=(metric == "wer"))
    lines += [f"## {name}", ""]
    cols = [c for c in ["model", "family", "accuracy", "macro_f1", "wer", "cer", "rtf",
                        "lang_id_acc", "train_s", "infer_ms", "size_mb"] if c in d.columns]
    lines.append(d[cols].round(4).to_markdown(index=False))
    lines.append("")
report = "\n".join(lines)
(RESULTS / "BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")
print(report[:3000])
print(f"\nwritten -> {RESULTS / 'BENCHMARK_REPORT.md'}")
