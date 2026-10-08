"""XLNet and Laya on the hand written generalization sets (v3/docs/final/experiments/generalization.py).

Reuses that script's sets, perturbations, seeds and scoring, so the numbers sit next to its
generalization.json rows without any re-implementation. The report's own data file is not touched.

    cd "model testing"
    .venv/Scripts/python src/latest_generalization.py

Writes results/latest_generalization.json and results/LATEST_GENERALIZATION.md.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
EXP = BENCH.parent / "v3" / "docs" / "final" / "experiments"
sys.path.insert(0, str(EXP))
import generalization as g  # noqa: E402  (loads sets, the de-duplicated split, LABELS)

from sklearn.metrics import roc_auc_score  # noqa: E402


class Latest:
    """Same contract as generalization.Models.predict: (labels, confidence per text)."""

    def __init__(self):
        import torch
        import laya
        from sklearn.linear_model import LogisticRegression
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from laya.shortlist import embed_fn_from_agent

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        folder = BENCH / "artifacts" / "intent_xlnet_finetuned"
        self.xl_tok = AutoTokenizer.from_pretrained(folder)
        self.xl = AutoModelForSequenceClassification.from_pretrained(folder).to(self.dev).eval()

        self.agent = laya.load("convaiinnovations/laya", device=self.dev)
        self.agent.cfg["head_max_len"] = 512  # as in notebook 06
        self.q = {"intent": {"type": "choice",
                             "instructions": "What does the customer want? Pick the single best intent.",
                             "criteria": g.td.intent_prompts(g.LABELS)}}
        self.embed = embed_fn_from_agent(self.agent, max_length=128, batch_size=64)
        # notebook 06's laya_encoder+logreg, refitted on the same train split (it was not saved)
        x = self.embed(g.train["text"].tolist())
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.lr = LogisticRegression(max_iter=3000, C=1.0).fit((x - self.mu) / self.sd, g.train["intent"])
        # notebook 06's SetFit models (MiniLM body, contrastive fine-tuning, LogReg head)
        from setfit import SetFitModel
        self.setfit = {k: SetFitModel.from_pretrained(str(BENCH / "artifacts" / f"intent_setfit_minilm_k{k}")).to(self.dev)
                       for k in (8, 64)}

    def predict(self, name, texts):
        texts = list(texts)
        if name == "xlnet_ft":
            probs = []
            with self.torch.no_grad():
                for i in range(0, len(texts), 256):
                    enc = self.xl_tok(texts[i:i + 256], padding=True, truncation=True, max_length=128,
                                      return_tensors="pt").to(self.dev)
                    probs.append(self.torch.softmax(self.xl(**enc).logits.float(), -1).cpu().numpy())
            p = np.concatenate(probs)
            return np.array([self.xl.config.id2label[i] for i in p.argmax(1)]), p.max(1)
        if name == "laya_zeroshot":
            res = self.agent.predict_batch(texts, self.q, batch_size=16)
            pred = np.array([r["answers"]["intent"]["choice"] for r in res])
            conf = np.array([max(r["answers"]["intent"]["probabilities"].values()) for r in res])
            return pred, conf
        if name == "laya_encoder_logreg":
            p = self.lr.predict_proba((self.embed(texts) - self.mu) / self.sd)
            return self.lr.classes_[p.argmax(1)], p.max(1)
        if name.startswith("setfit_k"):
            m = self.setfit[int(name[len("setfit_k"):])]
            p = m.predict_proba(texts, batch_size=256).cpu().numpy()
            return np.array([str(c) for c in m.model_head.classes_])[p.argmax(1)], p.max(1)
        raise KeyError(name)


NAMES = ["xlnet_ft", "laya_zeroshot", "laya_encoder_logreg", "setfit_k8", "setfit_k64"]


def evaluate(m, names) -> dict:
    """Run the four stress tests for `names` on model holder `m` (Latest or generalization.Models)."""
    NAMES = list(names)
    rng = random.Random(g.SEED)
    test, sets = g.test, g.sets
    y_test = test["intent"].tolist()
    out: dict = {"models": NAMES, "n_test": len(test)}

    base_pred, base_conf = {}, {}
    for n in NAMES:
        base_pred[n], base_conf[n] = m.predict(n, test["text"])
    out["in_distribution"] = {n: g.scores(y_test, base_pred[n]) for n in NAMES}

    para = [(i, t) for i, ts in sets["paraphrases"].items() for t in ts]
    y_p, x_p = [i for i, _ in para], [t for _, t in para]
    pp = {n: m.predict(n, x_p) for n in NAMES}
    out["paraphrase"] = {n: {**g.scores(y_p, pp[n][0]), "lenient_accuracy": float(g.lenient(y_p, pp[n][0]))}
                         for n in NAMES}
    out["paraphrase_selective"] = {}
    for n in NAMES:
        thr = float(np.percentile(base_conf[n], 5))
        keep = pp[n][1] >= thr
        correct = np.array(pp[n][0]) == np.array(y_p)
        out["paraphrase_selective"][n] = {
            "coverage": float(keep.mean()),
            "accuracy_when_answering": float(correct[keep].mean()) if keep.any() else None,
            "confidently_wrong": float((keep & ~correct).mean())}
    out["paraphrase_rows"] = [{"text": t, "true": y, **{n: str(pp[n][0][k]) for n in NAMES}}
                              for k, (y, t) in enumerate(para)]

    x_pad = [g.padded(t, rng) for t in test["text"]]
    out["padded"] = {n: g.scores(y_test, m.predict(n, x_pad)[0]) for n in NAMES}

    out["noise"] = {}
    for rate in (0.0, 0.03, 0.06, 0.10, 0.15, 0.20):
        x_n = [g.noisy(t, rate, random.Random(g.SEED + k)) for k, t in enumerate(test["text"])]
        # rate 0 is the in-distribution pass again; reuse it rather than re-running Laya
        out["noise"][str(rate)] = (out["in_distribution"] if rate == 0.0 else
                                   {n: g.scores(y_test, m.predict(n, x_n)[0]) for n in NAMES})
        print("noise", rate, {n: round(v["macro_f1"], 3) for n, v in out["noise"][str(rate)].items()}, flush=True)

    isp = sets["isp"]
    out["unknown"] = {}
    for n in NAMES:
        _, conf = m.predict(n, isp)
        thr = float(np.percentile(base_conf[n], 5))
        out["unknown"][n] = {
            "auroc": float(roc_auc_score([1] * len(base_conf[n]) + [0] * len(isp),
                                         np.concatenate([base_conf[n], conf]))),
            "accepted": float((conf >= thr).mean()),
            "mean_conf_in": float(base_conf[n].mean()), "mean_conf_unknown": float(conf.mean())}

    return out


def main() -> None:
    out = evaluate(Latest(), NAMES)
    (BENCH / "results" / "latest_generalization.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    old = json.loads((EXP.parent / "data" / "generalization.json").read_text(encoding="utf-8"))
    write_report([old, out])


def table(runs: list[dict]) -> list[tuple[str, dict]]:
    rows = [(n, d) for d in runs for n in d["models"]]
    return sorted(rows, key=lambda r: -r[1]["paraphrase"][r[0]]["macro_f1"])


def write_report(runs: list[dict]) -> None:
    rows = table(runs)
    rows.sort(key=lambda r: -r[1]["paraphrase"][r[0]]["macro_f1"])

    def f(x):
        return f"{x:.3f}"

    lines = ["# Hand written generalization: XLNet and Laya vs every intent model", "",
             "Sets, perturbations and scoring from `v3/docs/final/experiments/generalization.py`.", "",
             "| Model | In-dist F1 | Hand written F1 | Hand written lenient acc | Drop | Padded F1 | Noise 10% F1 | Noise 20% F1 | Unknown AUROC | Unknown accepted |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, d in rows:
        i, p = d["in_distribution"][n]["macro_f1"], d["paraphrase"][n]
        u = d["unknown"][n]
        lines.append(f"| {n} | {f(i)} | **{f(p['macro_f1'])}** | {f(p['lenient_accuracy'])} | {f(i - p['macro_f1'])} | "
                     f"{f(d['padded'][n]['macro_f1'])} | {f(d['noise']['0.1'][n]['macro_f1'])} | "
                     f"{f(d['noise']['0.2'][n]['macro_f1'])} | {f(u['auroc'])} | {u['accepted']:.0%} |")
    lines += ["", "Hand written = 108 sentences, 4 per intent. Lenient counts 10 confusable intent pairs as right.",
              "Unknown accepted = share of 30 out-of-taxonomy ISP messages whose confidence clears the 5th percentile",
              "of in-distribution confidence (lower is better)."]
    (BENCH / "results" / "LATEST_GENERALIZATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
