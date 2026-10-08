# XLNet and Laya on intent and priority

Source: `notebooks/_src/06_latest_models.py` (run log `06_latest_models.log`, rows in
`intent_latest_models.json` / `priority_latest_models.json`). RTX 4050 6 GB, torch 2.13, laya 0.3.27.

## Intent: Bitext, de-duplicated, notebook 01 split (27 classes)

| Model | macro-F1 | Acc | Infer ms | Labels used |
|---|---:|---:|---:|---|
| tfidf_word+char+linsvc (nb 01) | **0.9991** | 0.9992 | 1.4 | all |
| **xlnet_finetuned** | 0.9974 | 0.9975 | 20.4 | all |
| minilm_finetuned (nb 01) | 0.9966 | 0.9969 | 8.3 | all |
| distilbert_finetuned (nb 01) | 0.9956 | 0.9961 | 8.2 | all |
| minilm+logreg (nb 01) | 0.9923 | 0.9931 | 9.9 | all |
| **laya_encoder+logreg** | 0.9742 | 0.9767 | 26.7 | all |
| **setfit_minilm_k64** | 0.9685 | 0.9717 | 7.1 | 64/intent (1,728) |
| **setfit_minilm_k8** | 0.9269 | 0.9298 | 6.0 | 8/intent (216) |
| minilm_prototype_k10 (nb 01) | 0.9106 | 0.9195 | 6.1 | 10/intent |
| **laya_zeroshot** | 0.9073 | 0.9145 | 78.2 | **none** |
| **laya_zeroshot_shortlist8** | 0.8957 | 0.9054 | 55.7 | none |
| minilm_zeroshot_label_match (nb 01) | 0.7281 | 0.7386 | 9.8 | none |

## Priority: urgency band, trained on `TriageModel/data/train.jsonl` (584), scored on `gold.jsonl` (150)

| Model | macro-F1 | Acc | Infer ms |
|---|---:|---:|---:|
| **xlnet_finetuned** (mean of 3 seeds, std 0.036) | **0.7557** | 0.7467 | 20.4 |
| distilled_multitask (score_table) | 0.7244 | 0.7267 | 0.08 |
| **laya_encoder+logreg** | 0.7063 | 0.6933 | 28.1 |
| minilm+logreg (score_table) | 0.7006 | 0.6800 | 0.13 |
| tfidf+logreg (score_table) | 0.6861 | 0.6667 | 0.61 |
| distilbert_finetuned (mean of 3 seeds, std 0.135) | 0.6585 | 0.6889 | 7.7 |
| **setfit_minilm** (mean of 3 seeds, std 0.008) | 0.6561 | 0.6644 | 8.4 |
| **laya_zeroshot** (teacher rubric as criteria) | 0.2946 | 0.3667 | 44.2 |

Laya zero-shot urgency confusion (rows true, columns predicted):

| | critical | high | low | normal |
|---|---:|---:|---:|---:|
| critical | 5 | 14 | 0 | 1 |
| high | 6 | 21 | 0 | 16 |
| low | 9 | 8 | 1 | 21 |
| normal | 9 | 11 | 0 | 28 |

## Reading

* **Intent:** XLNet is the best transformer, 0.0008 F1 above MiniLM fine-tuned, but the random
  Bitext split is saturated (caveat 1 in the README), so that gap is noise-level. It costs 2.5x
  MiniLM's latency. Laya's real value is **zero-shot**: 0.907 macro-F1 with no labels, +0.18 over
  the MiniLM label-match baseline. That matches the MiniLM 10-shot prototype row (0.911) without
  any labels, which makes it a good cold-start router for a new intent taxonomy. The MiniLM shortlist did not
  help; the widened head (`head_max_len=512`) already copes with 27 options.
* **Priority:** XLNet fine-tuned is the new best (0.756, +0.031 over the distilled head), but the
  3-seed std is 0.036 on a 150-row gold set, so treat it as "on par or slightly better", not a
  settled win. DistilBERT is unstable at this data size (std 0.135). Laya zero-shot fails on
  urgency: it never predicts `low` and over-uses `high`/`critical`, because severity
  here is a policy written in the rubric, not a property of the text it learned. Its frozen
  encoder is only on par with MiniLM's at roughly 200x the latency.
* **SetFit (MiniLM body):** the best few-shot option. With 8 examples per intent it scores 0.927,
  above MiniLM prototypes at 10 per intent (0.911) and Laya zero-shot (0.907). With 64 per intent it
  reaches 0.969 at 7 ms. On priority it is worse than plain frozen MiniLM+LogReg (0.656 vs 0.701, stable
  across seeds). The contrastive pairs are dominated by `low`/`normal` (only 15 `critical` tickets),
  so balanced recall sinks to 0.64. It needs class-balanced pair sampling before it is worth using here.
* **Not tested:** full RLCD fine-tuning of Laya (421M ModernBERT-large) does not fit a 6 GB GPU.
  The model card reports 0.36 to 0.77 accuracy jumps from it, so it is the obvious next run on
  Kaggle 2xT4 with `train.jsonl`.
