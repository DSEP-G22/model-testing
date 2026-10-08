# model testing

Benchmark suite for the three DSEP22 data assets. Every notebook trains a spread of
traditional and deep models on one dataset, scores them on a fixed held-out split, saves
the trained artifacts, and writes machine-readable results into `results/`.

## Layout

```
model testing/
├── notebooks/            executable .ipynb (generated from notebooks/_src/*.py)
│   └── _src/             jupytext-percent sources — edit these, then rebuild
├── src/                  shared library imported by every notebook
│   ├── common.py         paths, seeding, metrics, ResultStore, plots
│   ├── text_data.py      Bitext loading, de-duplication, stratified split
│   ├── vision_data.py    COCO boxes -> crop-classification dataset + cache
│   ├── audio_data.py     .docx transcript parsing, call corpus, text normalisation
│   └── py2ipynb.py       percent-script -> notebook converter
├── artifacts/            trained models (.joblib, .pt, HF folders, transcripts)
├── results/              *.json benchmark rows, *.csv summaries, *.png figures
├── data_cache/           cached embeddings and image crops
└── requirements.txt
```

## Setup

```powershell
cd "d:\DSEP22\model testing"
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name dsep22-modeltesting --display-name "Python (model testing)"
```

Rebuild the notebooks after editing a source file:

```powershell
.\.venv\Scripts\python src\build_notebooks.py
```

## Notebooks

| Notebook | Dataset | Task | Models |
|---|---|---|---|
| `01_intent_classification` | Bitext 27K customer support | 27-way intent classification | TF-IDF + LogReg / LinearSVC / ComplementNB / SGD / RandomForest / SVD+HistGB; MiniLM frozen + LogReg / kNN / LinearSVC / MLP; MiniLM zero-shot and few-shot prototype matching; TextCNN from scratch; MiniLM and DistilBERT fine-tuned; KMeans / Agglomerative / GMM / HDBSCAN / LDA |
| `02_router_image_classification` | Roboflow router detection v38 | 15-way crop classification | HOG + LinearSVC / LogReg; colour histogram + RandomForest; pixels + PCA + LogReg; HOG+colour + HistGB; frozen ResNet18 / MobileNetV3 / EfficientNet-B0 + LogReg; the same three fine-tuned; KMeans / Agglomerative / HDBSCAN on CNN features |
| `03_voice_transcription_benchmark` | 11 call-centre recordings, 7 languages | transcription, translation, language ID | Whisper tiny/base/small/medium, faster-whisper small/medium (CTranslate2 int8) |
| `src/mt_benchmark.py` | FLORES-200 devtest (mteb/flores), si/ta <-> en | machine translation | NLLB-200 distilled 600M vs 1.3B, CTranslate2 int8 on CPU (see `results/TRANSLATION_BENCHMARK.md`) |
| `06_latest_models` | Bitext (nb 01 split) + TriageModel train/gold | intent and urgency | XLNet-base fine-tuned; Laya zero-shot, zero-shot + MiniLM shortlist, frozen encoder + LogReg; SetFit (MiniLM, 8/64 shots); DistilBERT on urgency (see `results/LATEST_MODELS_REPORT.md`) |
| `07_generalization_hand_written` | 108 hand written sentences, padded/typo test split, 30 off-taxonomy ISP messages (`v3/docs/final/content/generalization_sets.yaml`) | intent drift check | all 9 intent models: the report's six plus XLNet and the two Laya variants (see `results/LATEST_GENERALIZATION.md`) |
| `04_benchmark_summary` | — | consolidation | reads `results/*.json`, writes the summary CSVs, figures and `BENCHMARK_REPORT.md` |

Run 01–03 in any order, then 04. 05–07 read the Bitext split and artifacts from 01; 07 also needs 06 (XLNet artifact).

## Metrics

* **Classification** — accuracy, balanced accuracy, macro-F1, weighted-F1, macro
  precision/recall, top-3 accuracy, training seconds, single-item inference latency,
  artifact size on disk, parameter count.
* **Clustering** — ARI, NMI, V-measure, purity, Hungarian-matched accuracy, silhouette,
  discovered cluster count, noise fraction. Labels are used only for scoring.
* **ASR** — WER, CER, MER, WIL, embedding cosine similarity, real-time factor,
  language-ID accuracy; chrF++ and BLEU for the translate task. Aggregates are
  duration-weighted so long calls are not out-voted by short ones.

## Reading the numbers — two caveats that matter

1. **Bitext is template-generated.** Identical utterances repeat many times; notebook 01
   de-duplicates *before* splitting and reports how much leakage that removes. Quote the
   de-duplicated scores — the raw-file scores are inflated.
2. **The call-centre `.docx` transcripts are cleaned, not verbatim.** They are lightly
   paraphrased and punctuated by an LLM/human pass, so absolute WER is pessimistic for
   every model. Use WER to *rank* models; language-ID accuracy and real-time factor are
   exact and can be quoted directly.

Additionally the router test split is small (~200 crops, several classes in single
digits), so macro-F1 there is noisy — read it alongside the per-class report.

## Artifacts

Saved under `artifacts/`: scikit-learn pipelines as `.joblib`, PyTorch checkpoints as
`.pt` (with the label list, vocabulary and preprocessing constants embedded in the
payload), fine-tuned transformers as HuggingFace folders loadable with
`AutoModelForSequenceClassification.from_pretrained(...)`, and every ASR hypothesis as
`transcripts_<model>.json`.
