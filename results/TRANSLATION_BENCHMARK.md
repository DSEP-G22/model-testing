# Translation models: Sinhala and Tamil

Research and benchmark behind the `nllb-1.3b` option on the `mt_in` / `mt_out` bindings
(October 2026). Benchmark code: `DSEP-G22/model-testing`, `src/mt_benchmark.py`, run by the
`mt-benchmark` workflow ([run 36993759710](https://github.com/DSEP-G22/model-testing/actions/runs/36993759710)).

## What was added

| | |
|---|---|
| Model | [`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B), as the pre-converted CTranslate2 int8 build [`OpenNMT/nllb-200-distilled-1.3B-ct2-int8`](https://huggingface.co/OpenNMT/nllb-200-distilled-1.3B-ct2-int8) (pinned `70f572a`) |
| Binding impl | `nllb-1.3b`, selectable for `mt_in` and `mt_out` on Admin → Models |
| Default | unchanged: `nllb` (600M). The 1.3B loads on the first request that picks it, so it costs no RAM until chosen |
| Fallback | weights missing or failing to load → the binding serves 600M (`services/translation/app/main.py`, tested in `tests/test_bindings.py`) |
| Image | translation image grows by 1.3 GB (`nllb-large` stage in `services/translation/Dockerfile`) |

Same code path as 600M (`CT2NLLBTranslator`): sentence splitting, the LKR → Rs. rewrite, the
"numbers must survive" guard and the Sinhala ZWJ repair all apply unchanged.

## Candidates considered (Hugging Face)

| Model | Size | si | ta | Fit for Lanka Link | Verdict |
|---|---|---|---|---|---|
| NLLB-200 distilled 600M (current) | 0.6B | yes | yes | CT2 int8, 0.6 GB | baseline |
| **NLLB-200 distilled 1.3B** | 1.3B | yes | yes | same tokenizer, same code, CT2 int8 ready-made, 1.3 GB | **added** |
| NLLB-200 3.3B (`OpenNMT/nllb-200-3.3B-ct2-int8`) | 3.3B | yes | yes | ~3.3 GB RAM, ~5x slower than 600M | too heavy for the 2 vCPU / 7.7 GB VPS |
| MADLAD-400 3B MT (`google/madlad400-3b-mt`, CT2 build by Nextcloud-AI) | 3B | yes | yes | T5, ~3 GB int8 | too heavy; NLLB is stronger on FLORES for these two |
| IndicTrans2 (`ai4bharat/indictrans2-*`) | 0.2–1B | **no** | yes | gated, Indic-only | no Sinhala: rules it out |
| M2M-100 1.2B (`facebook/m2m100_1.2B`) | 1.2B | yes | yes | CT2 supported | superseded by NLLB on low-resource pairs |
| TranslateGemma 4B/12B (`google/translategemma-*`, Jan 2026) | 4–27B | 55 langs, si/ta not confirmed (gated card) | | LLM decoding on CPU: seconds per sentence | needs GPU; revisit with `compose.gpu.yaml` |
| SinLlama, Sinhala fine-tunes of Gemma/mT5 | 2–8B | si only | no | low downloads, no Tamil | not production grade |

LLM translation is already covered: `mt_out` falls back to the drafting LLM when translation is
down. A dedicated LLM backend would add per-message cloud cost and latency for less certain
handling of numbers, so a stronger local NMT model was the better next option.

## Benchmark

FLORES-200 devtest ([`mteb/flores`](https://huggingface.co/datasets/mteb/flores)), first 200
sentences per direction. Production settings: CTranslate2 int8, 4 threads, beam 2 into English,
beam 4 out, `no_repeat_ngram_size=4`. chrF++ is the primary metric (works for Sinhala and Tamil
script); BLEU (13a) only for English targets. GitHub runner: AMD EPYC 7763, 4 vCPU (shared, so
absolute latency is pessimistic; the ratio between models is the useful number).

| Direction | chrF++ 600M | chrF++ 1.3B | Δ | BLEU 600M | BLEU 1.3B |
|---|---|---|---|---|---|
| si → en (`mt_in`) | 56.18 | **59.49** | +3.31 | 30.43 | **33.60** |
| ta → en (`mt_in`) | 55.84 | **59.63** | +3.79 | 30.92 | **34.90** |
| en → si (`mt_out`) | 41.40 | **43.04** | +1.64 | – | – |
| en → ta (`mt_out`) | 50.62 | **51.69** | +1.07 | – | – |

| Cost | 600M | 1.3B |
|---|---|---|
| Weights on disk | 621 MB | 1,336 MB |
| RSS after load | ~0.8 GB | ~0.9–1.4 GB (mmap'd; budget 1.4 GB) |
| Load time | 1.1 s | 1.3 s |
| Throughput, batched, into English | 2.5 sent/s | 1.2 sent/s |
| Throughput, batched, out of English | 1.0–1.2 sent/s | 0.55–0.65 sent/s |
| Single sentence p50, si → en | 1.7 s | 3.3 s |

## Reading it

* 1.3B wins every direction. The gain is biggest where it matters most for triage: reading the
  customer in (si/ta → en, +3.3 to +3.8 chrF++, +3 to +4 BLEU).
* It is about 2x slower and needs ~1.4 GB more RAM when picked. On the VPS (2 vCPU) keep 600M as
  the default; switch `mt_in` to `nllb-1.3b` when inbound quality matters more than latency, or
  on the GPU profile where the latency gap closes.
* Outbound (en → si/ta) gains are smaller (+1 to +1.6 chrF++) and outbound sentences are longer,
  so `mt_out` on 600M is the sensible default.
* FLORES is Wikipedia-style text. Customer messages are shorter and noisier, so absolute scores
  will differ; the ranking is what carries over.

Reproduce: Actions → `mt-benchmark` → Run workflow (input `n`), or locally
`python src/mt_benchmark.py 200` after placing the models under `data_cache/`.
