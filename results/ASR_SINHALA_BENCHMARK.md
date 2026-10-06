# Sinhala ASR: whisper-small fine-tunes vs the base model

Source: `src/asr_si_benchmark.py`, run by the `asr-si-benchmark` workflow
([run 37520098701](https://github.com/DSEP-G22/model-testing/actions/runs/37520098701)) on a
4 vCPU AMD EPYC 7763 runner. Rows in `asr_sinhala.json` / `summary_asr_sinhala.csv`.

Decoded as the v3 audio service does (CTranslate2 int8, faster-whisper 1.2.1, greedy, VAD, no
timestamps, one retry at temperature 0.4, language forced to Sinhala). Scored on the first 200
clips (27.5 min) of `SPEAK-ASR/youtube-sinhala-asr` test: conversational YouTube Sinhala, often
code-mixed with English. Punctuation stripped; vowel signs and ZWJ kept.

| Model | WER | CER | Code-mixed WER / CER | RTF |
|---|---:|---:|---:|---:|
| faster-whisper small int8 (base) | 1.202 | 1.174 | 1.144 / 1.022 | 1.81 |
| **SinhaSpeech/whisper-small-sinhala-v6-e6-run11-best @b51781e** | **0.230** | **0.127** | **0.240 / 0.166** | 0.49 |
| dehanns/whisper-small-sinhala-lora-r32-seed-456 (merged) | 0.862 | 0.484 | 0.903 / 0.597 | 0.52 |
| Lingalingeswaran/whisper-small-sinhala_v3 | 0.780 | 0.468 | 0.847 / 0.601 | 0.49 |
| hlasith/whisper-sinhala-small | 0.915 | 0.556 | 0.944 / 0.653 | 0.55 |

* The SinhaSpeech repo is the renamed `Yohan2003/whisper-small-sinhala-run11-v6-e6` (same
  `model.safetensors` SHA-256). It is pinned at `b51781e` because later revisions ship a
  transformers 5 `tokenizer_config.json`.
* run11's training pool includes YouTube audio, so overlap with this test set is possible: read its
  absolute numbers as optimistic. The ranking matches the earlier 10-clip check.
* Base whisper-small loops on forced Sinhala ("අපි අපි අපි ..."), so its error rates are above 1.
* hlasith ships no `preprocessor_config.json`, and its transformers 5 tokenizer config needed
  whisper-small's own vocabulary to convert (see the workflow).

Why dehanns became v3's alternative, and the research behind each choice: `v3/docs/SINHALA-ASR.md`.
