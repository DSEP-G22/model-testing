# Tamil ASR: whisper-small fine-tunes vs the base model

Source: `src/asr_ta_benchmark.py`, run by the `asr-ta-benchmark` workflow
([run 37463770532](https://github.com/DSEP-G22/model-testing/actions/runs/37463770532)) on a
4 vCPU AMD EPYC 7763 runner. Rows in `asr_tamil.json` / `summary_asr_tamil.csv`.

Every model runs the way the v3 audio service runs it: converted to CTranslate2 int8 by
`v3/services/audio/convert.py`'s recipe, faster-whisper 1.2.1, greedy, VAD, no timestamps, one
retry at temperature 0.4, language forced to Tamil. Scored on the first 200 clips of FLEURS
`ta_in` test (43.5 minutes of read speech), punctuation stripped, Tamil vowel signs kept.

| Model | WER | CER | Real-time factor | p50 s / clip |
|---|---:|---:|---:|---:|
| faster-whisper small int8 (base, what v3 ran for Tamil) | 0.785 | 0.277 | 0.57 | 4.0 |
| **vasista22/whisper-tamil-small int8** | **0.282** | **0.100** | 0.50 | 3.8 |
| steja/whisper-small-tamil int8 | 0.429 | 0.116 | 0.54 | 4.0 |
| Lingalingeswaran/whisper-small-ta int8 | 0.461 | 0.160 | 0.55 | 4.0 |

## Reading

* **vasista22 wins on every metric**: WER down 64% and CER down 64% against the base model,
  at the same speed (all four are the same whisper-small architecture).
* **Caveat:** vasista22 and steja both trained on FLEURS train+dev, so this test split is
  in-domain for them (it is held out, not leaked). steja shares that advantage and still trails
  vasista22 by 15 WER points, and vasista22 also reports 7.95 WER on Common Voice 11 test, so the
  ranking holds. FLEURS is Indian Tamil read speech; Sri Lankan Tamil voice notes will score worse
  in absolute terms.
* Lingalingeswaran (Common Voice only) is the weakest fine-tune here.
* The `ram_mb` column in the JSON is only meaningful for the first model (process RSS deltas).

## Candidates not benchmarked

| Model | Why not |
|---|---|
| `BuzzASR/tamil` (whisper-large-v3, 2026) | 1.5B: too slow on the 2 vCPU VPS; its own card reports 37 WER on FLEURS |
| `Achitha/simple_tamil` | no training data or test set named; 14.6 "eval WER" on an unknown set |
| `ai4bharat/indicwhisper` / IndicConformer | not whisper-small; different runtime (NeMo), not a drop-in for faster-whisper |
| Other `*whisper-small-tamil*` repos (Kavin1701, Vengatesan, Syed-Azim-7, ...) | no model card, metrics or data |
