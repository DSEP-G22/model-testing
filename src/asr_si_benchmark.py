"""Sinhala ASR benchmark: whisper-small Sinhala fine-tunes vs the base model, all CTranslate2 int8 on CPU.

Same decode as production (v3/services/audio/app/main.py) and as asr_ta_benchmark.py: faster-whisper
int8, greedy, VAD, no timestamps, no conditioning, one retry at temperature 0.4, language forced to
Sinhala. Scored on SPEAK-ASR/youtube-sinhala-asr test: conversational YouTube Sinhala, often mixed
with English, which is closer to a customer's voice note than read speech. WER and CER after
stripping punctuation; Sinhala vowel signs and ZWJ are kept.

    python src/asr_si_benchmark.py [N]

Expects CT2 builds in data_cache/asr-si/<name> and the test parquet in data_cache/yt-si/test.parquet
(the asr-si-benchmark workflow fetches both).
"""

from __future__ import annotations

import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import jiwer
import pandas as pd
from faster_whisper import WhisperModel, decode_audio

from asr_ta_benchmark import norm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
A = CACHE / "asr-si"
#: name -> (CT2 directory or faster-whisper size, source repo @ revision, what backs it)
MODELS = {
    "faster-whisper-small-int8": ("small", "openai/whisper-small", "Radford et al. 2023 (Whisper)"),
    "sinhaspeech-run11-int8": (A / "run11", "SinhaSpeech/whisper-small-sinhala-v6-e6-run11-best@8e79fac",
                               "model card: 16.38 WER / 4.43 CER, 15,860-clip speaker-disjoint test"),
    "dehanns-lora-r32-s456-int8": (A / "dehanns", "dehanns/whisper-small-sinhala-lora-r32-seed-456@3d1a454 (merged)",
                                   "model card: 48.6 WER / 12.6 CER, 4,326-clip test; rank/seed study"),
    "lingalingeswaran-v3-int8": (A / "linga", "Lingalingeswaran/whisper-small-sinhala_v3@cf54883",
                                 "model card: 46.5 WER"),
    "hlasith-int8": (A / "hlasith", "hlasith/whisper-sinhala-small@21155f6", "model card: 61.7 WER / 14.9 CER, 10 clips"),
}


def main() -> None:
    df = pd.read_parquet(CACHE / "yt-si" / "test.parquet").head(N)
    clips = [decode_audio(io.BytesIO(a["bytes"])) for a in df["audio"]]
    refs = [norm(t) for t in df["text"]]
    audio_s = sum(len(c) for c in clips) / 16000
    mixed = df["is_code_mixed"].fillna(False).to_numpy()
    rows, samples = [], {"reference": df["text"].tolist()[:3]}
    for name, (path, source, backing) in MODELS.items():
        t0 = time.perf_counter()
        model = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=4)
        load_s = time.perf_counter() - t0
        hyps, lat = [], []
        for clip in clips:
            t1 = time.perf_counter()
            segs, _ = model.transcribe(clip, language="si", vad_filter=True, beam_size=1, without_timestamps=True,
                                       condition_on_previous_text=False, temperature=(0.0, 0.4))
            hyps.append(norm(" ".join(s.text.strip() for s in segs)))
            lat.append(time.perf_counter() - t1)

        def score(mask):
            r = [x for x, h, m in zip(refs, hyps, mask) if x and m]
            h = [h or "<empty>" for x, h, m in zip(refs, hyps, mask) if x and m]
            return (round(jiwer.wer(r, h), 4), round(jiwer.cer(r, h), 4)) if r else (None, None)

        wer, cer = score([True] * len(refs))
        cm_wer, cm_cer = score(mixed)
        row = {"model": name, "source": source, "backing": backing, "n": len(refs), "audio_min": round(audio_s / 60, 1),
               "wer": wer, "cer": cer, "code_mixed_wer": cm_wer, "code_mixed_cer": cm_cer,
               "empty": sum(not h for h in hyps), "rtf": round(sum(lat) / audio_s, 3),
               "p50_s_per_clip": round(statistics.median(lat), 2), "load_s": round(load_s, 1)}
        rows.append(row)
        samples[name] = hyps[:3]
        print(row, flush=True)
        del model
    res = ROOT / "results"
    (res / "asr_sinhala.json").write_text(json.dumps({"rows": rows, "samples": samples, "cpu": os.cpu_count(),
                                                      "dataset": "SPEAK-ASR/youtube-sinhala-asr test"},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(res / "summary_asr_sinhala.csv", index=False)


if __name__ == "__main__":
    main()
