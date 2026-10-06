"""Tamil ASR benchmark: base Whisper small vs Tamil whisper-small fine-tunes, all CTranslate2 int8 on CPU.

Mirrors production (v3/services/audio/app/main.py): faster-whisper, int8, greedy, VAD, no
timestamps, no conditioning on previous text, one retry at temperature 0.4, language forced to
Tamil (the audio service has already picked Tamil before it chooses the model). Scores FLEURS
ta_in test with WER and CER after stripping punctuation; Tamil vowel signs are kept, since
Whisper's BasicTextNormalizer deletes them and would hide real errors.

    python src/asr_ta_benchmark.py [N]

Expects the CT2 builds in data_cache/asr-ta/<name> and FLEURS in data_cache/fleurs-ta/test.parquet
(the asr-ta-benchmark workflow fetches both).
"""

from __future__ import annotations

import io
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd
import psutil
from faster_whisper import WhisperModel, decode_audio

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
#: name -> (CT2 directory or faster-whisper size, source repo @ revision)
MODELS = {
    "faster-whisper-small-int8": ("small", "openai/whisper-small (the base model production runs)"),
    "vasista22-whisper-tamil-small-int8": (CACHE / "asr-ta" / "vasista22", "vasista22/whisper-tamil-small@ac0d71c"),
    "steja-whisper-small-tamil-int8": (CACHE / "asr-ta" / "steja", "steja/whisper-small-tamil@eb8e84c"),
    "lingalingeswaran-whisper-small-ta-int8": (CACHE / "asr-ta" / "lingalingeswaran",
                                               "Lingalingeswaran/whisper-small-ta@c93432c"),
}


def norm(s: str) -> str:
    s = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in s)
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    df = pd.read_parquet(CACHE / "fleurs-ta" / "test.parquet").head(N)
    clips = [decode_audio(io.BytesIO(a["bytes"])) for a in df["audio"]]
    refs = [norm(t) for t in df["transcription"]]
    audio_s = sum(len(c) for c in clips) / 16000
    proc = psutil.Process()
    rows, samples = [], {"reference": df["transcription"].tolist()[:3]}
    for name, (path, source) in MODELS.items():
        base = proc.memory_info().rss
        t0 = time.perf_counter()
        model = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=4)
        load_s, ram_mb = time.perf_counter() - t0, (proc.memory_info().rss - base) / 2**20
        hyps, lat = [], []
        for clip in clips:
            t1 = time.perf_counter()
            segs, _ = model.transcribe(clip, language="ta", vad_filter=True, beam_size=1, without_timestamps=True,
                                       condition_on_previous_text=False, temperature=(0.0, 0.4))
            hyps.append(norm(" ".join(s.text.strip() for s in segs)))
            lat.append(time.perf_counter() - t1)
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        r_, h_ = [p[0] for p in pairs], [p[1] or "<empty>" for p in pairs]
        row = {"model": name, "source": source, "n": len(pairs), "audio_min": round(audio_s / 60, 1),
               "wer": round(jiwer.wer(r_, h_), 4), "cer": round(jiwer.cer(r_, h_), 4),
               "empty": sum(not h for h in hyps), "rtf": round(sum(lat) / audio_s, 3),
               "p50_s_per_clip": round(statistics.median(lat), 2), "load_s": round(load_s, 1), "ram_mb": round(ram_mb)}
        rows.append(row)
        samples[name] = hyps[:3]
        print(row, flush=True)
        del model
    res = ROOT / "results"
    (res / "asr_tamil.json").write_text(json.dumps({"rows": rows, "samples": samples, "threads": 4,
                                                    "cpu": os.cpu_count(), "dataset": "google/fleurs ta_in test"},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(res / "summary_asr_tamil.csv", index=False)


if __name__ == "__main__":
    main()
