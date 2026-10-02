"""Sinhala/Tamil MT benchmark: NLLB-200 600M vs 1.3B distilled, both CTranslate2 int8 on CPU.

Mirrors production (v3/services/translation/app/translate.py): int8, 4 intra threads, beam 2
into English, beam 4 out of it, no_repeat_ngram_size=4. Scores FLORES-200 devtest with chrF++
(primary, script-agnostic) and BLEU (13a, English targets only).

    .venv\\Scripts\\python src\\mt_benchmark.py [N]
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import ctranslate2
import pandas as pd
import psutil
import sacrebleu
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MODELS = {"nllb-600M-ct2-int8": CACHE / "nllb-600M-ct2", "nllb-1.3B-ct2-int8": CACHE / "nllb-1.3B-ct2"}
PAIRS = [("sin_Sinh", "eng_Latn"), ("tam_Taml", "eng_Latn"), ("eng_Latn", "sin_Sinh"), ("eng_Latn", "tam_Taml")]


def run(model, tok, src, tgt, sents, batch=16):
    tok.src_lang = src
    beam = 2 if tgt == "eng_Latn" else 4
    out = []
    for i in range(0, len(sents), batch):
        toks = [tok.convert_ids_to_tokens(tok.encode(s)) for s in sents[i:i + batch]]
        res = model.translate_batch(toks, target_prefix=[[tgt]] * len(toks), beam_size=beam,
                                    max_decoding_length=256, no_repeat_ngram_size=4)
        out += [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True).strip()
                for r in res]
    return out


def main() -> None:
    df = pd.read_parquet(CACHE / "flores" / "devtest.parquet").head(N)
    proc = psutil.Process()
    rows, samples = [], {}
    for name, path in MODELS.items():
        base = proc.memory_info().rss
        t0 = time.perf_counter()
        model = ctranslate2.Translator(str(path), device="cpu", compute_type="int8", intra_threads=4)
        tok = AutoTokenizer.from_pretrained(str(path))
        load_s, ram_mb = time.perf_counter() - t0, (proc.memory_info().rss - base) / 2**20
        disk_mb = sum(f.stat().st_size for f in path.iterdir()) / 2**20
        for src, tgt in PAIRS:
            srcs, refs = df[src].tolist(), df[tgt].tolist()
            t0 = time.perf_counter()
            hyps = run(model, tok, src, tgt, srcs)
            batch_s = time.perf_counter() - t0
            # Production translates one message at a time: time single sentences too.
            lat = []
            for s in srcs[:20]:
                t1 = time.perf_counter()
                run(model, tok, src, tgt, [s])
                lat.append((time.perf_counter() - t1) * 1000)
            row = {"model": name, "pair": f"{src[:3]}->{tgt[:3]}", "n": len(srcs),
                   "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
                   "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2) if tgt == "eng_Latn" else None,
                   "sent_per_s_batched": round(len(srcs) / batch_s, 2),
                   "p50_ms_single": round(statistics.median(lat)), "p95_ms_single": round(sorted(lat)[18]),
                   "load_s": round(load_s, 1), "ram_mb": round(ram_mb), "disk_mb": round(disk_mb)}
            rows.append(row)
            samples.setdefault(row["pair"], {"source": srcs[:3], "reference": refs[:3]})[name] = hyps[:3]
            print(row, flush=True)
        del model
    res = ROOT / "results"
    (res / "translation_mt.json").write_text(json.dumps({"rows": rows, "samples": samples, "threads": 4,
                                                         "cpu": os.cpu_count()}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    pd.DataFrame(rows).to_csv(res / "summary_translation.csv", index=False)


if __name__ == "__main__":
    main()
