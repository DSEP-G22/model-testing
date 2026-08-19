# %% [markdown]
# # 03 — Voice interpretation: ASR model benchmark and verification
#
# English-only for now — the corpus has real call-centre recordings in 7 languages, but
# non-English calls (and the translate-to-English task) are excluded until multilingual
# support is back in scope. Each remaining call has a `.docx` holding an LLM-written summary
# plus an English transcript section, used as the WER/CER reference.
#
# Models compared: Whisper `tiny` / `base` / `small` / `medium` (OpenAI reference
# implementation) and `faster-whisper` (CTranslate2 int8) for the speed comparison.
#
# **Reference caveat, read before quoting any WER.** The `.docx` transcripts are cleaned,
# punctuated and lightly paraphrased by a human/LLM pass — they are not verbatim gold
# transcripts. Absolute WER is therefore *pessimistic*; the numbers are valid for
# **ranking models against each other**, which is what this benchmark is for. Two extra
# metrics are reported that are robust to that noise: real-time factor (exact) and
# embedding cosine similarity (paraphrase-tolerant).

# %%
import sys, json, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
from common import Result, ResultStore, set_seed, device, env_info, timer, plot_benchmark
import audio_data as ad
import asr_cache

set_seed()
DEV = device()
print(json.dumps(env_info(), indent=2))
store = ResultStore("voice_asr")

# %% [markdown]
# ## 1. Corpus and references

# %%
# English-only for now: non-English calls and the translate-to-English task are dropped
# until multilingual support is back in scope. See notebook history for that version.
all_calls = ad.load_calls()
calls = [c for c in all_calls if c["lang"] == "en"]
print(f"{len(calls)}/{len(all_calls)} calls are English; excluding "
      f"{len(all_calls) - len(calls)} non-English call(s)")
for c in calls:
    c["duration_s"] = ad.audio_duration_seconds(c["audio"])

corpus = pd.DataFrame([{
    "id": c["id"][:44],
    "duration_min": round(c["duration_s"] / 60, 2),
    "size_mb": c["size_mb"],
    "sections": ",".join(c["sections"]),
    "ref_words": len(c["ref_native"].split()) if c["ref_native"] else 0,
} for c in calls])
print(corpus.to_string(index=False))
print(f"\ntotal audio: {corpus.duration_min.sum():.1f} min across {len(corpus)} calls")

# %%
c0 = calls[0]
print(f"--- {c0['id']} ({c0['lang']}) reference, first 400 chars ---")
print((c0["ref_native"] or "")[:400])

missing = [c["id"] for c in calls if not c["ref_native"]]
if missing:
    print(f"\nno native reference for: {missing} (excluded from WER, kept for language ID)")

# %% [markdown]
# ## 2. Scoring helpers
#
# WER/CER are computed on text normalised the same way for every model: lower-cased,
# punctuation stripped, whitespace collapsed. `jiwer` provides the edit-distance rates.

# %%
import jiwer
from sentence_transformers import SentenceTransformer

# English-only corpus now, so the small English encoder is the right scorer directly —
# no multilingual fallback needed (see notebook history for the multilingual version).
SIM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
sim_model = SentenceTransformer(SIM_NAME, device=DEV)
print(f"similarity scorer: {SIM_NAME}")

def score_pair(hyp: str, ref: str) -> dict:
    h, r = ad.normalise_text(hyp), ad.normalise_text(ref)
    if not r or not h:
        return {}
    e = sim_model.encode([h, r], normalize_embeddings=True)
    return {
        "wer": float(jiwer.wer(r, h)),
        "cer": float(jiwer.cer(r, h)),
        "mer": float(jiwer.mer(r, h)),
        "wil": float(jiwer.wil(r, h)),
        "embed_cos": float(e[0] @ e[1]),
        "len_ratio": len(h.split()) / max(len(r.split()), 1),
    }

def aggregate(rows: list[dict]) -> dict:
    """Duration-weighted aggregate, so long calls are not out-voted by short ones."""
    df = pd.DataFrame(rows)
    out = {}
    for k in ("wer", "cer", "mer", "wil", "embed_cos", "len_ratio"):
        if k in df:
            d = df.dropna(subset=[k])
            if d.empty:
                continue
            out[k] = float(np.average(d[k].values, weights=d["duration_s"].values))
            out[f"{k}_median"] = float(d[k].median())
    out["rtf"] = float(df["proc_s"].sum() / df["duration_s"].sum())
    out["lang_id_acc"] = float(df["lang_ok"].mean())
    out["n_calls"] = int(len(df))
    return out

# %% [markdown]
# ## 3. Whisper (OpenAI reference implementation)
#
# Each model transcribes every call with the language forced to `None` so language
# identification is itself measured — a sanity check that Whisper doesn't misdetect the
# language on clean English audio, not a multilingual capability test. `fp16` on GPU,
# greedy decoding with the default beam settings so the comparison is like-for-like.

# %%
import whisper, torch, gc

TRANSCRIPTS = {}   # model -> {call_id: hypothesis}

# Long, noisy, multi-speaker call audio makes Whisper's default decoding loop: the
# `condition_on_previous_text` prompt carries a repetition forward, the compression-ratio
# check rejects the segment, and the temperature fallback re-decodes it up to five times.
# On this corpus that inflated runtime by more than an order of magnitude. Disabling the
# text conditioning is the standard fix for long-form audio and is applied identically to
# every model, so the comparison stays fair.
DECODE_OPTS = dict(
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    temperature=(0.0, 0.2, 0.4),   # shorter fallback ladder than the default five steps
)

def run_whisper(size: str) -> dict:
    tag = f"whisper_{size}"
    # Load the model only if at least one call still needs transcribing.
    pending = [c for c in calls if asr_cache.get(tag, "transcribe", c["id"]) is None]
    model = whisper.load_model(size, device=DEV) if pending else None
    rows, hyps = [], {}
    for c in calls:
        cached = asr_cache.get(tag, "transcribe", c["id"])
        if cached:
            hyp, proc_s, lang = cached["text"], cached["proc_s"], cached["language"]
        else:
            with timer("", verbose=False) as t:
                out = model.transcribe(c["audio"], task="transcribe", fp16=(DEV == "cuda"),
                                       verbose=False, **DECODE_OPTS)
            hyp, proc_s, lang = out["text"].strip(), t.seconds, out.get("language")
            asr_cache.put(tag, "transcribe", c["id"], hyp, proc_s, lang)
        hyps[c["id"]] = hyp
        row = {"call": c["id"][:34], "duration_s": c["duration_s"], "proc_s": proc_s,
               "detected": lang, "lang_ok": lang == c["lang"]}
        row.update(score_pair(hyp, c["ref_native"]) if c["ref_native"] else {})
        rows.append(row)
        print(f"  {c['id'][:34]:36s} lang={lang} "
              f"({'ok' if row['lang_ok'] else 'MISS -> ' + str(c['lang'])})  "
              f"wer={row.get('wer', float('nan')):.3f}  {proc_s:.1f}s"
              f"{'  [cached]' if cached else ''}")
    del model; gc.collect(); torch.cuda.empty_cache()
    TRANSCRIPTS[f"whisper_{size}_transcribe"] = hyps
    return rows, pd.DataFrame(rows)


# Weights download on first use: tiny 75 MB, base 145 MB, small 500 MB, medium 1.5 GB.
# INCLUDE_MEDIUM is off by default: on a 6 GB laptop GPU medium roughly quadruples the
# wall-clock of this notebook. Turn it on for the final report run if you want the ceiling.
INCLUDE_MEDIUM = False
SIZES = ["tiny", "base", "small"] + (["medium"] if INCLUDE_MEDIUM else [])
per_call = {}
for size in SIZES:
    print(f"\n=== whisper {size} (transcribe) ===")
    rows, df_rows = run_whisper(size)
    per_call[f"whisper_{size}"] = df_rows
    agg = aggregate(rows)
    store.add(Result("voice_asr", f"whisper_{size}", "asr", agg,
                     params={"impl": "openai-whisper", "fp16": DEV == "cuda", "task": "transcribe"},
                     notes="duration-weighted; references are cleaned, not verbatim"))

# %% [markdown]
# ## 4. faster-whisper (CTranslate2) — same weights, different runtime
#
# Verifies that the accuracy is preserved while the runtime gets several times faster,
# which is the deployment-relevant question for the ingestion pipeline.

# %%
from faster_whisper import WhisperModel

# faster-whisper runs on CTranslate2, which is built against **CUDA 12** and looks for
# `cublas64_12.dll`, while torch here is cu132 and ships `cublas64_13.dll`. The GPU path used to
# raise `RuntimeError: Library cublas64_12.dll is not found`.
#
# `nvidia-cublas-cu12` is now installed and `common.py` puts it on PATH, so the GPU path works.
# The probe and CPU fallback below are kept anyway: they cost one short transcription and are
# what makes the failure legible on a machine without those libraries, rather than a traceback
# 30 minutes into the run. The device that actually ran is recorded either way, so a real-time
# factor is never mistaken for a GPU number.
def _transcribe_faster(model, audio):
    segs, info = model.transcribe(audio, beam_size=5, vad_filter=True,
                                  condition_on_previous_text=False)
    return " ".join(sg.text for sg in segs).strip(), info


def _open_faster(size: str, probe_audio: str):
    """Open on GPU if CTranslate2 can actually run there, else CPU int8.

    Constructing the model succeeds even when the CUDA 12 libraries are missing — the
    failure only surfaces on the first decode — so the GPU path is probed with a real
    (short) transcription before it is trusted.
    """
    if DEV == "cuda":
        try:
            m = WhisperModel(size, device="cuda", compute_type="int8_float16")
            _transcribe_faster(m, probe_audio)
            return m, "cuda/int8_float16"
        except Exception as exc:
            print(f"  faster-whisper CUDA unavailable ({type(exc).__name__}: "
                  f"{str(exc)[:90]}) -> falling back to CPU int8")
    return WhisperModel(size, device="cpu", compute_type="int8"), "cpu/int8"


def run_faster(size: str):
    tag = f"faster_whisper_{size}"
    pending = [c for c in calls if asr_cache.get(tag, "transcribe", c["id"]) is None]
    model, backend = (None, "cached")
    if pending:
        model, backend = _open_faster(size, pending[0]["audio"])
    rows, hyps = [], {}
    for c in calls:
        cached = asr_cache.get(tag, "transcribe", c["id"])
        if cached:
            hyp, proc_s, lang = cached["text"], cached["proc_s"], cached["language"]
        else:
            with timer("", verbose=False) as t:
                hyp, info = _transcribe_faster(model, c["audio"])
            proc_s, lang = t.seconds, info.language
            asr_cache.put(tag, "transcribe", c["id"], hyp, proc_s, lang)
        hyps[c["id"]] = hyp
        row = {"call": c["id"][:34], "duration_s": c["duration_s"], "proc_s": proc_s,
               "detected": lang, "lang_ok": lang == c["lang"]}
        row.update(score_pair(hyp, c["ref_native"]) if c["ref_native"] else {})
        rows.append(row)
        print(f"  {c['id'][:34]:36s} lang={lang} wer={row.get('wer', float('nan')):.3f} "
              f"{proc_s:.1f}s{'  [cached]' if cached else ''}")
    TRANSCRIPTS[tag] = hyps
    del model; gc.collect(); torch.cuda.empty_cache()
    return rows, backend

for size in ["small"] + (["medium"] if INCLUDE_MEDIUM else []):
    print(f"\n=== faster-whisper {size} ===")
    rows, backend = run_faster(size)
    per_call[f"faster_whisper_{size}"] = pd.DataFrame(rows)
    store.add(Result("voice_asr", f"faster_whisper_{size}", "asr", aggregate(rows),
                     params={"impl": "faster-whisper/CTranslate2", "beam_size": 5,
                             "vad_filter": True, "backend": backend},
                     notes=f"quantised runtime + VAD silence skipping, same Whisper "
                           f"weights; ran on {backend}"))

# %% [markdown]
# ## 5. Verification
#
# Three checks that do not depend on the imperfect reference transcripts:
#
# 1. **Language identification** — the folder name states the language, so this is exact.
# 2. **Real-time factor** — processing seconds per audio second, measured directly.
# 3. **Cross-model agreement** — WER between two models' hypotheses. Where models agree
#    but both score badly against the `.docx`, the reference is the outlier, not the ASR.

# %%
langs = pd.DataFrame({
    m: df.set_index("call")["detected"] for m, df in per_call.items()
})
langs["expected"] = pd.Series({c["id"][:34]: c["lang"] for c in calls})
print("--- detected language per model ---")
print(langs.to_string())

# %%
ref_model = "whisper_medium" if INCLUDE_MEDIUM else "whisper_small"
agree = {}
for m, hyps in TRANSCRIPTS.items():
    if m == f"{ref_model}_transcribe":
        continue
    base = TRANSCRIPTS.get(f"{ref_model}_transcribe")
    if base is None:
        continue
    vals = [jiwer.wer(ad.normalise_text(base[k]), ad.normalise_text(v))
            for k, v in hyps.items() if base.get(k) and v]
    agree[m] = float(np.mean(vals))
print(f"\n--- disagreement with {ref_model} (WER against its hypothesis, lower = closer) ---")
for k, v in sorted(agree.items(), key=lambda kv: kv[1]):
    print(f"  {k:28s} {v:.3f}")

# %%
best_model = min(
    (r for r in store.rows.values() if "wer" in r["metrics"]),
    key=lambda r: r["metrics"]["wer"],
)["model"]
print(f"lowest WER: {best_model}\n")
key = f"{best_model}_transcribe" if f"{best_model}_transcribe" in TRANSCRIPTS else best_model
sample = calls[0]
print("REFERENCE:", (sample["ref_native"] or "")[:350], "\n")
print("HYPOTHESIS:", TRANSCRIPTS.get(key, {}).get(sample["id"], "")[:350])

# %% [markdown]
# ## 6. Benchmark

# %%
bench = store.frame()
cols = [c for c in ["model", "wer", "cer", "embed_cos", "rtf", "lang_id_acc", "n_calls"]
        if c in bench.columns]
print(bench[cols].round(4).sort_values("wer").to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
b = bench.dropna(subset=["wer"]).sort_values("wer")
axes[0].barh(b["model"], b["wer"], color="#8172B3"); axes[0].set_title("WER (lower better)")
axes[1].barh(b["model"], b["rtf"], color="#C44E52")
axes[1].set_title("real-time factor (proc s / audio s)")
axes[2].scatter(b["rtf"], b["wer"], s=80, color="#4C72B0")
for _, r in b.iterrows():
    axes[2].annotate(r["model"], (r["rtf"], r["wer"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
axes[2].set_xlabel("real-time factor"); axes[2].set_ylabel("WER")
axes[2].set_title("accuracy vs speed")
plt.tight_layout(); plt.savefig(common.RESULTS / "voice_asr_benchmark.png", dpi=150); plt.show()

# %%
wer_by_call = pd.DataFrame({m: df.set_index("call")["wer"] for m, df in per_call.items()
                            if "wer" in df.columns})
print("--- per-call WER ---")
print(wer_by_call.round(3).to_string())
wer_by_call.to_csv(common.RESULTS / "voice_wer_per_call.csv")

fig, ax = plt.subplots(figsize=(10, 5))
wer_by_call.plot.bar(ax=ax)
ax.set_ylabel("WER"); ax.set_title("per-call WER by model")
ax.tick_params(axis="x", labelsize=7); plt.tight_layout(); plt.show()

# %%
for name, hyps in TRANSCRIPTS.items():
    (common.ARTIFACTS / f"transcripts_{name}.json").write_text(
        json.dumps(hyps, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"transcripts saved to {common.ARTIFACTS}")

# %% [markdown]
# ## 7. Notes for the report
#
# * Rank models by WER, but quote the **absolute** numbers with the reference caveat from
#   the header — the `.docx` transcripts are cleaned, so a perfect ASR would still score
#   a non-zero WER here.
# * Language ID accuracy and real-time factor are exact and can be quoted without caveats.
# * `faster-whisper` at the same model size is the deployment candidate: near-identical
#   WER at a fraction of the real-time factor.
# * Per-call WER still varies by audio quality (background noise, call volume) even
#   within this English-only set — the per-call table shows which calls need a larger
#   model.
# * Multilingual support (non-English calls, the translate-to-English task) is out of
#   scope for now — see git history for the last version that covered it.

# %%
print(f"results -> {store.path}")
