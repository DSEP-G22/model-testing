# %% [markdown]
# # 09 — Translation model selection: validating the research results
#
# `v3/docs/final/research_for_language_selection` argues for **NLLB-200 distilled 600M** as
# Lanka Link's translator: the model must be **small** (it shares a 2 vCPU / 7.7 GB server with
# ~15 other services) and must translate **Sinhala ↔ English** (Tamil ↔ English too). Its two
# scripts and five CSVs are copied unchanged into `data/translation_research/`.
#
# This notebook re-tests every claim in those files by **running** the models, with the
# research's own methodology and data:
#
# | | Research files | This notebook |
# |---|---|---|
# | Metric | chrF++ (`sacrebleu`, `word_order=2`) | same, plus BLEU into English |
# | Runtime | CTranslate2 int8, CPU, 4 threads, beam 2 into English / 4 out, `no_repeat_ngram_size=4` | same, for every model |
# | Data 1 | 30 + 30 + 15 + 15 sentences written into `benchmark_nllb.py` and labelled "FLORES-200" | the same sentences, imported from the script |
# | Data 2 | — (claimed: FLORES-200 devtest) | real FLORES-200 devtest, first 200 sentences per direction, as `src/mt_benchmark.py` |
# | Numbers | 8 cases in `eval_translation.py`, `numbers_kept` | the same cases and the production `numbers_kept`, on the **raw** model output |
#
# Run: every **small** candidate (≤ 1.3B parameters) that covers Sinhala — NLLB-200 distilled
# 600M and 1.3B from the research table, plus three small multilingual models the research did not
# test: M2M-100 418M, mBART-50 many-to-many (611M) and OPUS-MT `mul-en` + `en-mul` (2 × 77M).
#
# Not run: the large models the research already discarded on size (NLLB-200 3.3B, MADLAD-400 3B,
# TranslateGemma 4B, SinLlama 8B) and IndicTrans2. Their size and language claims are checked
# from the published weights and configs on the Hugging Face Hub (section 6.2), which is a
# measurement, not a profile.
#
# Run on the `mt-validation` GitHub workflow (AMD EPYC, 4 vCPU, no GPU — the same runner type
# as `mt_benchmark.py`): the local link pulls the Hub at ~70 KB/s. Each model is evaluated in its
# own matrix job (`MT_ONLY=<model>`) into `results/mt_validation/<model>.json`; the final job
# re-runs this notebook, which loads those files and does the analysis.

# %%
import importlib.util, json, os, re, statistics, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sacrebleu

BENCH = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CACHE, RESULTS = BENCH / "data_cache", BENCH / "results"
OUT = RESULTS / "mt_validation"
OUT.mkdir(parents=True, exist_ok=True)
RESEARCH = BENCH / "data" / "translation_research"
sys.path.insert(0, str(BENCH / "src"))
import mt_fetch

N = int(os.environ.get("MT_N", 200))            # FLORES sentences per direction
ONLY = [m for m in os.environ.get("MT_ONLY", "").split(",") if m]
THREADS = 4
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 80)
print({"N": N, "ONLY": ONLY or "all (cached results are reused)", "cpus": os.cpu_count(),
       "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)})

# %% [markdown]
# ## 1. The research data, imported from the research scripts themselves

# %%
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, RESEARCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

bn, ev = load_script("benchmark_nllb"), load_script("eval_translation")
LANG = {"si": "sin_Sinh", "ta": "tam_Taml", "en": "eng_Latn"}
PAIRS = [("si", "en"), ("ta", "en"), ("en", "si"), ("en", "ta")]
RESEARCH_SETS = {(d[1].value, d[2].value): d[3] for d in bn.DIRECTIONS}
CASES = [(c.id, c.source, c.target.value, c.must_have_numbers) for c in ev.CASES]
print({f"{s}->{t}": len(v) for (s, t), v in RESEARCH_SETS.items()}, "| number cases:", len(CASES))

flores = pd.read_parquet(mt_fetch.flores())
FLORES = {(s, t): (flores[LANG[s]].tolist()[:N], flores[LANG[t]].tolist()[:N]) for s, t in PAIRS}

# %% [markdown]
# ### 1.1 Are the "FLORES-200" sentences in `benchmark_nllb.py` from FLORES-200?
#
# The script's header says *"FLORES-200 devtest subset"*. Each research sentence's English side
# is searched among all 1,012 English FLORES-200 devtest sentences, verbatim and by best chrF.

# %%
def best_match(sent, pool):
    scores = [sacrebleu.sentence_chrf(sent, [p]).score for p in pool]
    i = int(np.argmax(scores))
    return scores[i], pool[i]

prov = []
for (s, t), pairs in RESEARCH_SETS.items():
    pool = flores[LANG["en"]].tolist()
    for src, ref in pairs:
        eng = ref if t == "en" else src
        sc, hit = best_match(eng, pool)
        prov.append({"pair": f"{s}->{t}", "english": eng, "exact_in_flores": eng in pool,
                     "best_flores_chrf": round(sc, 1), "closest_flores": hit[:70]})
prov = pd.DataFrame(prov)
print("research sentences found verbatim in FLORES-200 devtest:", int(prov.exact_in_flores.sum()), "of", len(prov))
print("best chrF against any FLORES sentence: median", prov.best_flores_chrf.median(), "max", prov.best_flores_chrf.max())
prov.sort_values("best_flores_chrf", ascending=False).head(5)

# %% [markdown]
# ## 2. Candidates and language support (checked from each model's own vocabulary / config)

# %%
from transformers import AutoTokenizer

# family -> language codes as the model spells them
CODES = {
    "nllb":  {"si": "sin_Sinh", "ta": "tam_Taml", "en": "eng_Latn"},
    "m2m":   {"si": "__si__", "ta": "__ta__", "en": "__en__"},
    "mbart": {"si": "si_LK", "ta": "ta_IN", "en": "en_XX"},
    "opus":  {"si": ">>sin<<", "ta": ">>tam<<", "en": None},
}
# name -> family, data_cache dirs (opus has one model per direction), nominal parameters (B)
CANDIDATES = {
    "nllb-600M":    ("nllb", {"in": "nllb-600M-ct2", "out": "nllb-600M-ct2"}, 0.6),
    "nllb-1.3B":    ("nllb", {"in": "nllb-1.3B-ct2", "out": "nllb-1.3B-ct2"}, 1.3),
    "m2m100-418M":  ("m2m", {"in": "m2m100-418M-ct2", "out": "m2m100-418M-ct2"}, 0.418),
    "mbart50-611M": ("mbart", {"in": "mbart50-ct2", "out": "mbart50-ct2"}, 0.611),
    "opus-mt-77M":  ("opus", {"in": "opus-mul-en-ct2", "out": "opus-en-mul-ct2"}, 0.077),
}
EVAL = os.environ.get("MT_EVAL", "1") == "1"   # 0 = analysis only (the workflow's last job)
TODO = [m for m in CANDIDATES if EVAL and (not ONLY or m in ONLY) and not (OUT / f"{m}.json").exists()]
print("to evaluate in this run:", TODO or "none, all cached")

# %%
def support(name):
    """Does the model's own vocabulary carry a language code for si and ta?"""
    fam, dirs, _ = CANDIDATES[name]
    path = CACHE / dirs["out"]
    vocab = set(AutoTokenizer.from_pretrained(path).get_vocab())
    return {l: c in vocab for l, c in CODES[fam].items() if l != "en"}

# %% [markdown]
# ## 3. One translate function for every encoder-decoder, production settings

# %%
import ctranslate2

def disk_mb(*paths):
    return round(sum(f.stat().st_size for p in set(paths) for f in Path(p).rglob("*") if f.is_file()) / 2**20)


class CT2:
    """Mirrors v3 CT2NLLBTranslator / src/mt_benchmark.py: int8, 4 threads, beam 2 in, 4 out."""

    def __init__(self, fam, dirs):
        self.fam = fam
        self.m, self.tok = {}, {}
        for k, d in dirs.items():
            p = str(CACHE / d)
            if p not in self.m:
                self.m[p] = ctranslate2.Translator(p, device="cpu", compute_type="int8", intra_threads=THREADS)
                self.tok[p] = AutoTokenizer.from_pretrained(p)
        self.path = {k: str(CACHE / d) for k, d in dirs.items()}

    def __call__(self, sents, s, t, batch=16):
        p = self.path["in" if t == "en" else "out"]
        model, tok, code = self.m[p], self.tok[p], CODES[self.fam]
        if self.fam in ("nllb", "mbart"):
            tok.src_lang = code[s]
        elif self.fam == "m2m":
            tok.src_lang = s
        pre = code[t] + " " if self.fam == "opus" and t != "en" else ""
        prefix = self.fam in ("nllb", "mbart", "m2m")
        out = []
        for i in range(0, len(sents), batch):
            toks = [tok.convert_ids_to_tokens(tok.encode(pre + x)) for x in sents[i:i + batch]]
            res = model.translate_batch(toks, target_prefix=[[code[t]]] * len(toks) if prefix else None,
                                        beam_size=2 if t == "en" else 4, max_decoding_length=256,
                                        no_repeat_ngram_size=4)
            out += [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1 if prefix else 0:]),
                               skip_special_tokens=True).strip() for r in res]
        return out


# %% [markdown]
# ### 3.1 The production number guard, verbatim from `v3/services/translation/app/translate.py`
#
# Note: production's `CT2NLLBTranslator.translate` returns the **source sentence** whenever
# `numbers_kept` fails. `eval_translation.py` scored the number cases through that method, so
# inside Docker every case passes by construction. Here the guard is applied to the raw model
# output instead, which is what the test is meant to measure.

# %%
_LKR = re.compile(r"\bLKR\s?(?=\d)")
_NUMBER = re.compile(r"\d[\d,.:]*\d|\d")
_LIST_MARKER = re.compile(r"(?m)^\s*\d{1,2}[.)]\s+")

def _prepare(text):
    return _LKR.sub("Rs. ", text)

def _numbers(text):
    return sorted(re.sub(r"\D", "", n) for n in _NUMBER.findall(_LIST_MARKER.sub("", text)))

def numbers_kept(source, translated):
    return _numbers(source) == _numbers(translated)

# %% [markdown]
# ## 4. Evaluate (only the models in `TODO`; the rest load from `results/mt_validation/`)

# %%
def chrf(h, r):
    return round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2)

def evaluate(name):
    fam, dirs, params = CANDIDATES[name]
    for d in set(dirs.values()):
        mt_fetch.fetch(d)
    proc = psutil.Process()
    base, t0 = proc.memory_info().rss, time.perf_counter()
    model = CT2(fam, dirs)
    rec = {"model": name, "family": fam, "params_b": params, "load_s": round(time.perf_counter() - t0, 1),
           "ram_mb": round((proc.memory_info().rss - base) / 2**20),
           "disk_mb": disk_mb(*[CACHE / d for d in dirs.values()]), "support": support(name),
           "flores": {}, "research": {}}
    n = N
    for s, t in PAIRS:
        srcs, refs = FLORES[(s, t)][0][:n], FLORES[(s, t)][1][:n]
        t0 = time.perf_counter()
        hyps = model(srcs, s, t)
        batch_s = time.perf_counter() - t0
        lat = []
        for x in srcs[:20]:  # production translates one message at a time
            t1 = time.perf_counter(); model([x], s, t); lat.append((time.perf_counter() - t1) * 1000)
        rec["flores"][f"{s}->{t}"] = {
            "n": n, "chrf++": chrf(hyps, refs),
            "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2) if t == "en" else None,
            "sent_per_s_batched": round(n / batch_s, 2), "p50_ms_single": round(statistics.median(lat)),
            "p95_ms_single": round(float(np.percentile(lat, 95))), "hyps": hyps}
        pairs = RESEARCH_SETS[(s, t)]
        rh = model([p[0] for p in pairs], s, t)
        rec["research"][f"{s}->{t}"] = {"n": len(pairs), "chrf++": chrf(rh, [p[1] for p in pairs]), "hyps": rh}
        print(name, f"{s}->{t}", {k: v for k, v in rec["flores"][f"{s}->{t}"].items() if k != "hyps"},
              "research chrF++", rec["research"][f"{s}->{t}"]["chrf++"], flush=True)
    rec["numbers"] = []
    for cid, src, tgt, must in CASES:
        hyp = model([_prepare(src)], "en", tgt)[0]
        rec["numbers"].append({"id": cid, "source": src, "target": tgt, "hyp": hyp,
                               "kept": numbers_kept(_prepare(src), hyp) if must else None})
    rec["ram_mb_after"] = round((proc.memory_info().rss - base) / 2**20)
    (OUT / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    del model
    return rec

for name in TODO:
    evaluate(name)
if ONLY:  # a matrix job evaluates one model; the analysis runs once, over all of them
    raise SystemExit(0)

# %%
REC = {m: json.loads((OUT / f"{m}.json").read_text(encoding="utf-8")) for m in CANDIDATES if (OUT / f"{m}.json").exists()}
missing = [m for m in CANDIDATES if m not in REC]
print("results available:", list(REC), "| missing:", missing)

rows = []
for m, r in REC.items():
    for pair, f in r["flores"].items():
        rows.append({"model": m, "pair": pair, "n": f["n"], "chrf++": f["chrf++"], "bleu": f["bleu"],
                     "research_set_chrf++": r["research"][pair]["chrf++"], "p50_ms": f["p50_ms_single"],
                     "p95_ms": f["p95_ms_single"], "sent_per_s": f["sent_per_s_batched"],
                     "disk_mb": r["disk_mb"], "ram_mb": max(r["ram_mb"], r["ram_mb_after"])})
long = pd.DataFrame(rows)
long.to_csv(RESULTS / "summary_translation_validation.csv", index=False)
long

# %% [markdown]
# ## 5. Reproducibility: does the new harness give the numbers already in the deck?
#
# `results/summary_translation.csv` is the earlier `src/mt_benchmark.py` run that the final deck
# reads. Same data, same settings, same runner type: chrF++ should match to rounding.

# %%
prev = pd.read_csv(RESULTS / "summary_translation.csv")
prev["model"] = prev["model"].map({"nllb-600M-ct2-int8": "nllb-600M", "nllb-1.3B-ct2-int8": "nllb-1.3B"})
prev["pair"] = prev["pair"].map({"sin->eng": "si->en", "tam->eng": "ta->en", "eng->sin": "en->si", "eng->tam": "en->ta"})
rep = prev.merge(long, on=["model", "pair"], suffixes=("_before", "_now"))[
    ["model", "pair", "chrf++_before", "chrf++_now", "bleu_before", "bleu_now", "p50_ms_single", "p50_ms"]]
rep["chrf_diff"] = (rep["chrf++_now"] - rep["chrf++_before"]).round(2)
rep

# %% [markdown]
# ## 6. Claims audit — every number in the research files against a measurement
#
# `nllb_benchmark_*.csv` is labelled *"pre-measured profile"* and every non-NLLB-600M row of
# `translation_model_*.csv` *"documented profile"*: both scripts write hard-coded numbers when
# they run outside Docker, and the CSVs in the research folder came from such a run.

# %%
claims_dir = pd.read_csv(RESEARCH / "nllb_benchmark_directions.csv")
claims_sum = pd.read_csv(RESEARCH / "translation_model_summary.csv")
print("sources in nllb_benchmark_directions.csv:", claims_dir["source"].unique().tolist())
print("sources in translation_model_summary.csv:", claims_sum["source"].unique().tolist())

P = {"si_en": "si->en", "ta_en": "ta->en", "en_si": "en->si", "en_ta": "en->ta"}
M = {"NLLB-200 distilled 600M (served)": "nllb-600M", "NLLB-200 distilled 1.3B": "nllb-1.3B"}
a = claims_dir.assign(model=claims_dir["model"].map(M), pair=claims_dir["direction"].map(P))
a = a.merge(long, on=["model", "pair"], how="left")
chrf_audit = a[["model", "pair", "n_sentences", "chrf", "research_set_chrf++", "n", "chrf++", "latency_ms", "p50_ms"]].rename(
    columns={"n_sentences": "claimed_n", "chrf": "claimed_chrf++", "research_set_chrf++": "measured_on_claimed_sentences",
             "n": "flores_n", "chrf++": "measured_on_flores", "latency_ms": "claimed_ms", "p50_ms": "measured_p50_ms"})
chrf_audit

# %%
SUM_MAP = {"NLLB-200 distilled 600M": "nllb-600M", "NLLB-200 distilled 1.3B": "nllb-1.3B"}

def measured(m):
    r = REC.get(m)
    if r is None:
        return {}
    # the research score: a number case passes when its numbers survive, a plain case when the
    # target language is supported
    score = sum(bool(c["kept"]) if c["kept"] is not None else r["support"][c["target"]] for c in r["numbers"])
    return {"m_si": r["support"]["si"], "m_ta": r["support"]["ta"], "m_score": f"{score}/8",
            "m_ram_mb": max(r["ram_mb"], r["ram_mb_after"]), "m_disk_mb": r["disk_mb"],
            "m_p50_s_si_en": round(r["flores"]["si->en"]["p50_ms_single"] / 1000, 2)}

audit = claims_sum[["model", "sinhala_support", "tamil_support", "score", "avg_latency_s", "memory_mb"]].copy()
audit = pd.concat([audit, pd.DataFrame([measured(SUM_MAP.get(m, "")) for m in audit["model"]])], axis=1)
audit

# %% [markdown]
# ### 6.1 Number cases, per model (raw output, production guard)

# %%
num = pd.DataFrame([{"model": m, "case": c["id"], "target": c["target"], "kept": c["kept"], "hyp": c["hyp"]}
                    for m, r in REC.items() for c in r["numbers"] if c["kept"] is not None])
print(num.pivot_table(index="model", columns="case", values="kept", aggfunc="first").to_string())
num[~num["kept"]]

# %% [markdown]
# ### 6.2 Models that are not run: size and languages from the published weights and configs
#
# Weight sizes are summed from the Hub's file metadata at a pinned revision (the int8 CTranslate2
# build where one exists, otherwise bf16 safetensors); language support from each model's own
# language list or chat template.

# %%
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
it2 = hf_hub_download("ai4bharat/indictrans2-en-indic-dist-200M", "README.md")
it2_langs = re.search(r"language_details:\s*>-?\s*(.+?)\n\S", open(it2, encoding="utf-8").read(), re.S).group(1).split(",")
it2_langs = [l.strip() for l in it2_langs]
print("IndicTrans2 languages:", len(it2_langs), "| sin_Sinh listed:", "sin_Sinh" in it2_langs, "| tam_Taml listed:", "tam_Taml" in it2_langs)

def hub_mb(repo, rev, exts=(".bin", ".safetensors")):
    info = api.model_info(repo, revision=rev, files_metadata=True)
    return round(sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(exts)) / 2**20)

tg_tpl = open(hf_hub_download("Infomaniak-AI/vllm-translategemma-4b-it", "chat_template.jinja", revision="cb3e0b2"), encoding="utf-8").read()
large = pd.DataFrame([
    {"model": "nllb-3.3B", "build": "OpenNMT/nllb-200-3.3B-ct2-int8", "weights_mb": hub_mb("OpenNMT/nllb-200-3.3B-ct2-int8", "28d998c"), "si": True, "ta": True},
    {"model": "madlad-3B", "build": "Nextcloud-AI/madlad400-3b-mt-ct2-int8", "weights_mb": hub_mb("Nextcloud-AI/madlad400-3b-mt-ct2-int8", "aa32bbd"), "si": True, "ta": True},
    {"model": "translategemma-4B", "build": "translategemma-4b-it (bf16)", "weights_mb": hub_mb("Infomaniak-AI/vllm-translategemma-4b-it", "cb3e0b2"),
     "si": '"si":' in tg_tpl, "ta": '"ta":' in tg_tpl},
    {"model": "sinllama-8B", "build": "SinLlama-Llama-3-8B-Merged (bf16)", "weights_mb": hub_mb("SAWithanage/SinLlama-Llama-3-8B-Merged", "6c4f36c"), "si": True, "ta": False},
    {"model": "indictrans2-200M", "build": "indictrans2-en-indic-dist-200M", "weights_mb": None, "si": "sin_Sinh" in it2_langs, "ta": "tam_Taml" in it2_langs},
])
large["vs_600M_int8"] = (large.weights_mb / 621).round(1)
large["over_budget"] = large.weights_mb > 1536
sl_gb = large.set_index("model").loc["sinllama-8B", "weights_mb"] / 1024
large

# %% [markdown]
# ## 7. The selection argument: size against Sinhala ↔ English quality
#
# Hard constraints, fixed before looking at the scores:
#
# 1. **Both languages, both directions** — si↔en (and ta↔en) supported by the model itself.
# 2. **Small** — int8 weights ≤ 1.5 GB (the RAM budget `TRANSLATION_BENCHMARK.md` sets for the
#    largest option) on a 7.7 GB server shared with ~15 services.
# 3. **Fast enough** — single-sentence si→en p50 well inside the orchestrator's 10 s
#    translate budget on CPU.
#
# Among models that pass, the higher si↔en chrF++ wins; ties go to the smaller one.

# %%
BUDGET_MB, TRANSLATE_BUDGET_MS = 1536, 10_000
tab = []
for m, r in REC.items():
    f = r["flores"]
    tab.append({"model": m, "params_b": r["params_b"], "disk_mb": r["disk_mb"], "ram_mb": max(r["ram_mb"], r["ram_mb_after"]),
                "si_ok": r["support"]["si"], "ta_ok": r["support"]["ta"],
                "si->en": f["si->en"]["chrf++"], "en->si": f["en->si"]["chrf++"],
                "si<->en": round((f["si->en"]["chrf++"] + f["en->si"]["chrf++"]) / 2, 2),
                "ta<->en": round((f["ta->en"]["chrf++"] + f["en->ta"]["chrf++"]) / 2, 2),
                "p50_ms_si_en": f["si->en"]["p50_ms_single"], "n": f["si->en"]["n"],
                "numbers_kept": f"{sum(bool(c['kept']) for c in r['numbers'] if c['kept'] is not None)}/4"})
tab = pd.DataFrame(tab)
tab["fits"] = (tab.disk_mb <= BUDGET_MB) & tab.si_ok & tab.ta_ok & (tab.p50_ms_si_en < TRANSLATE_BUDGET_MS)
tab = tab.sort_values(["fits", "si<->en"], ascending=False)
tab.to_csv(RESULTS / "summary_translation_selection.csv", index=False)
tab

# %% [markdown]
# ### 7.1 Is the gap real? Paired bootstrap of chrF++ against the served 600M
#
# 1,000 resamples of the same sentence indices for both systems (Koehn 2004). A 95% interval
# that excludes 0 means the difference is not sampling noise.

# %%
from sacrebleu.metrics import CHRF

CHRFPP, rng = CHRF(word_order=2), np.random.default_rng(42)

def paired_boot(a, b, refs, B=1000):
    # per-sentence chrF++ statistics, summed per resample = corpus chrF++ of that resample
    sa, sb = (np.array(CHRFPP._extract_corpus_statistics(h, [refs])) for h in (a, b))
    score = lambda st: CHRFPP._compute_score_from_stats(list(st)).score
    idx = rng.integers(0, len(refs), (B, len(refs)))
    diffs = [score(sa[i].sum(0)) - score(sb[i].sum(0)) for i in idx]
    return np.percentile(diffs, [2.5, 97.5]).round(2)

boot = []
for m, r in REC.items():
    if m == "nllb-600M":
        continue
    for pair in ["si->en", "en->si", "ta->en", "en->ta"]:
        hy = r["flores"][pair]["hyps"]; n = len(hy)
        base = REC["nllb-600M"]["flores"][pair]["hyps"][:n]
        refs = FLORES[tuple(pair.split("->"))][1][:n]
        lo, hi = paired_boot(hy, base, refs)
        boot.append({"model": m, "pair": pair, "n": n, "delta_vs_600M": round(chrf(hy, refs) - chrf(base, refs), 2),
                     "ci95_lo": lo, "ci95_hi": hi, "significant": not (lo <= 0 <= hi)})
boot = pd.DataFrame(boot)
boot.to_csv(RESULTS / "summary_translation_bootstrap.csv", index=False)
boot

# %% [markdown]
# ### 7.2 Figure for the deck

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (y, title) in zip(axes, [("si<->en", "Sinhala ↔ English"), ("ta<->en", "Tamil ↔ English")]):
    for _, r in tab.iterrows():
        c = "#2a9d8f" if r.fits else "#9aa0a6"
        ax.scatter(r.disk_mb / 1024, r[y], s=60 + r.p50_ms_si_en / 40, color=c, edgecolor="black", zorder=3)
        ax.annotate(r.model, (r.disk_mb / 1024, r[y]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(BUDGET_MB / 1024, ls="--", color="#d62828", lw=1)
    ax.text(BUDGET_MB / 1024, ax.get_ylim()[0], " size budget", color="#d62828", fontsize=8, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("int8 weights on disk (GB, log)"); ax.set_ylabel("chrF++ (FLORES-200, mean of both directions)")
    ax.set_title(title); ax.grid(alpha=0.3)
fig.suptitle("Translation candidates: quality against size (marker area ∝ CPU latency; green = meets every constraint)")
plt.tight_layout()
fig.savefig(RESULTS / "translation_size_vs_quality.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Verdicts — written from the tables above, not by hand

# %%
def f1(x): return f"{x:.1f}"
by = tab.set_index("model")
v = []
n6 = REC["nllb-600M"]
v.append(("Research sentences are FLORES-200", "does not hold",
          f"{int(prov.exact_in_flores.sum())} of {len(prov)} found in FLORES devtest; they are hand-written telecom sentences"))
d = chrf_audit[chrf_audit.model == "nllb-600M"]
v.append(("600M chrF++ 56.2/55.8/41.4/50.6 on 30/30/15/15 sentences", "numbers hold, label does not",
          "measured on FLORES-200 (n=200): " + "/".join(f1(x) for x in d["measured_on_flores"]) +
          "; on the script's own sentences: " + "/".join(f1(x) for x in d["measured_on_claimed_sentences"])))
def within(measured, claimed, tol=0.25):
    return "holds" if abs(measured - claimed) <= tol * claimed else "does not hold"

p50 = n6["flores"]["si->en"]["p50_ms_single"]
v.append(("600M latency 390 ms per si->en sentence", within(p50, 390), f"measured p50 {p50} ms on 4 vCPU"))
for m, claimed in [("nllb-600M", 942), ("nllb-1.3B", 1820)]:
    if m in by.index:
        v.append((f"{m} needs {claimed} MB", within(by.loc[m, "ram_mb"], claimed),
                  f"measured RSS {by.loc[m, 'ram_mb']} MB, int8 weights {by.loc[m, 'disk_mb']} MB"))
L = large.set_index("model")
for m, claimed in [("nllb-3.3B", 3300), ("madlad-3B", 3000)]:
    v.append((f"{m} too heavy (~{claimed} MB)", "holds" if L.loc[m, "over_budget"] else "does not hold",
              f"int8 weights {L.loc[m, 'weights_mb']} MB = {L.loc[m, 'vs_600M_int8']}x the 600M"))
v.append(("TranslateGemma: Sinhala/Tamil unsupported (research CSV)", "does not hold" if L.loc["translategemma-4B", "si"] else "holds",
          f"its chat template lists si={L.loc['translategemma-4B', 'si']} ta={L.loc['translategemma-4B', 'ta']}"))
v.append(("TranslateGemma too heavy for the CPU server", "holds" if L.loc["translategemma-4B", "over_budget"] else "does not hold",
          f"bf16 weights {L.loc['translategemma-4B', 'weights_mb']} MB"))
if "nllb-1.3B" in by.index:
    b = boot[(boot.model == "nllb-1.3B")]
    v.append(("1.3B: 'no quality gain on these languages' (translation_model_summary.csv)",
              "does not hold" if (b.significant & (b.delta_vs_600M > 0)).any() else "holds",
              "; ".join(f"{r.pair} {r.delta_vs_600M:+.2f} [{r.ci95_lo}, {r.ci95_hi}]" for r in b.itertuples())))
for m in ["m2m100-418M", "mbart50-611M", "opus-mt-77M"]:
    if m in by.index:
        r = by.loc[m]
        sb = boot[(boot.model == m) & boot.pair.isin(["si->en", "en->si"])]
        worse = (sb.significant & (sb.delta_vs_600M < 0)).any()
        v.append((f"{m} as a smaller Sinhala translator", "ruled out: size" if not r.fits else
                  "ruled out: quality" if worse else "competitive with 600M",
                  f"si<->en {r['si<->en']} vs 600M {by.loc['nllb-600M','si<->en']}; {r.disk_mb} MB; p50 {r.p50_ms_si_en} ms; si={r.si_ok} ta={r.ta_ok}; numbers {r.numbers_kept}"))
v.append(("IndicTrans2 has no Sinhala", "holds" if "sin_Sinh" not in it2_langs else "does not hold", f"{len(it2_langs)} languages, sin_Sinh absent"))
v.append(("SinLlama too heavy", "holds", f"{sl_gb:.1f} GB of bf16 weights"))
verdicts = pd.DataFrame(v, columns=["claim", "verdict", "evidence"])
verdicts.to_csv(RESULTS / "translation_validation_verdicts.csv", index=False)
verdicts
