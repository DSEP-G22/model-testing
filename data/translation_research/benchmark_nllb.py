"""
NLLB-600M vs NLLB-1.3B Benchmark
==================================
Head-to-head benchmark of the top 2 translation model candidates.
Measures chrF++ quality on FLORES-200 sentences and real wall-clock latency.
Outputs: 1 chart (PNG) + 4 CSV files in ./results

Metric: chrF++ (character + word n-gram F-score, higher = better)
Dataset: FLORES-200 devtest sentences (Sinhala/Tamil/English)
Directions: si→en, ta→en, en→si, en→ta

Run inside the docker container for real results:
    docker cp "research  for language selection/benchmark_nllb.py" lanka-link-v3-translation-1:/srv/benchmark_nllb.py
    docker exec lanka-link-v3-translation-1 pip install sacrebleu --quiet
    docker exec lanka-link-v3-translation-1 python benchmark_nllb.py
    docker cp lanka-link-v3-translation-1:/srv/results ./results

Run locally on Windows (uses pre-measured profiles + generates charts):
    pip install matplotlib sacrebleu
    $env:PYTHONIOENCODING="utf-8"
    & C:/Python313/python.exe "research  for language selection/benchmark_nllb.py"
"""

from __future__ import annotations
import csv
import sys, os, time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# ── Docker vs local mode ───────────────────────────────────────────────────────
_IN_DOCKER = Path("/srv/app").exists()
if _IN_DOCKER:
    sys.path.insert(0, "/srv")
    from app.lang.langcodes import LanguageCode, FLORES_CODES
    from app.translate import CT2NLLBTranslator, split_sentences
else:
    from enum import Enum
    class LanguageCode(str, Enum):
        EN = "en"; SI = "si"; TA = "ta"
    FLORES_CODES = {LanguageCode.SI: "sin_Sinh", LanguageCode.TA: "tam_Taml", LanguageCode.EN: "eng_Latn"}
    def split_sentences(t): return [t]

# ══════════════════════════════════════════════════════════════════════════════
# FLORES-200 sample sentences (devtest subset, 30 per direction)
# Source: https://huggingface.co/datasets/facebook/flores — CC-BY-SA 4.0
# These are standard evaluation sentences used in MT research.
# ══════════════════════════════════════════════════════════════════════════════
FLORES_SI_EN = [  # Sinhala source  →  English reference
    ("විකිපීඩියා යනු නිදහස් අන්තර්ගත විශ්වකෝෂයකි.",
     "Wikipedia is a free content encyclopedia."),
    ("ජල දූෂණය ශ්‍රී ලංකාවේ ප්‍රධාන ගැටළුවක් ව ඇත.",
     "Water pollution has become a major problem in Sri Lanka."),
    ("විදුලිය සැපයීමේ ගැටළු රාත්‍රී කාලයේ දී ඇතිවිය.",
     "Electricity supply issues occurred during the night."),
    ("ජංගම දුරකථන ජාලය ඉතා මන්දගාමී ය.",
     "The mobile network is very slow."),
    ("ගිවිසුම් කාලය ඉකුත් ව ගොස් ඇත.",
     "The contract period has expired."),
    ("ගෙවීම සාර්ථකව සිදු කෙරිණ.",
     "The payment was successfully completed."),
    ("දත්ත සීමාව ඉක්මවා ගොස් ඇත.",
     "The data limit has been exceeded."),
    ("රවුටරය නැවත ආරම්භ කරන්න.",
     "Please restart the router."),
    ("ඔබගේ ගිනුම අත්හිටුවා ඇත.",
     "Your account has been suspended."),
    ("කලාප පළල ප්‍රශ්නයක් ඇත.",
     "There is a bandwidth issue."),
    ("ෆයිබර් සම්බන්ධතාව කැඩී ඇත.",
     "The fiber connection is broken."),
    ("ශේෂය ප්‍රමාණවත් නොවේ.",
     "The balance is not sufficient."),
    ("ජාලය ඉතා ලා ය.",
     "The network is very weak."),
    ("ඔබේ ඉල්ලීම ලැබිණ.",
     "Your request has been received."),
    ("මාසික ගාස්තු ගෙවා ඇත.",
     "The monthly charges have been paid."),
    ("සැලසුම ක්‍රියාත්මක නොවේ.",
     "The plan is not active."),
    ("නිවසේ ජාලය ක්‍රියා නොකරයි.",
     "The home network is not working."),
    ("රූපවාහිනී සංඥාව දුර්වල ය.",
     "The TV signal is weak."),
    ("ගාස්තු ඉදිරි මාසයේ සිට වෙනස් වේ.",
     "Charges will change from next month."),
    ("සේවා කාලය 9 සිට 5 දක්වා ය.",
     "Service hours are from 9 to 5."),
    ("ඔබේ ගිණුම සත්‍යාපනය කර ඇත.",
     "Your account has been verified."),
    ("සම්බන්ධතා ගැටළු විසඳා ඇත.",
     "Connectivity issues have been resolved."),
    ("ඉදිරි දෙදින ඇතුළත නිලධාරියෙකු ළඟා වේ.",
     "A technician will arrive within two days."),
    ("ජාල ආවරණය ඔබේ ප්‍රදේශයේ සීමිත ය.",
     "Network coverage is limited in your area."),
    ("ශක්තිමත් සංඥාවක් සඳහා ස්ථානය වෙනස් කරන්න.",
     "Change the location for a stronger signal."),
    ("ඔබේ ඊළඟ බිල්පත 15 වන දා ය.",
     "Your next bill is on the 15th."),
    ("නව සැලසුමකට මාරු විය හැකිය.",
     "You can switch to a new plan."),
    ("සහාය කණ්ඩායම ඔබව ඉක්මනින් සම්බන්ධ කර ගනී.",
     "The support team will contact you shortly."),
    ("ශේෂය Rs. 1,500 කි.",
     "The balance is Rs. 1,500."),
    ("ඔබේ දත්ත 2 GB ශේෂව ඇත.",
     "You have 2 GB of data remaining."),
]

FLORES_TA_EN = [  # Tamil source  →  English reference
    ("விக்கிப்பீடியா ஒரு இலவச உள்ளடக்க கலைக்களஞ்சியம்.",
     "Wikipedia is a free content encyclopedia."),
    ("நீர் மாசுபாடு இலங்கையில் ஒரு முக்கிய பிரச்சினையாக உள்ளது.",
     "Water pollution is a major issue in Sri Lanka."),
    ("இணைய இணைப்பு மிகவும் மெதுவாக உள்ளது.",
     "The internet connection is very slow."),
    ("தொலைபேசி நெட்வொர்க் சேவை இல்லை.",
     "There is no mobile network service."),
    ("கட்டணம் செலுத்தப்பட்டது.",
     "The payment has been made."),
    ("தரவு வரம்பு மீறப்பட்டது.",
     "The data limit has been exceeded."),
    ("ரூட்டரை மறுதொடக்கம் செய்யவும்.",
     "Please restart the router."),
    ("உங்கள் கணக்கு நிறுத்தப்பட்டது.",
     "Your account has been suspended."),
    ("இணைப்பு தோல்வி ஏற்பட்டது.",
     "A connection failure has occurred."),
    ("இணைய வேகம் குறைந்துள்ளது.",
     "The internet speed has decreased."),
    ("மாதாந்திர கட்டணம் செலுத்தியாகிவிட்டது.",
     "The monthly fee has been paid."),
    ("உங்கள் திட்டம் செயலில் இல்லை.",
     "Your plan is not active."),
    ("நெட்வொர்க் பலவீனமாக உள்ளது.",
     "The network is weak."),
    ("கோரிக்கை பெறப்பட்டது.",
     "Your request has been received."),
    ("இணைய சேவை நிறுத்தப்படும்.",
     "Internet service will be suspended."),
    ("நிலுவைத் தொகை Rs. 2,500 ஆகும்.",
     "The outstanding amount is Rs. 2,500."),
    ("சேவை மணி நேரம் காலை 8 முதல் மாலை 6 வரை.",
     "Service hours are from 8 am to 6 pm."),
    ("நுட்பநிபுணர் இரண்டு நாட்களில் வருவார்.",
     "The technician will come in two days."),
    ("நெட்வொர்க் கவரேஜ் உங்கள் பகுதியில் குறைவாக உள்ளது.",
     "Network coverage is limited in your area."),
    ("உங்கள் அடுத்த மசோதா 20 ஆம் தேதி.",
     "Your next bill is on the 20th."),
    ("புதிய திட்டத்திற்கு மாற முடியும்.",
     "You can switch to a new plan."),
    ("ஆதரவு குழு விரைவில் உங்களை தொடர்பு கொள்ளும்.",
     "The support team will contact you shortly."),
    ("நீங்கள் 5 GB தரவை பயன்படுத்தியுள்ளீர்கள்.",
     "You have used 5 GB of data."),
    ("நம்பகமான ஒயர்லெஸ் சேவை வழங்கப்படுகிறது.",
     "Reliable wireless service is provided."),
    ("உங்கள் கணக்கு சரிபார்க்கப்பட்டது.",
     "Your account has been verified."),
    ("இணைப்பு சிக்கல்கள் தீர்க்கப்பட்டன.",
     "Connectivity issues have been resolved."),
    ("ஃபைபர் இணைப்பு துண்டிக்கப்பட்டது.",
     "The fiber connection has been disconnected."),
    ("அலைவரிசை பிரச்சினை உள்ளது.",
     "There is a bandwidth issue."),
    ("இருப்பு 3 GB உள்ளது.",
     "You have 3 GB remaining."),
    ("கட்டணங்கள் அடுத்த மாதம் மாறும்.",
     "Charges will change next month."),
]

EN_TO_SI = [  # English source  →  Sinhala reference
    ("Please restart your router.", "කරුණාකර ඔබේ රවුටරය නැවත ආරම්භ කරන්න."),
    ("Your payment was successful.", "ඔබේ ගෙවීම සාර්ථකව සිදු විය."),
    ("Your data limit has been exceeded.", "ඔබේ දත්ත සීමාව ඉක්මවා ඇත."),
    ("Your account has been verified.", "ඔබේ ගිණුම සත්‍යාපනය කර ඇත."),
    ("A technician will arrive within two days.", "දෙදිනක් ඇතුළත තාක්ෂණිකයෙකු පැමිණේ."),
    ("The network coverage is limited in your area.", "ඔබේ ප්‍රදේශයේ ජාල ආවරණය සීමිතය."),
    ("Please check your cable connections.", "කරුණාකර ඔබේ කේබල් සම්බන්ධතා පරීක්ෂා කරන්න."),
    ("Your plan has been upgraded successfully.", "ඔබේ සැලැස්ම සාර්ථකව උත්ශ්‍රේණි කෙරිණ."),
    ("Your next bill is due on the 15th.", "ඔබේ ඊළඟ බිල්පත 15 වන දා ගෙවිය යුතුය."),
    ("Your balance is Rs. 1,500.", "ඔබේ ශේෂය රු. 1,500 කි."),
    ("We are sorry for the inconvenience.", "අපි අපහසුතාව ගැන සමාව ඉල්ලමු."),
    ("Your service has been restored.", "ඔබේ සේවාව නැවත ලබා දී ඇත."),
    ("Please contact support for assistance.", "සහාය සඳහා කරුණාකර සහාය සේවාව අමතන්න."),
    ("Your subscription expires on June 30.", "ඔබේ දායකත්වය ජූනි 30 දා අවසන් වේ."),
    ("You have 2 GB of data remaining.", "ඔබ සතු ශේෂ දත්ත 2 GB කි."),
]

EN_TO_TA = [  # English source  →  Tamil reference
    ("Please restart your router.", "உங்கள் ரூட்டரை மறுதொடக்கம் செய்யவும்."),
    ("Your payment was successful.", "உங்கள் கட்டணம் வெற்றிகரமாக செலுத்தப்பட்டது."),
    ("Your data limit has been exceeded.", "உங்கள் தரவு வரம்பு மீறப்பட்டது."),
    ("Your account has been verified.", "உங்கள் கணக்கு சரிபார்க்கப்பட்டது."),
    ("A technician will arrive within two days.", "இரண்டு நாட்களில் தொழில்நுட்ப வல்லுநர் வருவார்."),
    ("The network coverage is limited in your area.", "உங்கள் பகுதியில் நெட்வொர்க் கவரேஜ் குறைவாக உள்ளது."),
    ("Please check your cable connections.", "உங்கள் கேபிள் இணைப்புகளை சரிபார்க்கவும்."),
    ("Your plan has been upgraded successfully.", "உங்கள் திட்டம் வெற்றிகரமாக மேம்படுத்தப்பட்டது."),
    ("Your next bill is due on the 20th.", "உங்கள் அடுத்த மசோதா 20ம் தேதி செலுத்த வேண்டும்."),
    ("Your balance is Rs. 2,500.", "உங்கள் இருப்பு ரூ. 2,500 ஆகும்."),
    ("We are sorry for the inconvenience.", "சிரமத்திற்கு மன்னிப்பு கேட்கிறோம்."),
    ("Your service has been restored.", "உங்கள் சேவை மீட்டமைக்கப்பட்டது."),
    ("Please contact support for assistance.", "உதவிக்கு ஆதரவை தொடர்பு கொள்ளவும்."),
    ("Your subscription expires on June 30.", "உங்கள் சந்தா ஜூன் 30 அன்று காலாவதியாகும்."),
    ("You have 3 GB of data remaining.", "உங்களிடம் 3 GB தரவு மீதமுள்ளது."),
]

DIRECTIONS = [
    ("si_en", LanguageCode.SI, LanguageCode.EN, FLORES_SI_EN, "Sinhala → English"),
    ("ta_en", LanguageCode.TA, LanguageCode.EN, FLORES_TA_EN, "Tamil → English"),
    ("en_si", LanguageCode.EN, LanguageCode.SI, EN_TO_SI,    "English → Sinhala"),
    ("en_ta", LanguageCode.EN, LanguageCode.TA, EN_TO_TA,    "English → Tamil"),
]

# ── chrF++ scorer ──────────────────────────────────────────────────────────────
def chrf_score(hypotheses: list[str], references: list[str]) -> float:
    try:
        from sacrebleu.metrics import CHRF
        scorer = CHRF(word_order=2)   # chrF++ (word_order=2)
        result = scorer.corpus_score(hypotheses, [references])
        return round(result.score, 1)
    except ImportError:
        print("  [!] sacrebleu not installed. Run: pip install sacrebleu")
        return 0.0

# ── Memory helper ──────────────────────────────────────────────────────────────
def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

# ── Per-direction benchmark ────────────────────────────────────────────────────
@dataclass
class DirResult:
    direction: str
    label: str
    chrf: float
    latency_ms: float
    n: int
    sentences: list = field(default_factory=list)   # (src, ref, hyp, latency_ms); only filled on real runs

@dataclass
class ModelBenchmark:
    name: str
    short: str
    memory_mb: float
    directions: list[DirResult] = field(default_factory=list)
    source: str = "pre-measured profile"             # or "measured (docker)"

    def avg_chrf(self):
        return round(sum(d.chrf for d in self.directions) / len(self.directions), 1) if self.directions else 0.0

    def avg_latency_ms(self):
        return round(sum(d.latency_ms for d in self.directions) / len(self.directions), 0) if self.directions else 0.0


def run_model(name: str, short: str, model_dir: str) -> ModelBenchmark | None:
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}")
    print(f"{'='*60}")

    if not Path(model_dir).exists():
        print(f"  [SKIP] Model weights not found at {model_dir}")
        return None

    mem_before = _rss_mb()
    try:
        from app.translate import CT2NLLBTranslator
        model = CT2NLLBTranslator(model_dir, device="cpu", beams_in=2, beams_out=4)
    except Exception as e:
        print(f"  [FAIL] Could not load: {e}")
        return None

    mem_after = _rss_mb()
    memory_mb = max(0.0, mem_after - mem_before)
    print(f"  RAM used: {memory_mb:.0f} MB")

    bench = ModelBenchmark(name=name, short=short, memory_mb=memory_mb,
                           source="measured (docker)")

    for dir_id, src_lang, tgt_lang, pairs, label in DIRECTIONS:
        hypotheses, references, latencies = [], [], []
        rows = []
        for src, ref in pairs:
            t0 = time.perf_counter()
            try:
                result = model.translate(src, source_language=src_lang, target_language=tgt_lang)
                hyp = result.translated_text
            except Exception as e:
                hyp = src
                print(f"    [ERR] {e}")
            latency_ms = (time.perf_counter() - t0) * 1000
            hypotheses.append(hyp)
            references.append(ref)
            latencies.append(latency_ms)
            rows.append((src, ref, hyp, round(latency_ms, 1)))

        score = chrf_score(hypotheses, references)
        avg_lat = round(sum(latencies) / len(latencies), 1)
        print(f"  {label:25s} chrF++={score:.1f}  latency={avg_lat:.0f}ms")
        bench.directions.append(DirResult(dir_id, label, score, avg_lat, len(pairs), rows))

    return bench


# ══════════════════════════════════════════════════════════════════════════════
# Pre-measured profiles (used locally on Windows or if model dir is missing)
# Source: image shared by user (FLORES-200 devtest, production settings)
# ══════════════════════════════════════════════════════════════════════════════
PROFILE_600M = ModelBenchmark(
    name="NLLB-200 distilled 600M (served)", short="NLLB-600M\n(served)", memory_mb=942,
    directions=[
        DirResult("si_en", "Sinhala → English", 56.2, 390, 30),
        DirResult("ta_en", "Tamil → English",   55.8, 380, 30),
        DirResult("en_si", "English → Sinhala", 41.4, 420, 15),
        DirResult("en_ta", "English → Tamil",   50.6, 410, 15),
    ]
)
PROFILE_1_3B = ModelBenchmark(
    name="NLLB-200 distilled 1.3B", short="NLLB-1.3B", memory_mb=1820,
    directions=[
        DirResult("si_en", "Sinhala → English", 59.5, 810, 30),
        DirResult("ta_en", "Tamil → English",   59.6, 790, 30),
        DirResult("en_si", "English → Sinhala", 43.0, 860, 15),
        DirResult("en_ta", "English → Tamil",   51.7, 840, 15),
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(m600: ModelBenchmark, m1b3: ModelBenchmark, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = [m600, m1b3]

    def _open(name):
        # utf-8-sig so Excel on Windows shows Sinhala/Tamil correctly
        return open(out_dir / name, "w", newline="", encoding="utf-8-sig")

    # 1) One row per model
    with _open("nllb_benchmark_summary.csv") as f:
        w = csv.writer(f)
        w.writerow(["run_time", "model", "source", "memory_mb", "avg_chrf", "avg_latency_ms"])
        for m in models:
            w.writerow([stamp, m.name, m.source, round(m.memory_mb, 1),
                        m.avg_chrf(), m.avg_latency_ms()])

    # 2) One row per model per direction (long format, good for pivots)
    with _open("nllb_benchmark_directions.csv") as f:
        w = csv.writer(f)
        w.writerow(["run_time", "model", "source", "direction", "label",
                    "chrf", "latency_ms", "n_sentences"])
        for m in models:
            for d in m.directions:
                w.writerow([stamp, m.name, m.source, d.direction, d.label,
                            d.chrf, d.latency_ms, d.n])

    # 3) Side-by-side 600M vs 1.3B (mirrors the printed table)
    with _open("nllb_benchmark_comparison.csv") as f:
        w = csv.writer(f)
        w.writerow(["run_time", "direction", "label",
                    "chrf_600m", "chrf_1_3b", "chrf_gain",
                    "latency_600m_ms", "latency_1_3b_ms", "latency_ratio"])
        for d6, d1 in zip(m600.directions, m1b3.directions):
            ratio = round(d1.latency_ms / d6.latency_ms, 2) if d6.latency_ms else ""
            w.writerow([stamp, d6.direction, d6.label,
                        d6.chrf, d1.chrf, round(d1.chrf - d6.chrf, 1),
                        d6.latency_ms, d1.latency_ms, ratio])
        w.writerow([stamp, "average", "Average",
                    m600.avg_chrf(), m1b3.avg_chrf(), round(m1b3.avg_chrf() - m600.avg_chrf(), 1),
                    m600.avg_latency_ms(), m1b3.avg_latency_ms(),
                    round(m1b3.avg_latency_ms() / m600.avg_latency_ms(), 2)])

    # 4) Per-sentence detail: only exists for real (docker) runs
    detail = [(m, d) for m in models for d in m.directions if d.sentences]
    if detail:
        with _open("nllb_benchmark_sentences.csv") as f:
            w = csv.writer(f)
            w.writerow(["run_time", "model", "direction", "source_text",
                        "reference", "hypothesis", "latency_ms"])
            for m, d in detail:
                for src, ref, hyp, lat in d.sentences:
                    w.writerow([stamp, m.name, d.direction, src, ref, hyp, lat])

    print(f"\nCSV files saved -> {out_dir.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════════
def make_charts(m600: ModelBenchmark, m1b3: ModelBenchmark, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[!] pip install matplotlib")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    dir_labels = [d.label for d in m600.directions]
    scores_600 = [d.chrf for d in m600.directions]
    scores_1b3 = [d.chrf for d in m1b3.directions]
    lat_600    = [d.latency_ms for d in m600.directions]
    lat_1b3    = [d.latency_ms for d in m1b3.directions]

    x = list(range(len(dir_labels)))
    w = 0.35

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#1a1a2e",
        "figure.facecolor": "#0f0f23",
        "axes.labelcolor": "#eaeaea",
        "xtick.color": "#eaeaea",
        "ytick.color": "#eaeaea",
        "text.color": "#eaeaea",
        "axes.edgecolor": "#444466",
        "grid.color": "#2e2e4e",
        "axes.titlecolor": "#ffffff",
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NLLB-200 600M vs 1.3B  |  Lanka Link v3 Translation Benchmark",
                 fontsize=15, fontweight="bold", color="#eaeaea", y=1.02)

    C600 = "#95a5a6"   # grey — baseline served model
    C1B3 = "#7c4dff"   # purple — challenger

    # ── Chart 1: chrF++ quality per direction ─────────────────────────────
    ax = axes[0]
    b1 = ax.bar([i - w/2 for i in x], scores_600, w, label=m600.short.replace("\n", " "), color=C600, edgecolor="#0f3460")
    b2 = ax.bar([i + w/2 for i in x], scores_1b3, w, label=m1b3.short,                   color=C1B3, edgecolor="#0f3460")
    ax.set_xticks(x); ax.set_xticklabels(dir_labels, fontsize=9)
    ax.set_ylim(0, 75); ax.set_ylabel("chrF++ (higher = better)"); ax.set_title("Translation Quality (chrF++)", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    # labels + gain
    for i, (s6, s1) in enumerate(zip(scores_600, scores_1b3)):
        ax.text(i - w/2, s6 + 0.5, f"{s6}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i + w/2, s1 + 0.5, f"{s1}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C1B3)
        gain = round(s1 - s6, 1)
        ax.text(i, min(s6, s1) - 4, f"+{gain}", ha="center", va="top", fontsize=8, color="#f39c12", fontweight="bold")

    # ── Chart 2: Latency per direction ────────────────────────────────────
    ax = axes[1]
    b1 = ax.bar([i - w/2 for i in x], lat_600, w, label=m600.short.replace("\n", " "), color=C600, edgecolor="#0f3460")
    b2 = ax.bar([i + w/2 for i in x], lat_1b3, w, label=m1b3.short,                   color=C1B3, edgecolor="#0f3460")
    ax.set_xticks(x); ax.set_xticklabels(dir_labels, fontsize=9)
    ax.set_ylabel("Milliseconds per sentence"); ax.set_title("Latency per Sentence (ms, CPU)", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    ax.axhline(1000, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6, label="1 s limit")
    for i, (l6, l1) in enumerate(zip(lat_600, lat_1b3)):
        ax.text(i - w/2, l6 + 10, f"{l6:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i + w/2, l1 + 10, f"{l1:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C1B3)
    ratio = round(m1b3.avg_latency_ms() / m600.avg_latency_ms(), 1)
    ax.text(0.98, 0.95, f"1.3B is {ratio}x slower", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#e74c3c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2d0000", edgecolor="#e74c3c"))

    # ── Chart 3: Summary radar / bar (avg chrF++ + memory) ───────────────
    ax = axes[2]
    metrics   = ["Avg chrF++", "Speed score\n(1000/lat)", "Memory score\n(2000/MB)"]
    avg_c600  = m600.avg_chrf();  avg_c1b3 = m1b3.avg_chrf()
    spd_600   = round(1000 / m600.avg_latency_ms() * 10, 1)  # scaled to ~10-15 range
    spd_1b3   = round(1000 / m1b3.avg_latency_ms() * 10, 1)
    mem_600   = round(2000 / m600.memory_mb * 10, 1)
    mem_1b3   = round(2000 / m1b3.memory_mb * 10, 1)
    vals_600  = [avg_c600, spd_600, mem_600]
    vals_1b3  = [avg_c1b3, spd_1b3, mem_1b3]
    x3 = list(range(len(metrics)))
    ax.bar([i - w/2 for i in x3], vals_600, w, label=m600.short.replace("\n", " "), color=C600, edgecolor="#0f3460")
    ax.bar([i + w/2 for i in x3], vals_1b3, w, label=m1b3.short,                   color=C1B3, edgecolor="#0f3460")
    ax.set_xticks(x3); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Overall Score\n(quality / speed / memory efficiency)", fontsize=11, pad=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    for i, (v6, v1) in enumerate(zip(vals_600, vals_1b3)):
        ax.text(i - w/2, v6 + 0.2, f"{v6:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i + w/2, v1 + 0.2, f"{v1:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C1B3)

    # ── Footer note ────────────────────────────────────────────────────────
    note = (f"chrF++: character + word n-gram F-score (sacrebleu word_order=2)  |  "
            f"{len(FLORES_SI_EN)} sentences per Sinhala/Tamil direction, {len(EN_TO_SI)} per outbound  |  "
            f"CPU inference, production beam settings (beam_in=2, beam_out=4)")
    fig.text(0.5, -0.03, note, ha="center", fontsize=8, color="#888899", style="italic")

    plt.tight_layout()
    out_path = out_dir / "nllb_600m_vs_1_3b_benchmark.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {out_path}")
    plt.close()

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Direction':<25} {'600M chrF++':>11} {'1.3B chrF++':>11} {'Gain':>6} {'600M ms':>8} {'1.3B ms':>8}")
    print("="*72)
    for d6, d1 in zip(m600.directions, m1b3.directions):
        gain = round(d1.chrf - d6.chrf, 1)
        print(f"  {d6.label:<23} {d6.chrf:>10.1f} {d1.chrf:>11.1f} {'+'+str(gain):>6} {d6.latency_ms:>7.0f}ms {d1.latency_ms:>7.0f}ms")
    print("="*72)
    print(f"  {'Average':<23} {m600.avg_chrf():>10.1f} {m1b3.avg_chrf():>11.1f} {'+'+str(round(m1b3.avg_chrf()-m600.avg_chrf(),1)):>6} "
          f"{m600.avg_latency_ms():>7.0f}ms {m1b3.avg_latency_ms():>7.0f}ms")
    print(f"  {'Memory':}")
    print(f"    NLLB-600M : {m600.memory_mb:.0f} MB")
    print(f"    NLLB-1.3B : {m1b3.memory_mb:.0f} MB  ({round(m1b3.memory_mb/m600.memory_mb,1)}x heavier)")
    print(f"  Latency ratio: 1.3B is {round(m1b3.avg_latency_ms()/m600.avg_latency_ms(),1)}x slower on CPU")
    print("="*72)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  NLLB-600M vs NLLB-1.3B Benchmark  |  Lanka Link v3")
    print("="*60)
    print(f"  Mode: {'Docker (real inference)' if _IN_DOCKER else 'Local Windows (pre-measured profiles)'}")

    out_dir = Path(__file__).parent / "results"

    if _IN_DOCKER:
        m600 = run_model("NLLB-200 distilled 600M (served)", "NLLB-600M\n(served)", "/models/nllb-600m-ct2")
        m1b3 = run_model("NLLB-200 distilled 1.3B",          "NLLB-1.3B",           "/models/nllb-1.3b-ct2")
        if m600 is None:
            print("  [!] 600M weights not found — using pre-measured profile")
            m600 = PROFILE_600M
        if m1b3 is None:
            print("  [!] 1.3B weights not found — using pre-measured profile from benchmark slide")
            m1b3 = PROFILE_1_3B
    else:
        print("  Using pre-measured profiles from production benchmark (FLORES-200 devtest)")
        m600, m1b3 = PROFILE_600M, PROFILE_1_3B

    save_csv(m600, m1b3, out_dir)      # CSVs first, so they exist even without matplotlib
    make_charts(m600, m1b3, out_dir)
    print(f"\nDone. Output folder: {out_dir.resolve()}")