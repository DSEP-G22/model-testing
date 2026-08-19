"""Call-centre audio samples + their .docx reference transcripts.

Every call folder holds one .mp3 and one .docx. The .docx starts with an
LLM-written summary block, then one or two transcript sections marked by headings
like "EN transcription" / "GE transcription" / "RU transcription". For non-English
calls the EN section is a *translation* and the other section is the native-language
transcript, which gives us references for both the transcribe and translate tasks.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from common import BENCH_ROOT, CALLS_DIR


def ensure_ffmpeg() -> str:
    """Put a plain `ffmpeg.exe` on PATH and return its path.

    openai-whisper shells out to the literal command `ffmpeg`, but the only binary on this
    machine is the one vendored by imageio-ffmpeg under a versioned filename
    (`ffmpeg-win-x86_64-v7.1.exe`), which is not on PATH. Without this shim every
    `transcribe()` call dies with `FileNotFoundError: [WinError 2]`.
    """
    import os
    import shutil

    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    import imageio_ffmpeg

    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    bin_dir = BENCH_ROOT / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    dst = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not dst.exists():
        shutil.copy2(src, dst)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(dst)


FFMPEG = ensure_ffmpeg()

TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
# Heading variants seen across the corpus: "EN transcription", "GE transcription",
# "English translation", "Portuguese Transcription", "Original Transcription".
SECTION_RE = re.compile(r"^([A-Za-z]{2,12})\s+(transcription|translation)\s*$", re.IGNORECASE)

# Section tags used in the documents -> ISO-639-1 codes Whisper reports.
TAG_TO_LANG = {
    "EN": "en",
    "ENGLISH": "en",
    "GE": "de",
    "DE": "de",
    "GERMAN": "de",
    "RU": "ru",
    "RUSSIAN": "ru",
    "PL": "pl",
    "POLISH": "pl",
    "FR": "fr",
    "FRENCH": "fr",
    "ES": "es",
    "SPANISH": "es",
    "PT": "pt",
    "PORTUGUESE": "pt",
    "IT": "it",
    "ITALIAN": "it",
}

# Folder names carry the language in a parenthesised note; this maps them to codes.
FOLDER_LANG_RE = re.compile(r"\((\w+)[^)]*\)")
FOLDER_TOKEN_TO_LANG = {
    "EN": "en",
    "RU": "ru",
    "PL": "pl",
    "FR": "fr",
    "DE": "de",
    "ES": "es",
    "Portuguese": "pt",
}


def docx_paragraphs(path: Path) -> list[str]:
    """Extract paragraph text from a .docx without any third-party dependency."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    xml = xml.replace("</w:p>", "\n")
    text = re.sub(r"<[^>]+>", "", xml)
    text = (
        text.replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&apos;", "'")
    )
    return [ln.strip() for ln in text.split("\n")]


def parse_transcripts(path: Path, native_lang: str | None = None) -> dict:
    """Return {lang_code: transcript_text} for each transcription section found.

    `native_lang` resolves headings that only say "Original Transcription".
    """
    paras = docx_paragraphs(path)
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in paras:
        if not line:
            continue
        m = SECTION_RE.match(line)
        if m:
            tag = m.group(1).upper()
            if tag in ("ORIGINAL", "NATIVE", "SOURCE"):
                current = native_lang or "orig"
            else:
                current = TAG_TO_LANG.get(tag, tag.lower())
            sections.setdefault(current, [])
            continue
        if current is None:
            continue  # still inside the summary block
        if TIMESTAMP_RE.match(line):
            continue  # drop timestamp markers
        sections[current].append(line)
    return {k: " ".join(v).strip() for k, v in sections.items() if v}


def folder_language(name: str) -> str | None:
    m = FOLDER_LANG_RE.search(name)
    if not m:
        return None
    token = m.group(1)
    return FOLDER_TOKEN_TO_LANG.get(token) or FOLDER_TOKEN_TO_LANG.get(token.upper())


def load_calls() -> list[dict]:
    """One record per call: audio path, expected language, and both references."""
    calls = []
    for folder in sorted(p for p in CALLS_DIR.iterdir() if p.is_dir()):
        mp3 = next(iter(folder.glob("*.mp3")), None)
        docx = next(iter(folder.glob("*.docx")), None)
        if mp3 is None or docx is None:
            continue
        lang = folder_language(folder.name)
        sections = parse_transcripts(docx, native_lang=lang)
        native = sections.get(lang) if lang else None
        # English-language calls only carry the single EN section.
        if native is None and lang == "en":
            native = sections.get("en")
        calls.append(
            {
                "id": folder.name,
                "audio": str(mp3),
                "docx": str(docx),
                "lang": lang,
                "ref_native": native,
                "ref_en": sections.get("en"),
                "sections": list(sections),
                "size_mb": round(mp3.stat().st_size / 1024**2, 2),
            }
        )
    return calls


def normalise_text(s: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace, for WER/CER scoring."""
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def audio_duration_seconds(path: str) -> float:
    import subprocess

    out = subprocess.run(
        [FFMPEG, "-i", path, "-f", "null", "-"],
        capture_output=True,
        text=True,
        errors="ignore",
    ).stderr
    times = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", out)
    if not times:
        return float("nan")
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)
