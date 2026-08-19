"""Bitext customer-support intent dataset: loading, cleaning, splitting."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import BITEXT_CSV, SEED

# Bitext masks entities with {{Order Number}} style placeholders. Keeping them as
# literal tokens leaks the intent (e.g. "{{Order Number}}" only appears in order
# intents), so we normalise them to a single generic token instead.
PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")


def load_bitext(normalise_placeholders: bool = True) -> pd.DataFrame:
    df = pd.read_csv(BITEXT_CSV)
    df = df.rename(columns={"instruction": "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df["raw_text"] = df["text"]
    if normalise_placeholders:
        df["text"] = df["text"].str.replace(PLACEHOLDER_RE, "<ent>", regex=True)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()
    return df[["text", "raw_text", "category", "intent", "flags", "n_chars", "n_words"]]


def dedupe(df: pd.DataFrame, key: str = "text") -> pd.DataFrame:
    """Drop exact duplicate utterances (case/whitespace-insensitive).

    Bitext is template-generated, so the same utterance recurs many times. Left in,
    duplicates straddle the train/test boundary and inflate every score.
    """
    norm = df[key].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df.loc[~norm.duplicated()].reset_index(drop=True)


def split(
    df: pd.DataFrame,
    label: str = "intent",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
):
    """Stratified train/val/test split."""
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df, test_size=test_size + val_size, stratify=df[label], random_state=seed
    )
    rel = test_size / (test_size + val_size)
    val, test = train_test_split(
        temp, test_size=rel, stratify=temp[label], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def label_maps(labels) -> tuple[dict, dict]:
    classes = sorted(set(labels))
    return {c: i for i, c in enumerate(classes)}, {i: c for i, c in enumerate(classes)}


def intent_prompts(intents) -> dict:
    """Natural-language descriptions of each intent, for zero-shot semantic matching.

    Derived mechanically from the snake_case intent name so the mapping stays honest
    (no hand-tuned prompt engineering per class).
    """
    out = {}
    for i in intents:
        phrase = i.replace("_", " ")
        out[i] = f"the customer wants to {phrase}"
    return out


def train_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["category", "intent"]).size().rename("n").reset_index()
    g["share_%"] = (100 * g["n"] / g["n"].sum()).round(2)
    return g.sort_values("n", ascending=False).reset_index(drop=True)
