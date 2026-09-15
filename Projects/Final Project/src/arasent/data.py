"""Loads the Arabic e-commerce review dataset.

Expected source (download manually — Kaggle is not reachable from this environment):
  https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews
  (or any Arabic product-review dataset with a text column and a 1-5 star / label column)

Place the CSV at data/raw/arabic_reviews.csv with at least two columns: one text column
(any of: text, review, review_text) and one label column (any of: label, rating, sentiment).
Ratings are mapped 1-2 -> negative, 3 -> neutral, 4-5 -> positive.

If the file is not present, we fall back to a small synthetic Arabic-review generator so
the rest of the pipeline (packaging, MLflow, DVC, API, Docker, tests) can be built and
verified end to end without network access. Swap in the real file for the real numbers —
same as Mini Project 1's NYC-taxi synthetic fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from arasent.config import settings
from arasent.logging_conf import get_logger

logger = get_logger(__name__)

TEXT_COLUMN_CANDIDATES = ["text", "review", "review_text", "Review"]
LABEL_COLUMN_CANDIDATES = ["label", "rating", "sentiment", "Rating"]

_POSITIVE_PHRASES = [
    "المنتج رائع جدا وسريع في التوصيل",
    "جودة ممتازة وأنصح بالشراء",
    "تجربة شراء رائعة والخدمة سريعة",
    "المنتج مطابق للوصف وسعره ممتاز",
    "راضية جدا عن الطلب وسأكرر الشراء",
]
_NEGATIVE_PHRASES = [
    "المنتج سيء جدا ولا يستحق السعر",
    "التوصيل متأخر جدا والخدمة سيئة",
    "المنتج وصل مكسور ولم يتم استرجاع المال",
    "جودة رديئة ولا أنصح به إطلاقا",
    "تجربة محبطة ولن أشتري مرة أخرى",
]
_NEUTRAL_PHRASES = [
    "المنتج عادي لا بأس به",
    "التوصيل في الوقت المتوقع والمنتج مقبول",
    "لا سيء ولا ممتاز، متوسط الجودة",
    "السعر مناسب لكن الجودة متوسطة",
    "المنتج كما هو متوقع بدون مفاجآت",
]


def _generate_synthetic_reviews(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pools = [_NEGATIVE_PHRASES, _NEUTRAL_PHRASES, _POSITIVE_PHRASES]
    labels = rng.integers(0, 3, size=n)  # 0=negative, 1=neutral, 2=positive
    texts = [rng.choice(pools[label]) for label in labels]
    df = pd.DataFrame({"text": texts, "label": labels})
    logger.info("generated_synthetic_reviews", extra={"n_rows": n})
    return df


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _rating_to_label(rating: float) -> int:
    if rating <= 2:
        return 0  # negative
    if rating == 3:
        return 1  # neutral
    return 2  # positive


def load_raw_reviews() -> pd.DataFrame:
    if settings.raw_data_path.exists():
        logger.info("loading_raw_csv", extra={"path": str(settings.raw_data_path)})
        raw = pd.read_csv(settings.raw_data_path)
        text_col = _find_column(raw, TEXT_COLUMN_CANDIDATES)
        label_col = _find_column(raw, LABEL_COLUMN_CANDIDATES)
        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not find text/label columns in {settings.raw_data_path}. "
                f"Expected one of {TEXT_COLUMN_CANDIDATES} and one of {LABEL_COLUMN_CANDIDATES}."
            )
        df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        if df["label"].max() > 2:  # looks like a 1-5 star rating, not a 0-2 class label
            df["label"] = df["label"].apply(_rating_to_label)
        return df.dropna().reset_index(drop=True)

    logger.warning("raw_csv_missing_using_synthetic_data", extra={"path": str(settings.raw_data_path)})
    return _generate_synthetic_reviews()


def clean_text(text: str) -> str:
    """Minimal Arabic text normalization: strip tatweel, normalize alef/yeh variants."""
    text = str(text).strip()
    text = text.replace("ـ", "")  # tatweel
    for alef_variant in ["أ", "إ", "آ"]:
        text = text.replace(alef_variant, "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def load_train_val_split(
    test_size: float | None = None, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw_reviews()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    resolved_test_size = test_size or settings.test_size
    n_test = round(len(df) * resolved_test_size)
    class_counts = df["label"].value_counts()

    # stratify requires: (a) more than one class, (b) every class has >= 2 members (so it
    # can appear on both sides of the split), and (c) the test set is large enough to hold
    # at least one row per class. Tiny datasets (e.g. a smoke-test CSV) fail all three —
    # fall back to a plain random split rather than crashing the whole pipeline.
    can_stratify = (
        df["label"].nunique() > 1
        and class_counts.min() >= 2
        and n_test >= df["label"].nunique()
    )

    train_df, val_df = train_test_split(
        df,
        test_size=resolved_test_size,
        random_state=seed or settings.random_seed,
        stratify=df["label"] if can_stratify else None,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
