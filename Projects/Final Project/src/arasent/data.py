from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from arasent.config import settings
from arasent.logging_conf import get_logger

logger = get_logger(__name__)

TEXT_COLUMN_CANDIDATES = ["text", "review", "review_text", "Review"]
LABEL_COLUMN_CANDIDATES = ["label", "rating", "sentiment", "Rating"]

_POSITIVE_PHRASES = ["المنتج رائع جدا وسريع في التوصيل","جودة ممتازة وأنصح بالشراء","تجربة شراء رائعة والخدمة سريعة","المنتج مطابق للوصف وسعره ممتاز","راضية جدا عن الطلب وسأكرر الشراء",]
_NEGATIVE_PHRASES = ["المنتج سيء جدا ولا يستحق السعر","التوصيل متأخر جدا والخدمة سيئة","المنتج وصل مكسور ولم يتم استرجاع المال","جودة رديئة ولا أنصح به إطلاقا","تجربة محبطة ولن أشتري مرة أخرى",]
_NEUTRAL_PHRASES = ["المنتج عادي لا بأس به","التوصيل في الوقت المتوقع والمنتج مقبول","لا سيء ولا ممتاز، متوسط الجودة","السعر مناسب لكن الجودة متوسطة","المنتج كما هو متوقع بدون مفاجآت",]

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

_STRING_LABELS = {"negative": 0,"neg": 0,"mixed": 1, "neutral": 1,"positive": 2,"pos": 2,}

def _rating_to_label(rating: float) -> int:
    if rating <= 2:
        return 0  
    if rating == 3:
        return 1  
    return 2  

def _normalize_labels(labels: pd.Series) -> pd.Series:
    if labels.dtype == object or pd.api.types.is_string_dtype(labels):
        lowered = labels.astype(str).str.strip().str.lower()
        if lowered.isin(_STRING_LABELS.keys()).all():
            return lowered.map(_STRING_LABELS).astype(int)
        labels = pd.to_numeric(lowered, errors="coerce")  
    labels = labels.dropna()
    if labels.max() > 2:  
        return labels.apply(_rating_to_label).astype(int)
    return labels.astype(int)

def _read_table(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    return pd.read_csv(path, sep=sep, on_bad_lines="skip", encoding="utf-8")

def _find_local_data_file() -> Path | None:
    for candidate in (settings.raw_data_path, settings.raw_data_path.parent / settings.kaggle_file):
        if candidate.exists():
            return candidate
    return None

def download_from_kaggle() -> Path:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    logger.info("downloading_from_kaggle",extra={"dataset": settings.kaggle_dataset, "file": settings.kaggle_file},)
    loader = getattr(kagglehub, "dataset_load", None) or kagglehub.load_dataset
    df = loader(KaggleDatasetAdapter.PANDAS,settings.kaggle_dataset,settings.kaggle_file,pandas_kwargs={"sep": "\t"},)
    settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.raw_data_path, index=False)
    logger.info("saved_raw_csv", extra={"path": str(settings.raw_data_path), "n_rows": len(df)})
    return settings.raw_data_path

def load_raw_reviews() -> pd.DataFrame:
    path = _find_local_data_file()
    if path is None and settings.auto_download:
        try:
            path = download_from_kaggle()
        except Exception as exc:  
            logger.warning("kaggle_download_failed", extra={"error": str(exc)})
    if path is not None:
        logger.info("loading_raw_data", extra={"path": str(path)})
        raw = _read_table(path)
        text_col = _find_column(raw, TEXT_COLUMN_CANDIDATES)
        label_col = _find_column(raw, LABEL_COLUMN_CANDIDATES)
        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not find text/label columns in {path}. "
                f"Expected one of {TEXT_COLUMN_CANDIDATES} and one of {LABEL_COLUMN_CANDIDATES}." )
        df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        df = df.dropna().reset_index(drop=True)
        df["label"] = _normalize_labels(df["label"])
        if settings.max_rows and len(df) > settings.max_rows:
            df = df.sample(n=settings.max_rows, random_state=settings.random_seed)
        return df.dropna().reset_index(drop=True)
    logger.warning("raw_data_missing_using_synthetic_data", extra={"path": str(settings.raw_data_path)})
    return _generate_synthetic_reviews()

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("ـ", "")
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
    can_stratify = ( df["label"].nunique() > 1and class_counts.min() >= 2 and n_test >= df["label"].nunique())

    train_df, val_df = train_test_split(
        df,
        test_size=resolved_test_size,
        random_state=seed or settings.random_seed,
        stratify=df["label"] if can_stratify else None,)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
