import pandas as pd

from arasent.data import _generate_synthetic_reviews, clean_text, load_raw_reviews


def test_generate_synthetic_reviews_has_all_three_labels():
    df = _generate_synthetic_reviews(n=300, seed=1)
    assert set(df["label"].unique()) <= {0, 1, 2}
    assert len(df) == 300


def test_clean_text_normalizes_alef_and_tatweel():
    assert clean_text("أحمـد") == "احمد"
    assert clean_text("إيه") == "ايه"


def test_load_raw_reviews_falls_back_to_synthetic_when_file_missing():
    df = load_raw_reviews()
    assert len(df) > 0
    assert "text" in df.columns and "label" in df.columns


def test_normalize_labels_handles_kaggle_string_labels():
    from arasent.data import _normalize_labels

    out = _normalize_labels(pd.Series(["Positive", "Mixed", "Negative", "positive "]))
    assert out.tolist() == [2, 1, 0, 2]


def test_normalize_labels_handles_star_ratings_and_class_ids():
    from arasent.data import _normalize_labels

    assert _normalize_labels(pd.Series([1, 2, 3, 4, 5])).tolist() == [0, 0, 1, 2, 2]
    assert _normalize_labels(pd.Series([0, 1, 2])).tolist() == [0, 1, 2]


def test_load_raw_reviews_reads_tsv_with_string_labels(tmp_path, monkeypatch):
    from arasent import data
    from arasent.config import settings

    tsv = tmp_path / "ar_reviews_100k.tsv"
    tsv.write_text("label\ttext\nPositive\tرائع جدا\nMixed\tعادي\nNegative\tسيء جدا\n", encoding="utf-8")
    monkeypatch.setattr(settings, "raw_data_path", tmp_path / "arabic_reviews.csv")
    df = data.load_raw_reviews()
    assert sorted(df["label"].tolist()) == [0, 1, 2]
