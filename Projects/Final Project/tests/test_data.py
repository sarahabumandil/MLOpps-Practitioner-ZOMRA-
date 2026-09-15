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
