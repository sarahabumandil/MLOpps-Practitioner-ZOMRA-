from __future__ import annotations

import pandas as pd
import pytest

from prodml.features import build_pu_do, dataframe_to_features, row_to_features


def test_build_pu_do_formats_pair():
    assert build_pu_do(43, 151) == "43_151"


def test_row_to_features_happy_path(sample_features):
    feats = row_to_features(43, 151, 3.2)
    assert feats == sample_features


@pytest.mark.parametrize(
    "pu,do,distance,expectation",
    [
        (0, 0, 0.0, "zero distance"),
        (1, 1, -5.0, "negative distance"),
        (None, 2, 1.0, "missing category"),
    ],
)
def test_row_to_features_rejects_invalid_distance(pu, do, distance, expectation):
    if distance <= 0:
        with pytest.raises(ValueError):
            row_to_features(pu, do, distance)


def test_row_to_features_unseen_pu_do_pair_does_not_raise():
    # An unseen PU_DO pair is a valid string feature — DictVectorizer handles
    # it as an all-zero one-hot row downstream, it should never error here.
    feats = row_to_features(999, 998, 1.0)
    assert feats["PU_DO"] == "999_998"


def test_dataframe_to_features_matches_row_level():
    df = pd.DataFrame(
        {"PULocationID": [43, 10], "DOLocationID": [151, 20], "trip_distance": [3.2, 1.0]}
    )
    feats = dataframe_to_features(df)
    assert feats[0] == {"PU_DO": "43_151", "trip_distance": 3.2}
    assert feats[1] == {"PU_DO": "10_20", "trip_distance": 1.0}
