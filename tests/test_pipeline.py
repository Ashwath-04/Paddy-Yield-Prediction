import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import load_clean
from src.evaluation import regression_metrics
from src.features import FEATURE_DOCS, add_engineered_features, feature_columns, redundancy_filter
from src.preprocessing import (CategoricalEncoder, add_per_ha_target, drop_exact_duplicates,
                               encode_wind_direction, split)
from src.validation import invalid_value_report, leakage_report


@pytest.fixture(scope="module")
def raw():
    return load_clean()


@pytest.fixture(scope="module")
def prepared(raw):
    return encode_wind_direction(add_engineered_features(add_per_ha_target(drop_exact_duplicates(raw))))


def test_schema(raw):
    assert raw.shape == (2789, 45)
    assert raw.isna().sum().sum() == 0
    assert config.TARGET_RAW in raw


def test_no_invalid_values(raw):
    rep = invalid_value_report(raw)
    assert rep["negative_values"] == 0 and rep["tmin_gt_tmax"] == 0
    assert rep["unknown_wind_direction_codes"] == []


def test_dedup_removes_all_exact_duplicates(raw):
    d = drop_exact_duplicates(raw)
    assert not d.duplicated().any()
    assert len(d) == len(raw) - raw.duplicated().sum()


def test_features_exclude_targets_and_leakage(prepared):
    cols = feature_columns(prepared)
    for bad in (config.TARGET_RAW, config.TARGET_PER_HA, *config.LEAKAGE_COLUMNS):
        assert bad not in cols


def test_engineered_features_finite(prepared):
    assert set(FEATURE_DOCS) <= set(prepared.columns)
    assert np.isfinite(prepared[list(FEATURE_DOCS)].to_numpy()).all()


def test_split_disjoint_and_stratified(prepared):
    tr, te = split(prepared)
    assert set(tr.index).isdisjoint(te.index)
    # No exact feature+target row is shared between train and test.
    key = lambda d: set(map(tuple, d.astype(str).to_numpy()))
    assert key(tr).isdisjoint(key(te))
    assert set(te["agriblock"]) == set(tr["agriblock"])


def test_encoder_uses_train_categories_only(prepared):
    tr, te = split(prepared)
    enc = CategoricalEncoder().fit(tr)
    a, b = enc.transform(tr), enc.transform(te)
    assert list(a.columns) == list(b.columns)


def test_redundancy_filter_drops_constants_and_duplicates():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 1, 1, 1], "d": [4, 1, 3, 2]})
    kept, dropped = redundancy_filter(X)
    assert kept == ["a", "d"] and dropped["c"] == "constant" and "b" in dropped


def test_leakage_findings_hold(raw):
    rep = leakage_report(raw)
    assert all(v == 1 for v in rep["input_qty_unique_values_per_hectare"].values())
    assert rep["weather_cols_determined_by_agriblock"] == rep["n_weather_cols"]
    assert rep["temporal_columns_present"] == []


def test_metrics_perfect_prediction():
    y = np.array([5000.0, 6000.0, 7000.0])
    m = regression_metrics(y, y)
    assert m["R2"] == 1 and m["RMSE"] == 0 and m["MAE"] == 0 and m["MAPE_%"] == 0
