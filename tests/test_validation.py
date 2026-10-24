import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from fraud_detection.validation import forward_chaining_oof, out_of_time_split


def _frame(n=100):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="6h"),
        "x": np.arange(n, dtype=float),
        "y": np.arange(n) % 7 == 0,
    })


def test_out_of_time_split_respects_the_date():
    df = _frame()
    train, test, cut = out_of_time_split(df, "2026-01-10")
    assert train.timestamp.max() < cut <= test.timestamp.min()
    assert len(train) + len(test) == len(df)


def test_out_of_time_split_default_cut_is_midnight():
    _, _, cut = out_of_time_split(_frame(), None)
    assert cut == cut.normalize()


def test_out_of_time_split_rejects_empty_partition():
    with pytest.raises(ValueError):
        out_of_time_split(_frame(), "2030-01-01")


class _RecordsTrainingRange(BaseEstimator, ClassifierMixin):
    """Predicts the largest x it was trained on, exposing what it saw."""

    def fit(self, X, y):
        self.max_seen_ = X["x"].max()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.max_seen_ / 1000)
        return np.c_[1 - p, p]


def test_forward_chaining_only_trains_on_the_past():
    df = _frame(100)
    oof = forward_chaining_oof(_RecordsTrainingRange(), df[["x"]], df["y"],
                               n_splits=4)
    predicted = ~np.isnan(oof)
    assert not predicted[:20].any()          # first chunk has no past
    assert predicted[20:].all()
    # every prediction came from a model that saw only earlier rows
    assert (oof[predicted] * 1000 < df.x[predicted]).all()


def test_forward_chaining_skips_folds_without_fraud():
    df = _frame(100).assign(y=lambda d: d.index >= 60)
    oof = forward_chaining_oof(_RecordsTrainingRange(), df[["x"]], df["y"],
                               n_splits=4)
    # folds trained on rows < 60 have a single class and are left out
    assert np.isnan(oof[:80]).all()
    assert not np.isnan(oof[80:]).any()
