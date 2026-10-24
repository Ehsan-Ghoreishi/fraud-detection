import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from fraud_detection.validation import (
    apply_label_delay,
    forward_chaining_oof,
    out_of_time_split,
)


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


def test_label_delay_drops_immature_labels():
    df = _frame()
    train, _, cut = out_of_time_split(df, "2026-01-20")
    kept = apply_label_delay(train, cut, label_delay_days=5)
    assert kept.timestamp.max() < cut - pd.Timedelta(days=5)
    assert kept.timestamp.max() >= cut - pd.Timedelta(days=5, hours=6)


def test_label_delay_rejects_empty_training_set():
    train, _, cut = out_of_time_split(_frame(), "2026-01-05")
    with pytest.raises(ValueError):
        apply_label_delay(train, cut, label_delay_days=30)


def test_forward_chaining_gap_separates_train_and_validation():
    df = _frame(100)
    oof = forward_chaining_oof(_RecordsTrainingRange(), df[["x"]], df["y"],
                               n_splits=4, gap=10)
    predicted = ~np.isnan(oof)
    # every validation row is at least 10 rows after anything trained on
    assert (df.x[predicted] - oof[predicted] * 1000 > 10).all()


def test_forward_chaining_reports_fold_of_each_prediction():
    df = _frame(100)
    oof, folds = forward_chaining_oof(_RecordsTrainingRange(), df[["x"]],
                                      df["y"], n_splits=4, return_folds=True)
    assert (folds[:20] == -1).all()
    assert folds[20:].tolist() == sorted(folds[20:].tolist())
    assert set(folds[20:]) == {0, 1, 2, 3}
    assert np.isnan(oof[folds == -1]).all()


def test_protocol_measures_time_in_fractional_days():
    from fraud_detection.protocol import _span_days, cv_gap_rows

    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=37,
                                                  freq="h")})   # 36 hours
    assert _span_days(df) == 1.5
    # a quarter-day embargo is a sixth of a 1.5-day history
    assert cv_gap_rows(pd.concat([df] * 100), 0.25, cv_folds=2) == 617
