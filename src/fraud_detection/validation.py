"""Time-aware splitting: out-of-time test set and forward-chaining CV.

Fraud models are trained on the past and used on the future. A random split
lets the model learn from transactions that happen *after* the ones it is
evaluated on, and hides drift. Both helpers here respect time order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit


def out_of_time_split(df: pd.DataFrame, test_start: str | pd.Timestamp | None,
                      train_fraction: float = 2 / 3):
    """Split a time-sorted frame into (train, test) at ``test_start``.

    Without an explicit date, the cut is placed at ``train_fraction`` of the
    covered period, rounded down to midnight.
    """
    ts = pd.to_datetime(df["timestamp"])
    if test_start is None:
        span = ts.max() - ts.min()
        test_start = (ts.min() + span * train_fraction).floor("D")
    cut = pd.Timestamp(test_start)
    train, test = df[ts < cut], df[ts >= cut]
    if train.empty or test.empty:
        raise ValueError(f"split at {cut.date()} leaves an empty partition")
    return train, test, cut


def apply_label_delay(train: pd.DataFrame, cut: pd.Timestamp,
                      label_delay_days: float) -> pd.DataFrame:
    """Drop training rows whose labels would not be known yet at ``cut``.

    Chargebacks and fraud reports arrive days to weeks after the
    transaction. A model deployed at ``cut`` can only learn from
    transactions whose outcome is already known, so the last
    ``label_delay_days`` before the cut are excluded from training (they
    still count as *history* for behavioural features, which need no
    labels).
    """
    embargo_start = cut - pd.Timedelta(days=label_delay_days)
    kept = train[pd.to_datetime(train["timestamp"]) < embargo_start]
    if kept.empty:
        raise ValueError("label delay leaves no training data")
    return kept


def forward_chaining_oof(model, X: pd.DataFrame, y: pd.Series,
                         n_splits: int = 4, gap: int = 0,
                         return_folds: bool = False):
    """Out-of-fold probabilities where each fold is trained on the past only.

    Rows must be in time order. ``gap`` rows between each training window
    and its validation window play the role of the label-delay embargo.
    The first chunk is never predicted (there is nothing earlier to train
    on), so its entries are NaN — as are folds whose training window
    contains no fraud yet.

    With ``return_folds`` also returns, per row, the index of the fold that
    predicted it (0 = earliest, -1 = not predicted).
    """
    oof = np.full(len(X), np.nan)
    folds = np.full(len(X), -1)
    splits = TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X)
    for k, (train_idx, val_idx) in enumerate(splits):
        if y.iloc[train_idx].nunique() < 2:
            continue
        fold = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fold.predict_proba(X.iloc[val_idx])[:, 1]
        folds[val_idx] = k
    return (oof, folds) if return_folds else oof
