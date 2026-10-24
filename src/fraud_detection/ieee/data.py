"""Download and prepare the IEEE-CIS Fraud Detection data (real transactions).

The data comes from Vesta Corporation's e-commerce payments, released for the
2019 IEEE-CIS Kaggle competition. Only the labelled ``train_*`` files are
used; the competition's test set has no public labels. The files cannot be
redistributed, so they are downloaded with the Kaggle API (after accepting
the competition rules) and never committed.

Two things the raw data lacks are reconstructed here:

* **Time.** ``TransactionDT`` is a time delta in seconds from an undisclosed
  reference; the widely used convention (also adopted here) anchors it at
  2017-12-01, which makes the period December 2017 – May 2018.
* **Customer.** There is no customer ID. Following the approach that top
  competition solutions validated, a client is identified by card
  (``card1``), billing address (``addr1``) and the day the account was
  opened (transaction day minus ``D1``, "days since first transaction").
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.config import DATA_DIR

log = logging.getLogger(__name__)

COMPETITION = "ieee-fraud-detection"
IEEE_DIR = DATA_DIR / "ieee"
FILES = ("train_transaction.csv", "train_identity.csv")
REFERENCE_DATE = pd.Timestamp("2017-12-01")
US_COUNTRY_CODE = 87.0          # addr2 value of ~88% of rows: the home market


def download(dest: Path = IEEE_DIR) -> list[Path]:
    """Fetch the labelled IEEE-CIS files with the Kaggle API.

    Needs a Kaggle API token (``~/.kaggle/kaggle.json``, ``~/.kaggle/
    access_token`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``) and the competition
    rules accepted at kaggle.com/competitions/ieee-fraud-detection/rules.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as err:            # optional dependency
        raise RuntimeError("pip install 'fraud-detection[ieee]' to download "
                           "the IEEE-CIS data") from err
    api = KaggleApi()
    api.authenticate()
    paths = []
    for name in FILES:
        target = dest / name
        if not target.exists():
            log.info("Downloading %s ...", name)
            api.competition_download_file(COMPETITION, name, path=str(dest),
                                          quiet=True)
            archive = dest / f"{name}.zip"
            if archive.exists():
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(dest)
                archive.unlink()
        paths.append(target)
    return paths


def load_raw(src: Path = IEEE_DIR) -> pd.DataFrame:
    """Read and join the transaction and identity tables.

    Numeric columns are read as float32 (the V/C/D blocks are ~400 columns),
    and the result is cached as a pickle next to the CSVs.
    """
    cache = src / "train_joined.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    missing = [f for f in FILES if not (src / f).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not found in {src}: run "
                                "`fraud-detection ieee-download` first")
    txn = _read(src / FILES[0])
    ident = _read(src / FILES[1])
    df = txn.merge(ident, on="TransactionID", how="left")
    df.to_pickle(cache)
    return df


def _read(path: Path) -> pd.DataFrame:
    # columns that look like floats in a sample are read as float32
    head = pd.read_csv(path, nrows=1000)
    floats = {c: "float32" for c in head.columns if head[c].dtype.kind == "f"}
    return pd.read_csv(path, dtype=floats)


def to_log(raw: pd.DataFrame) -> pd.DataFrame:
    """Map IEEE-CIS columns onto the project's transaction-log schema.

    Adds the columns that the point-in-time behaviour features, the rule
    engine and the attack-level metrics expect, and keeps every original
    column for the model. ``fraud_episode`` groups the fraud transactions
    of one reconstructed client (labels in this data propagate across a
    compromised account), and ``fraud_type`` holds the product code of
    fraud rows so detection can be broken down by product.
    """
    df = raw.copy()
    day = df["TransactionDT"] // 86_400
    account_start = (day - df["D1"]).astype("float64")
    df["customer_id"] = (df["card1"].astype(str) + "_"
                         + df["addr1"].fillna(-1).astype(int).astype(str) + "_"
                         + account_start.fillna(-1).astype(int).astype(str))
    df["transaction_id"] = df["TransactionID"]
    df["timestamp"] = REFERENCE_DATE + pd.to_timedelta(df["TransactionDT"],
                                                       unit="s")
    df["amount"] = df["TransactionAmt"].astype("float64")
    df["is_fraud"] = df["isFraud"].astype(int)
    df["merchant_category"] = df["ProductCD"]
    # every transaction is card-not-present; only ~25% carry device data
    device = df["DeviceInfo"] if "DeviceInfo" in df else pd.Series(np.nan, df.index)
    df["device_id"] = device.fillna("").astype(str)
    df["channel"] = np.where(device.notna(), "online", "online_unidentified")
    df["foreign_country"] = (df["addr2"].notna()
                             & (df["addr2"] != US_COUNTRY_CODE)).astype(int)
    df["distance_from_home"] = df["dist1"].fillna(0.0)
    fraud = df["is_fraud"] == 1
    df["fraud_episode"] = -1
    df.loc[fraud, "fraud_episode"] = pd.factorize(df.loc[fraud, "customer_id"])[0]
    df["fraud_type"] = np.where(fraud, "product_" + df["ProductCD"].astype(str),
                                "none")
    return df.sort_values(["timestamp", "transaction_id"],
                          kind="stable").reset_index(drop=True)
