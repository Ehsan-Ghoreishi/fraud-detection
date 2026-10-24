"""A small frame with the IEEE-CIS schema, for tests (CI has no Kaggle access).

Mimics the parts of the real data the code relies on: TransactionDT in
seconds, card/addr/D1 for client reconstruction, string categoricals, sparse
identity columns, and fraud labels that propagate across a compromised
client from the first fraud onward.
"""

import numpy as np
import pandas as pd


def make_ieee_like(n_clients: int = 300, days: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    tid = 3_000_000
    for _ in range(n_clients):
        card1 = int(rng.integers(1000, 18000))
        addr1 = float(rng.choice([204.0, 325.0, 299.0, np.nan], p=[.4, .3, .25, .05]))
        start_day = int(rng.integers(-300, days - 10))
        compromised = rng.random() < 0.12
        fraud_from = rng.uniform(20, days) if compromised else np.inf
        n = int(rng.poisson(14)) + 2
        times = np.sort(rng.uniform(max(start_day, 1), days, n)) * 86_400
        email = rng.choice(["gmail.com", "yahoo.com", "anonymous.com", np.nan])
        device = rng.choice(["Windows", "iOS Device", "MacOS"]) \
            if rng.random() < 0.3 else np.nan
        for t in times:
            fraud = t / 86_400 >= fraud_from
            day = int(t // 86_400)
            rows.append({
                "TransactionID": tid, "isFraud": int(fraud),
                "TransactionDT": int(t),
                "TransactionAmt": round(
                    float(rng.lognormal(4.8 if fraud else 4, 1)), 3),
                "ProductCD": rng.choice(["W", "C", "H"] if not fraud else ["C", "H"]),
                "card1": card1, "card2": float(rng.integers(100, 600)),
                "card4": rng.choice(["visa", "mastercard"]),
                "card6": rng.choice(["debit", "credit"]),
                "addr1": addr1, "addr2": 87.0 if rng.random() < 0.9 else 60.0,
                "dist1": float(rng.exponential(30)) if rng.random() < 0.5 else np.nan,
                "P_emaildomain": email,
                "R_emaildomain": rng.choice(["gmail.com", np.nan]),
                "C1": float(rng.poisson(6 if fraud else 1.5)),
                "C2": float(rng.poisson(1.5)),
                "D1": float(day - start_day),
                "D15": float(rng.integers(0, 400)) if rng.random() < 0.8 else np.nan,
                "M4": rng.choice(["M0", "M1", "M2", np.nan]),
                "M6": rng.choice(["T", "F"]),
                **{f"V{i}": float(rng.normal(1.5 if fraud else 0, 1))
                   for i in range(1, 6)},
                "id_01": -5.0 if isinstance(device, str) else np.nan,
                "id_31": "chrome 63.0" if isinstance(device, str) else np.nan,
                "DeviceType": ("desktop" if device in ("Windows", "MacOS")
                               else "mobile") if isinstance(device, str) else np.nan,
                "DeviceInfo": device,
            })
            tid += 1
    return pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)


def split_tables(df: pd.DataFrame):
    """Split into Kaggle's two files: transaction and (sparse) identity."""
    id_cols = ["id_01", "id_31", "DeviceType", "DeviceInfo"]
    ident = df.loc[df["DeviceInfo"].notna(), ["TransactionID", *id_cols]]
    return df.drop(columns=id_cols), ident
