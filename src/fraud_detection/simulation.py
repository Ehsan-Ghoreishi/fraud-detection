"""
Customer-level card transaction simulator.

Real payment data is confidential, so this module simulates a card portfolio
whose difficulty resembles production fraud data rather than a textbook:

* **Customers have habits.** Each customer has a typical spend level,
  activity rate, preferred hours (some are night owls), a primary device that
  may change mid-period, and may travel abroad. "Unusual" is only meaningful
  relative to the customer's own history.
* **Legitimate traffic overlaps with fraud.** Travel creates foreign,
  far-from-home transactions; new phones create new devices; shopping
  sessions create bursts; night owls shop at 3am.
* **Fraud comes in typologies** with different footprints:

  ==================  ======================================================
  ``card_testing``    bursts of tiny online charges, then a large cash-out
  ``account_takeover`` new device, several purchases far above usual spend
  ``cloned_card``     in-store spending far from home (often abroad)
  ``stealth``         one purchase that looks like the customer's own
  ==================  ======================================================

* **Fraud adapts.** In the last two months account takeover grows and card
  testers raise their test amounts to slip under small-amount rules, so an
  out-of-time test set sees some drift.
* **Labels are noisy.** Some fraud is never reported (``is_fraud`` = 0 while
  ``fraud_type`` records the truth), as happens when cardholders don't
  notice or don't dispute.

``fraud_type`` and ``fraud_episode`` (the attack a fraud transaction belongs
to, -1 for legit) are ground truth for analysis only and must never be used
as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.config import RANDOM_STATE

# category -> (share of legit txns, log-amount offset, online share)
MERCHANT_CATEGORIES = {
    "grocery":        (0.28, -0.3, 0.05),
    "restaurant":     (0.18, 0.0, 0.05),
    "fuel":           (0.10, 0.3, 0.00),
    "online_retail":  (0.18, 0.4, 1.00),
    "digital_goods":  (0.10, -0.8, 1.00),
    "electronics":    (0.06, 1.4, 0.50),
    "travel":         (0.05, 1.8, 0.80),
    "gift_cards":     (0.03, 0.9, 0.70),
    "jewelry":        (0.02, 1.9, 0.20),
}
CATEGORIES = list(MERCHANT_CATEGORIES)
# mean merchant risk score per category (merchant-level scores vary around it)
_CATEGORY_RISK = {
    "grocery": 0.08, "restaurant": 0.10, "fuel": 0.12, "online_retail": 0.25,
    "digital_goods": 0.40, "electronics": 0.35, "travel": 0.20,
    "gift_cards": 0.50, "jewelry": 0.30,
}

FRAUD_TYPES = ["card_testing", "account_takeover", "cloned_card", "stealth"]
_TYPE_MIX_EARLY = [0.25, 0.25, 0.20, 0.30]
_TYPE_MIX_LATE = [0.20, 0.40, 0.10, 0.30]      # drift in the last two months
_UNREPORTED = {"card_testing": 0.05, "account_takeover": 0.03,
               "cloned_card": 0.03, "stealth": 0.20}

_DAY_HOURS = np.array([0.5, 0.3, 0.2, 0.2, 0.3, 0.6, 1.2, 2.2, 3.0, 3.5, 4.0,
                       4.5, 5.0, 4.5, 4.0, 4.0, 4.5, 5.0, 4.8, 4.0, 3.2, 2.4,
                       1.6, 1.0])
_OWL_HOURS = np.array([2.5, 2.2, 1.8, 1.4, 1.0, 0.6, 0.5, 0.6, 1.0, 1.5, 2.0,
                       2.5, 3.0, 3.0, 3.0, 3.0, 3.2, 3.5, 3.8, 4.0, 4.0, 3.8,
                       3.5, 3.0])
_ANY_HOURS = np.ones(24)


@dataclass(frozen=True)
class SimulationConfig:
    n_customers: int = 2_500
    start: str = "2026-01-01"
    days: int = 181
    mean_daily_txns: float = 0.45
    fraud_episode_rate: float = 0.030    # episodes per customer per month
    drift_from_day: int = 120            # fraud mix changes from this day
    n_merchants: int = 1_500


def simulate_transactions(config: SimulationConfig | None = None,
                          seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Simulate a transaction log, sorted by time, one row per transaction."""
    config = config or SimulationConfig()
    if config.n_customers < 1 or config.days < 14:
        raise ValueError("need at least one customer and 14 days")
    rng = np.random.default_rng(seed)
    merchants = _merchants(config.n_merchants, rng)
    customers = _customers(config, rng)
    legit = _legit_transactions(config, customers, merchants, rng)
    fraud = _fraud_transactions(config, customers, merchants, legit, rng)

    df = pd.concat([legit, fraud], ignore_index=True)
    df["timestamp"] = (pd.Timestamp(config.start)
                       + pd.to_timedelta(df.pop("t_days"), unit="D")).dt.floor("s")
    df = df.sort_values(["timestamp", "customer_id"], kind="stable")
    df = df.reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(len(df)))
    df["amount"] = df["amount"].clip(0.5, None).round(2)
    df["distance_from_home"] = df["distance_from_home"].round(1)
    cols = ["transaction_id", "timestamp", "customer_id", "merchant_id",
            "merchant_category", "merchant_risk", "channel", "device_id",
            "foreign_country", "distance_from_home", "amount",
            "is_fraud", "fraud_type", "fraud_episode"]
    return df[cols]


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
def _merchants(n: int, rng) -> pd.DataFrame:
    shares = np.array([v[0] for v in MERCHANT_CATEGORIES.values()])
    cat = rng.choice(CATEGORIES, size=n, p=shares / shares.sum())
    mean = np.array([_CATEGORY_RISK[c] for c in cat])
    # beta with the category mean and moderate spread: risky merchants exist
    # in every category, and most merchants in risky categories are honest
    conc = 12.0
    risk = rng.beta(mean * conc, (1 - mean) * conc)
    return pd.DataFrame({"merchant_id": np.arange(n), "merchant_category": cat,
                         "merchant_risk": risk.round(3)})


def _customers(config: SimulationConfig, rng) -> pd.DataFrame:
    n = config.n_customers
    travels = rng.random(n) < 0.15
    trip_start = rng.uniform(0, config.days - 3, n)
    trip_len = rng.uniform(3, 15, n)
    changes_phone = rng.random(n) < 0.25
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "log_spend": rng.normal(3.1, 0.55, n),
        "daily_rate": rng.gamma(2.0, config.mean_daily_txns / 2.0, n),
        "night_owl": rng.random(n) < 0.10,
        "online_affinity": rng.beta(2, 3, n),
        "trip_start": np.where(travels, trip_start, np.inf),
        "trip_end": np.where(travels, trip_start + trip_len, -np.inf),
        "phone_change_day": np.where(changes_phone,
                                     rng.uniform(0, config.days, n), np.inf),
    })


def _pick_hours(n, weights, rng):
    return rng.choice(24, size=n, p=weights / weights.sum())


def _pick_merchants(categories, merchants, rng, risk_power=0.0):
    """One merchant per requested category; optionally favour risky ones."""
    out = np.empty(len(categories), dtype=int)
    for cat in np.unique(categories):
        idx = np.flatnonzero(categories == cat)
        pool = merchants[merchants.merchant_category == cat]
        w = pool.merchant_risk.to_numpy() ** risk_power
        out[idx] = rng.choice(pool.merchant_id.to_numpy(), size=idx.size,
                              p=w / w.sum())
    return out


def _attach_merchants(df, merchants, rng, risk_power=0.0):
    df["merchant_id"] = _pick_merchants(df["merchant_category"].to_numpy(),
                                        merchants, rng, risk_power)
    df["merchant_risk"] = merchants.merchant_risk.to_numpy()[df["merchant_id"]]
    return df


# --------------------------------------------------------------------------- #
# Legitimate behaviour
# --------------------------------------------------------------------------- #
def _legit_transactions(config, customers, merchants, rng) -> pd.DataFrame:
    counts = rng.poisson(customers.daily_rate * config.days)
    cust = np.repeat(customers.customer_id.to_numpy(), counts)
    c = customers.iloc[cust].reset_index(drop=True)
    n = len(cust)

    day = rng.uniform(0, config.days, n).astype(int)
    hour = np.where(c.night_owl, _pick_hours(n, _OWL_HOURS, rng),
                    _pick_hours(n, _DAY_HOURS, rng))
    t = day + (hour + rng.random(n)) / 24

    # shopping sessions: some purchases are followed by 1-3 more within the
    # hour, so legitimate traffic also produces velocity bursts
    follow = rng.random(n) < 0.12
    n_follow = rng.integers(1, 4, follow.sum())
    src = np.repeat(np.flatnonzero(follow), n_follow)
    t = np.concatenate([t, t[src] + rng.uniform(2, 60, src.size) / 1440])
    cust = np.concatenate([cust, cust[src]])
    c = customers.iloc[cust].reset_index(drop=True)
    n = len(cust)

    shares = np.array([v[0] for v in MERCHANT_CATEGORIES.values()])
    cat = rng.choice(CATEGORIES, size=n, p=shares / shares.sum())
    offset = np.array([MERCHANT_CATEGORIES[k][1] for k in cat])
    online_share = np.array([MERCHANT_CATEGORIES[k][2] for k in cat])
    # customers who like online shopping use online channels more where possible
    p_online = np.clip(online_share * (0.6 + c.online_affinity), 0, 1)
    p_online = np.where(online_share >= 1, 1.0, p_online)
    online = rng.random(n) < p_online
    amount = np.exp(c.log_spend + offset + rng.normal(0, 0.75, n))

    abroad = (t >= c.trip_start) & (t <= c.trip_end)
    cross_border_online = online & (rng.random(n) < 0.04)
    foreign = abroad | cross_border_online
    distance = np.where(abroad, rng.uniform(400, 4000, n),
                        np.where(rng.random(n) < 0.04,
                                 rng.exponential(80, n),
                                 rng.exponential(6, n)))
    distance = np.where(online & ~abroad, rng.exponential(3, n), distance)

    # devices: one primary device, replaced at the phone-change day; a few
    # purchases from a borrowed / work device
    slot = np.where(t > c.phone_change_day, "d1", "d0").astype(object)
    other = rng.random(n) < 0.03
    slot[other] = [f"x{k}" for k in rng.integers(0, 3, other.sum())]
    device = np.array([f"c{cid}-{s}" for cid, s in zip(cust, slot, strict=True)],
                      dtype=object)
    device = np.where(online, device, "")

    df = pd.DataFrame({
        "t_days": t, "customer_id": cust, "merchant_category": cat,
        "channel": np.where(online, "online", "in_store"), "device_id": device,
        "foreign_country": foreign.astype(int), "distance_from_home": distance,
        "amount": amount, "is_fraud": 0, "fraud_type": "none",
        "fraud_episode": -1,
    })
    df = df[df.t_days < config.days]
    return _attach_merchants(df.reset_index(drop=True), merchants, rng)


# --------------------------------------------------------------------------- #
# Fraud typologies
# --------------------------------------------------------------------------- #
def _fraud_transactions(config, customers, merchants, legit, rng) -> pd.DataFrame:
    months = config.days / 30.4
    n_episodes = rng.poisson(config.fraud_episode_rate * config.n_customers
                             * months)
    active = legit.customer_id.unique()
    rows = []
    for ep in range(n_episodes):
        cid = int(rng.choice(active))
        # leave the first week as history so behaviour has a baseline
        start = rng.uniform(7, config.days - 0.5)
        late = start >= config.drift_from_day
        ftype = rng.choice(FRAUD_TYPES, p=_TYPE_MIX_LATE if late
                           else _TYPE_MIX_EARLY)
        cust = customers.iloc[cid]
        rows.append(_EPISODES[ftype](ep, cid, cust, start, late, rng))
    if not rows:
        return legit.iloc[:0]

    df = pd.concat(rows, ignore_index=True)
    df = df[df.t_days < config.days].reset_index(drop=True)
    df["fraud_type"] = df["fraud_type"].astype(str)
    unreported = rng.random(len(df)) < df.fraud_type.map(_UNREPORTED)
    df["is_fraud"] = (~unreported).astype(int)

    # fraudsters favour risky merchants, except stealth fraud which blends in
    risky = df.fraud_type != "stealth"
    parts = [_attach_merchants(df[risky].copy(), merchants, rng, risk_power=1.5),
             _attach_merchants(df[~risky].copy(), merchants, rng, risk_power=0.5)]
    return pd.concat(parts).sort_index()


def _frame(ep, cid, t, cat, online, device, foreign, distance, amount, ftype):
    n = len(t)
    return pd.DataFrame({
        "t_days": t, "customer_id": cid, "merchant_category": cat,
        "channel": np.where(online, "online", "in_store"),
        "device_id": np.where(online, device, ""),
        "foreign_country": np.broadcast_to(foreign, n).astype(int),
        "distance_from_home": distance, "amount": amount,
        "fraud_type": ftype, "fraud_episode": ep,
    })


def _card_testing(ep, cid, cust, start, late, rng):
    n_test = rng.integers(2, 7)
    n_cash = rng.integers(1, 3)
    hour0 = _pick_hours(1, _OWL_HOURS, rng)[0]
    t0 = int(start) + hour0 / 24
    t_test = t0 + np.cumsum(rng.uniform(0.5, 8, n_test)) / 1440
    t_cash = t_test[-1] + np.sort(rng.uniform(10, 180, n_cash)) / 1440
    # later card testers use larger "test" amounts to dodge small-amount rules
    test_amt = rng.uniform(1, 9, n_test) if late else rng.uniform(0.5, 3, n_test)
    cash_amt = rng.lognormal(5.4, 0.5, n_cash)
    foreign = rng.random() < 0.4
    n = n_test + n_cash
    return _frame(
        ep, cid, np.r_[t_test, t_cash],
        np.r_[np.full(n_test, "digital_goods"),
              rng.choice(["electronics", "gift_cards"], n_cash)],
        np.ones(n, bool), f"fraud-{ep}", foreign,
        rng.exponential(3, n), np.r_[test_amt, cash_amt], "card_testing")


def _account_takeover(ep, cid, cust, start, late, rng):
    n = rng.integers(2, 6)
    t = start + np.sort(rng.uniform(0, 6, n)) / 24
    amount = np.exp(cust.log_spend + rng.uniform(1.0, 2.4, n))
    device = f"fraud-{ep}" if rng.random() < 0.9 else f"c{cid}-d0"
    return _frame(
        ep, cid, t,
        rng.choice(["online_retail", "electronics", "gift_cards", "travel"], n),
        np.ones(n, bool), device, rng.random() < 0.3,
        rng.exponential(3, n), amount, "account_takeover")


def _cloned_card(ep, cid, cust, start, late, rng):
    n = rng.integers(2, 7)
    t = int(start) + (_pick_hours(n, _DAY_HOURS, rng) + rng.random(n)) / 24
    t = np.sort(t + rng.integers(0, 2, n))
    # skimmed cards are often cashed out domestically, in another city
    abroad = rng.random() < 0.6
    distance = rng.uniform(800, 6000, n) if abroad else rng.uniform(30, 400, n)
    return _frame(
        ep, cid, t,
        rng.choice(["electronics", "jewelry", "restaurant", "fuel"], n),
        np.zeros(n, bool), "", abroad,
        distance, rng.lognormal(4.9, 0.8, n), "cloned_card")


def _stealth(ep, cid, cust, start, late, rng):
    hour = _pick_hours(1, _OWL_HOURS if cust.night_owl else _DAY_HOURS, rng)
    t = np.array([int(start) + (hour[0] + rng.random()) / 24])
    cat = rng.choice(["grocery", "restaurant", "online_retail",
                      "digital_goods", "electronics"], 1)
    online = MERCHANT_CATEGORIES[cat[0]][2] > 0.5
    device = f"c{cid}-d1" if start > cust.phone_change_day else f"c{cid}-d0"
    amount = np.exp(cust.log_spend + MERCHANT_CATEGORIES[cat[0]][1]
                    + rng.normal(0.5, 0.5, 1))
    return _frame(ep, cid, t, cat, np.array([online]), device, False,
                  rng.exponential(6, 1), amount, "stealth")


_EPISODES = {"card_testing": _card_testing,
             "account_takeover": _account_takeover,
             "cloned_card": _cloned_card, "stealth": _stealth}
