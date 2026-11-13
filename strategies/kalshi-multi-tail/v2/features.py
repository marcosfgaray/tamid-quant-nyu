"""Predictor v2 features: settlement series and realized vol (spec section 5, F-vol)."""
from datetime import date

import numpy as np
import pandas as pd
import yaml


def settlement_series(markets):
    """Settlement value per expiry for non-BTC markets, from raw markets.csv rows.

    Keeps settled rows with a non-null expiration_value and an on-schedule expiry (same rule as
    multi_obs.build_market_obs); one row per expiry (first value), sorted."""
    mk = markets[markets.result.isin(["yes", "no"]) & markets.expiration_value.notna()].copy()
    mk["expiry"] = pd.to_datetime(mk.expiry, utc=True, format="ISO8601")
    on_schedule = mk.expiry.dt.minute.isin([0, 30]) & (mk.expiry.dt.second == 0) & (mk.expiry.dt.microsecond == 0)
    mk = mk[on_schedule].drop_duplicates("expiry", keep="first")
    out = pd.DataFrame({"expiry": mk.expiry, "value": mk.expiration_value.astype(float)})
    return out.sort_values("expiry").reset_index(drop=True)


def btc_settlement_series(spot, expiries):
    """BTC settlement = open of the hourly candle whose time (candle open) equals the expiry."""
    sp = spot.copy()
    sp["time"] = pd.to_datetime(sp["time"], utc=True)
    opens = sp.drop_duplicates("time").set_index("time")["open"]
    exp = pd.Series(pd.to_datetime(pd.Series(expiries), utc=True).unique())
    exp = exp[exp.isin(opens.index)]
    out = pd.DataFrame({"expiry": exp.values, "value": opens.reindex(exp).astype(float).values})
    out["expiry"] = pd.to_datetime(out.expiry, utc=True)
    return out.sort_values("expiry").reset_index(drop=True)


def realized_sigma(series, t, n_changes=20, min_changes=10):
    """Std (ddof=1) of the last n_changes log changes among values with expiry < t; NaN if too few."""
    s = series[series.expiry < t].sort_values("expiry")
    v = s.value.to_numpy(dtype=float)[-(n_changes + 1):]
    if len(v) - 1 < min_changes:
        return np.nan
    return float(np.std(np.diff(np.log(v)), ddof=1))


def pav_decreasing(p):
    """Pool-adjacent-violators fit of a non-increasing sequence (unit weights)."""
    blocks = []  # [sum, count]
    for v in np.asarray(p, dtype=float):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    return np.concatenate([np.full(n, s / n) for s, n in blocks]) if blocks else np.array([])


def ladder_implied(strikes, bid, ask, max_spread=0.10):
    """Implied median M and sigma from a ladder of P(settle > K) quotes; (NaN, NaN) if unusable.

    Only two-sided quotes with ask - bid <= max_spread are used. Mids are PAV-smoothed
    (non-increasing in K); K at p = 0.5/0.25/0.75 is interpolated on the collapsed fit."""
    K, b, a = (np.asarray(x, dtype=float) for x in (strikes, bid, ask))
    ok = np.isfinite(K) & np.isfinite(b) & np.isfinite(a) & (a - b <= max_spread)
    K, mid = K[ok], ((b + a) / 2)[ok]
    if len(K) < 2:
        return np.nan, np.nan
    order = np.argsort(K, kind="stable")
    K, fit = K[order], pav_decreasing(mid[order])
    if fit.max() < 0.75 or fit.min() > 0.25:
        return np.nan, np.nan
    levels = np.unique(fit)  # increasing p; mean K per level collapses flat runs
    kbar = np.array([K[fit == lv].mean() for lv in levels])
    k50, k25, k75 = (float(np.interp(q, levels, kbar)) for q in (0.5, 0.25, 0.75))
    return k50, (k25 - k75) / (2 * 0.6745 * k50)


def ladder_table(hourly, markets, decision_times):
    """Ladder (M, sigma_imp) per (event_ticker, decision_ts), from hourly quotes at exactly decision_ts."""
    mk = markets[markets.strike_type == "greater"][["ticker", "event_ticker", "floor"]]
    h = hourly[["ticker", "ts", "bid", "ask"]].copy()
    h["ts"] = pd.to_datetime(h.ts, utc=True)
    h = h.merge(mk, on="ticker")
    rows = []
    for ev, ts in zip(decision_times.event_ticker, decision_times.decision_ts):
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        g = h[(h.event_ticker == ev) & (h.ts == ts)]
        M, sig = ladder_implied(g.floor.to_numpy(float), g.bid.to_numpy(float), g.ask.to_numpy(float))
        rows.append((ev, ts, M, sig))
    return pd.DataFrame(rows, columns=["event_ticker", "decision_ts", "M", "sigma_imp"])


def load_events(path):
    """Event calendar: market -> set of NY dates with a scheduled release (union over families)."""
    with open(path) as fh:
        fams = yaml.safe_load(fh)["families"]
    out = {}
    for fam in fams.values():
        days = {d if isinstance(d, date) else date.fromisoformat(str(d)) for d in fam["dates"]}
        for m in fam["markets"]:
            out.setdefault(m, set()).update(days)
    return out


def event_day(market, day, events):
    """1 if `day` (NY date) is a scheduled event day for `market`, else 0 (unmapped markets give 0)."""
    return int(day in events.get(market, ()))
