"""Predictor v2 features: settlement series and realized vol (spec section 5, F-vol)."""
import numpy as np
import pandas as pd


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
