import numpy as np
import pandas as pd

from v2.features import btc_settlement_series, realized_sigma, settlement_series


def _series(vals, start="2025-03-03T21:00:00Z"):
    exp = pd.date_range(start, periods=len(vals), freq="D", tz="UTC")
    return pd.DataFrame({"expiry": exp, "value": vals})


def test_realized_sigma_hand():
    r = np.random.default_rng(0).normal(0, 0.01, 25)
    vals = 100 * np.exp(np.cumsum(r))
    s = _series(vals)
    t = s.expiry.iloc[-1] + pd.Timedelta(hours=1)
    got = realized_sigma(s, t)
    # 25 values give 24 changes; the last 21 values give the last 20 diffs
    exp = np.std(np.diff(np.log(vals))[-20:], ddof=1)
    assert abs(got - exp) < 1e-12


def test_realized_sigma_min_changes():
    rng = np.random.default_rng(1)
    for n, finite in ((10, False), (11, True)):
        s = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
        t = s.expiry.iloc[-1] + pd.Timedelta(hours=1)
        assert np.isfinite(realized_sigma(s, t)) == finite


def test_realized_sigma_strictly_before():
    s = _series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, 15))))
    t = s.expiry.iloc[-1]
    assert realized_sigma(s, t) == realized_sigma(s.iloc[:-1], t + pd.Timedelta(days=1))
    assert realized_sigma(s, t) != realized_sigma(s, t + pd.Timedelta(days=1))


def test_settlement_series_drops_off_schedule():
    m = pd.DataFrame({
        "expiry": ["2026-01-17T17:53:55.869664+00:00", "2026-01-17T21:00:00+00:00",
                   "2026-01-17T21:00:00+00:00", "2026-01-16T21:30:00+00:00"],
        "expiration_value": [5.0, 7.0, 8.0, 6.0],
        "result": ["yes", "yes", "no", "yes"]})
    out = settlement_series(m)
    assert list(out.value) == [6.0, 7.0]
    assert out.expiry.is_monotonic_increasing
    assert list(out.columns) == ["expiry", "value"]


def test_btc_settlement_uses_open():
    T = pd.Timestamp("2024-03-17 03:00:00", tz="UTC")
    spot = pd.DataFrame({"time": ["2024-03-17 03:00:00+00:00", "2024-03-17 04:00:00+00:00"],
                         "open": [1.0, 3.0], "high": [2, 4], "low": [0.5, 2], "close": [2.0, 3.5]})
    out = btc_settlement_series(spot, pd.Series([T, T + pd.Timedelta(hours=5)]))
    assert len(out) == 1 and out.value.iloc[0] == 1.0 and out.expiry.iloc[0] == T
