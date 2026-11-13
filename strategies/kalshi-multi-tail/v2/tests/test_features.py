import numpy as np
import pandas as pd

from datetime import date
from math import erf, sqrt
from pathlib import Path

import yaml

from v2.features import (btc_settlement_series, event_day, ladder_implied, ladder_table, load_events,
                         realized_sigma, settlement_series)

EVENTS = Path(__file__).resolve().parent.parent / "events.yaml"


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


def _phi(x):
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in np.atleast_1d(x)])


def _ladder():
    K = 100 * np.exp(np.linspace(-0.1, 0.1, 30))
    p = 1 - _phi(np.log(K / 100) / 0.03)
    return K, p


def test_ladder_lognormal():
    K, p = _ladder()
    M, sig = ladder_implied(K, p - 0.005, p + 0.005)
    assert abs(M / 100 - 1) < 0.01
    assert abs(sig / 0.03 - 1) < 0.01


def test_ladder_nonmonotone():
    K, p = _ladder()
    M0, _ = ladder_implied(K, p - 0.005, p + 0.005)
    p2 = p.copy()
    p2[12] += 0.05
    M1, _ = ladder_implied(K, p2 - 0.005, p2 + 0.005)
    assert abs(M1 / M0 - 1) < 0.01


def test_ladder_not_spanning():
    K = np.linspace(90, 110, 10)
    p = np.linspace(0.95, 0.3, 10)
    M, sig = ladder_implied(K, p - 0.005, p + 0.005)
    assert np.isnan(M) and np.isnan(sig)


def test_ladder_wide_spread_ignored():
    K, p = _ladder()
    base = ladder_implied(K, p - 0.005, p + 0.005)
    got = ladder_implied(np.append(K, 100.0), np.append(p - 0.005, 0.0), np.append(p + 0.005, 1.0))
    assert got == base


def test_ladder_uses_only_decision_ts():
    K, p = _ladder()
    t0 = pd.Timestamp("2025-02-18 14:00:00", tz="UTC")
    t1 = t0 + pd.Timedelta(hours=1)
    tick = [f"T{i}" for i in range(len(K))]
    markets = pd.DataFrame({"ticker": tick + ["R1"], "event_ticker": ["E"] * (len(K) + 1),
                            "strike_type": ["greater"] * len(K) + ["between"],
                            "floor": list(K) + [100.0]})

    def q(t, tk, pp):
        return pd.DataFrame({"ticker": tk, "ts": str(t), "bid": pp - 0.005, "ask": pp + 0.005,
                             "close": pp, "volume": 1, "open_interest": 1})
    h0 = pd.concat([q(t0, tick, p), q(t0, ["R1"], np.array([0.99]))])
    later = q(t1, tick, np.full(len(K), 0.5))
    dt = pd.DataFrame({"event_ticker": ["E"], "decision_ts": [t0]})
    base = ladder_table(h0, markets, dt)
    assert list(base.columns) == ["event_ticker", "decision_ts", "M", "sigma_imp"]
    assert abs(base.M.iloc[0] / 100 - 1) < 0.01
    pd.testing.assert_frame_equal(ladder_table(pd.concat([h0, later]), markets, dt), base)
    # a strike with no candle at decision_ts is excluded (only its t1 quote exists)
    h2 = pd.concat([h0[h0.ticker != "T5"], later[later.ticker == "T5"]])
    ex = ladder_table(h2, markets, dt)
    K2, p2 = np.delete(K, 5), np.delete(p, 5)
    M, sig = ladder_implied(K2, p2 - 0.005, p2 + 0.005)
    assert abs(ex.M.iloc[0] - M) < 1e-9 and abs(ex.sigma_imp.iloc[0] - sig) < 1e-9


def test_event_day_on_date():
    assert event_day("SPX", date(2025, 12, 10), load_events(EVENTS)) == 1


def test_event_day_previous_date():
    assert event_day("SPX", date(2025, 12, 11), load_events(EVENTS)) == 0


def test_event_unmapped_market():
    ev = load_events(EVENTS)
    assert event_day("NATGAS", date(2025, 12, 10), ev) == 0  # FOMC day; gas report that week is Thu 12-11
    assert event_day("WTI", date(2025, 12, 10), ev) == 1  # EIA petroleum report is Wed 12-10


def test_events_yaml_sane():
    fam = yaml.safe_load(EVENTS.read_text())["families"]
    assert set(fam) == {"fomc", "cpi", "jobs", "eia_petroleum", "eia_gas"}
    for name, f in fam.items():
        assert f["source"] and f["markets"] and f["time_et"]
        ds = [pd.Timestamp(d).date() for d in f["dates"]]
        assert len(ds) == len(set(ds)), name
        assert all(date(2024, 7, 1) <= d <= date(2027, 6, 30) for d in ds), name
    assert len(fam["eia_petroleum"]["dates"]) >= 140 and len(fam["eia_gas"]["dates"]) >= 140
    for y in (2025, 2026):
        assert sum(pd.Timestamp(d).year == y for d in fam["fomc"]["dates"]) == 8
