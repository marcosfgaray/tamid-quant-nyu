import numpy as np
import pandas as pd
import pytest

from tail_strategy import BINS, CellStats, Config, best_side, tail_candidates


def _obs(mid, bid=None, ask=None, H=6):
    bid = mid - 0.005 if bid is None else bid
    ask = mid + 0.005 if ask is None else ask
    return pd.DataFrame(dict(H=[H], mid=[mid], bid=[bid], ask=[ask]))


def test_tail_bounds_are_strict():
    assert len(tail_candidates(_obs(0.10, 0.095, 0.105))) == 0
    assert len(tail_candidates(_obs(0.90, 0.895, 0.905))) == 0
    assert len(tail_candidates(_obs(0.03))) == 1 and len(tail_candidates(_obs(0.97))) == 1


def test_tail_filters_spread_and_extremes():
    assert len(tail_candidates(_obs(0.05, 0.01, 0.08))) == 0      # spread 7c
    assert len(tail_candidates(_obs(0.995, 0.99, 1.00))) == 0     # ask > 0.99


def test_cellstats_empty_and_shrinkage():
    cs = CellStats(50, BINS)
    f, n = cs.estimate(6, 0.02)
    assert np.isnan(f) and n == 0
    cs.update(pd.DataFrame(dict(H=[6] * 100, mid=[0.02] * 100, y=[0.0] * 100)))
    f, n = cs.estimate(6, 0.02)
    assert n == 100 and f == pytest.approx(50 * 0.02 / 150)
    assert cs.estimate(3, 0.02)[1] == 0                               # cells are per horizon


def test_best_side_sells_overpriced_low_tail():
    side, price, edge = best_side(0.05, 0.06, 0.005, Config())
    assert side == -1 and price == pytest.approx(0.03)
    assert edge == pytest.approx(0.03 - 0.005 - 0.07 * 0.03 * 0.97)


# ---------------------------------------------------------------- simulator
import datetime as dt

from tail_strategy import run


def _synth(days=40):
    rows = []
    for d in range(days):
        day = dt.date(2026, 6, 1) + dt.timedelta(days=d)
        for H in (6, 3):
            for i in range(21):
                if i <= 3:
                    bid, ask, y = 0.935, 0.945, 1.0
                elif i <= 16:
                    bid, ask, y = 0.49, 0.51, 1.0 if i < 10 else 0.0
                else:
                    bid, ask, y = 0.055, 0.065, 0.0
                rows.append(dict(ticker=f"D{d}-T{i}", expiry=f"{day} 21:00:00+00:00", day=day, H=H, y=y, S=65_000.0,
                                 bid=bid, ask=ask, bid_f=bid, ask_f=ask, strike_type="greater",
                                 floor=60_000.0 + 500 * i, mid=(bid + ask) / 2))
    return pd.DataFrame(rows)


def test_portfolio_never_breaches_without_slippage():
    tr, dy = run(_synth(), Config())
    assert len(tr) > 0
    assert (dy.assumed_worst >= -30 - 1e-9).all() and not dy.breach.any()


def test_no_trades_during_warmup():
    tr, dy = run(_synth(days=25), Config())
    assert tr.empty and (dy.pnl == 0).all() and len(dy) == 25


def test_no_lookahead():
    obs = _synth(days=40)
    tr1, _ = run(obs, Config())
    cut = sorted(obs.day.unique())[34]
    obs2 = obs.copy()
    later = obs2.day > cut
    obs2.loc[later, "y"] = np.random.default_rng(1).permutation(obs2.loc[later, "y"].values)
    tr2, _ = run(obs2, Config())
    a = tr1[tr1.day <= cut].drop(columns=["y", "pnl"]).reset_index(drop=True)
    b = tr2[tr2.day <= cut].drop(columns=["y", "pnl"]).reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)


def test_unusable_fill_is_unfilled():
    obs = _synth()
    obs.loc[obs.mid < 0.10, "bid_f"] = 0.0
    obs.loc[obs.mid > 0.90, "ask_f"] = 1.0
    tr, _ = run(obs, Config())
    assert tr.empty


def test_day_without_3h_round_still_settles():
    obs = _synth()
    last = obs.day.max()
    obs = obs[~((obs.day == last) & (obs.H == 3))]
    tr, dy = run(obs, Config())
    assert last in set(dy.day)
    assert dy.loc[dy.day == last, "pnl"].iloc[0] == pytest.approx(tr[tr.day == last].pnl.sum())
    assert (tr[tr.day == last].H == 6).all() and (tr.day == last).any()


def test_bad_fill_shows_in_realized_worst():
    obs = _synth()
    obs.loc[obs.mid < 0.10, "bid_f"] = 0.01   # sells fill 2.5c below the sized price
    obs.loc[obs.mid > 0.90, "ask_f"] = 0.99   # buys fill 2.5c above the sized price
    _, dy = run(obs, Config())
    traded = dy[dy.n_trades > 0]
    assert len(traded) > 0 and (traded.realized_worst < traded.assumed_worst).all()
    assert (dy.breach == (dy.realized_worst < -30 - 1e-9)).all()


def test_per_trade_baseline_can_exceed_cap():
    _, dy = run(_synth(), Config(), sizing="per_trade")
    assert (dy.realized_worst < -30).any()


# ---------------------------------------------------------------- reporting / CLI
from tail_strategy import main, summarize


def test_summarize_empty():
    tr, dy = run(_synth(days=25), Config())
    assert summarize(tr, dy, Config(), "portfolio") == "portfolio: no trades"


def test_summarize_reports_cap_and_breaches():
    tr, dy = run(_synth(), Config())
    s = summarize(tr, dy, Config(), "portfolio")
    assert "max assumed worst case" in s and "breaches" in s and "max drawdown" in s


def test_main_writes_outputs(tmp_path):
    p = tmp_path / "obs.csv"
    _synth().to_csv(p, index=False)
    main(["--data", str(p), "--out-dir", str(tmp_path / "out"), "--no-plot"])
    for f in ("tail_trades.csv", "tail_daily.csv", "tail_summary.txt"):
        assert (tmp_path / "out" / f).exists()


# ---------------------------------------------------------------- final-review fixes
def test_unfillable_hedge_leg_does_not_break_cap():
    obs = _synth()
    last = obs.day.max()
    obs.loc[(obs.day == last) & (obs.mid > 0.90), "ask_f"] = 1.0   # buys cannot fill, sells can
    tr, dy = run(obs, Config())
    d = dy[dy.day == last].iloc[0]
    assert (tr[tr.day == last].side == -1).all() and d.n_trades > 0
    assert d.assumed_worst >= -30 - 1e-9 and not d.breach


def test_summary_reports_clipped_sizing_prices():
    tr, dy = run(_synth(), Config())
    s = summarize(tr, dy, Config(), "portfolio")
    assert f"0 of {len(tr)} trades sized at the 1c/99c clip" in s
    tr2 = tr.copy()
    tr2.loc[tr2.index[:3], "size_price"] = 0.01
    assert f"3 of {len(tr)} trades sized at the 1c/99c clip" in summarize(tr2, dy, Config(), "portfolio")


def _exact_realized_worst(t):
    from tail_sizer import order_fee, payoff_matrix
    grid = np.unique(t.floor)
    gross = payoff_matrix(grid, t.floor.values, t.side.values, t.fill.values, fee_rate=0.0) @ t.q.values
    return float(gross.min() - sum(order_fee(q, p) for q, p in zip(t.q, t.fill)))


def test_trade_pnl_uses_rounded_order_fee():
    from tail_sizer import order_fee
    tr, _ = run(_synth(), Config())
    assert len(tr) > 0
    for r in tr.itertuples():
        assert r.fee == pytest.approx(order_fee(r.q, r.fill))
        assert r.pnl == pytest.approx(r.q * r.side * (r.y - r.fill) - order_fee(r.q, r.fill))


def test_realized_worst_uses_rounded_fees_and_stays_within_cap():
    obs = _synth()
    tr, dy = run(obs, Config())
    floors = obs.drop_duplicates("ticker").set_index("ticker").floor
    for day, t in tr.assign(floor=tr.ticker.map(floors)).groupby("day"):
        w = _exact_realized_worst(t)
        assert dy.loc[dy.day == day, "realized_worst"].iloc[0] == pytest.approx(w)
        assert w >= -30 - 1e-9
