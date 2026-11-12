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
