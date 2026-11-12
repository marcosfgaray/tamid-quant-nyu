#!/usr/bin/env python3
"""
BTC Kalshi tail strategy with portfolio worst-case sizing
=========================================================
Trades only tail contracts (mid < 10% or > 90%) 6h and 3h before the daily 5pm ET expiry, and sizes each expiry
day's book so that the worst possible BTC settlement loses at most CAP_FRAC of bankroll (see tail_sizer.py).

Edge is estimated walk-forward: for each (horizon, price bin) cell, the realized YES rate on prior expiry days,
shrunk toward the market price. Reads output/btc_pm_results/btc_observations.csv written by
btc_digital_model_backtest.py.

Spec: docs/superpowers/specs/2026-11-11-btc-tail-strategy-design.md
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tail_sizer import fee

HERE = Path(__file__).resolve().parent
DATA = HERE / "output" / "btc_pm_results" / "btc_observations.csv"
OUT_DIR = HERE / "output" / "btc_tail_results"

BANKROLL = 1000.0
CAP_FRAC = 0.03
Q_MAX = 50                 # placeholder per ticker per round: the data has no order-book depth
MIN_EDGE = 0.005
SLIP = 0.02                # conservative buffer on decision-time quotes, since fills happen one hour later
N0 = 50                    # prior strength (observations) shrinking each cell toward the market price
MIN_CELL_N = 100
WARMUP_DAYS = 30
HOURS = (6, 3)
MAX_SPREAD = 0.06
FEE_RATE = 0.07            # ASSUMPTION copied from the main backtest; check Kalshi's fee schedule
TAIL_LO, TAIL_HI = 0.10, 0.90
BINS = [0, .005, .015, .025, .035, .05, .075, .10, .90, .925, .95, .965, .975, .985, .995, 1.0001]


@dataclass
class Config:
    bankroll: float = BANKROLL
    cap_frac: float = CAP_FRAC
    q_max: int = Q_MAX
    min_edge: float = MIN_EDGE
    slip: float = SLIP
    n0: float = N0
    min_cell_n: int = MIN_CELL_N
    warmup_days: int = WARMUP_DAYS
    fee_rate: float = FEE_RATE

    @property
    def cap(self) -> float:
        return self.bankroll * self.cap_frac


def load_obs(path):
    o = pd.read_csv(path)
    o = o[o.H.isin(HOURS) & (o.strike_type == "greater")].copy()
    o["day"] = pd.to_datetime(o.day).dt.date
    o["mid"] = (o.bid + o.ask) / 2
    return o.reset_index(drop=True)


def tail_candidates(obs):
    m = (((obs.mid < TAIL_LO) | (obs.mid > TAIL_HI)) & (obs.bid >= 0.01) & (obs.ask <= 0.99)
         & ((obs.ask - obs.bid) <= MAX_SPREAD + 1e-12))
    return obs[m]


class CellStats:
    """Per-(horizon, price bin) running counts of prior observations: n, sum of outcomes, sum of mids."""

    def __init__(self, n0, bins):
        self.n0 = n0
        self.bins = np.asarray(bins, float)
        self.cells = {}

    def _bin(self, mid):
        return int(np.searchsorted(self.bins, mid, side="right") - 1)

    def update(self, df):
        for H, mid, y in zip(df.H, df.mid, df.y):
            c = self.cells.setdefault((int(H), self._bin(mid)), [0, 0.0, 0.0])
            c[0] += 1
            c[1] += y
            c[2] += mid

    def estimate(self, H, mid):
        n, sy, sm = self.cells.get((int(H), self._bin(mid)), (0, 0.0, 0.0))
        if n == 0:
            return float("nan"), 0
        m = sm / n
        return (sy + self.n0 * m) / (n + self.n0), n


def best_side(bid, ask, f, cfg):
    """(side, sizing price, net edge per contract) for the better of buying or selling YES."""
    buy_price = min(ask + cfg.slip, 0.99)
    sell_price = max(bid - cfg.slip, 0.01)
    buy = f - buy_price - fee(buy_price, cfg.fee_rate)
    sell = sell_price - f - fee(sell_price, cfg.fee_rate)
    return (1, buy_price, buy) if buy >= sell else (-1, sell_price, sell)
