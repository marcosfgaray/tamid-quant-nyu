"""
Worst-case sizing for one expiry day's book of Kalshi "BTC above K" contracts.

Each position pays off as a step function of BTC's settlement price. A book's payoff is evaluated in every
settlement scenario between the day's strikes:
  row 0      settlement at or below the lowest strike (every contract settles NO)
  row j >= 1 settlement just above grid[j-1]           (contract i is YES iff strike_i <= grid[j-1])
Side convention: +1 = buy YES, -1 = sell YES. Payoffs are per contract, net of the Kalshi fee.
"""
import numpy as np


def fee(p, rate=0.07):
    """Kalshi taker fee per contract at price p (ASSUMPTION copied from the main backtest)."""
    return rate * p * (1 - p)


def payoff_matrix(grid, strikes, sides, prices, fee_rate=0.07):
    """Per-contract payoff of each position in each settlement scenario, shape (len(grid) + 1, len(strikes))."""
    grid = np.asarray(grid, float)
    strikes = np.asarray(strikes, float)
    sides = np.asarray(sides, float)
    prices = np.asarray(prices, float)
    yes = np.vstack([np.zeros((1, len(strikes)), bool), strikes[None, :] <= grid[:, None]])
    return sides[None, :] * (yes - prices[None, :]) - fee(prices, fee_rate)[None, :]


def worst_case(P, q, base=None):
    """Lowest book payoff over all scenarios (negative = loss), including already-held positions in base."""
    P = np.asarray(P, float)
    tot = P @ np.asarray(q, float)
    if base is not None:
        tot = tot + np.asarray(base, float)
    return float(tot.min())


def size_portfolio(edges, P, base, cap, q_max):
    """Whole contracts per candidate that maximize expected edge while every scenario of base + P @ q stays >= -cap.

    Returns zeros when there are no candidates, held positions already breach the cap, or the LP fails.
    """
    from scipy.optimize import linprog

    edges = np.asarray(edges, float)
    n = len(edges)
    zeros = np.zeros(n, dtype=int)
    if n == 0:
        return zeros
    P = np.asarray(P, float)
    base = np.asarray(base, float)
    if worst_case(P, zeros, base) < -cap - 1e-9:
        return zeros
    res = linprog(c=-edges, A_ub=-P, b_ub=base + cap, bounds=[(0, q_max)] * n, method="highs")
    if res.status != 0:
        return zeros
    x = np.clip(res.x, 0, q_max)
    scale = 1.0
    while True:
        q = np.floor(x * scale + 1e-9).astype(int)
        if worst_case(P, q, base) >= -cap - 1e-9 or not q.any():
            return q
        scale *= 0.95


def size_per_trade(P, cap, q_max):
    """Baseline: size each candidate so its own worst case is within cap, ignoring every other position."""
    P = np.asarray(P, float)
    loss = np.maximum(-P.min(axis=0), 1e-12)
    return np.minimum(q_max, np.floor(cap / loss)).astype(int)
