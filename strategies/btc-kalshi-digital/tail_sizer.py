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
