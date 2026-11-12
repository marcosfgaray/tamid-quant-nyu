import numpy as np
import pytest

from tail_sizer import fee, payoff_matrix, worst_case


def test_fee_values():
    assert fee(0.5) == pytest.approx(0.0175)
    assert fee(0.05) == pytest.approx(0.07 * 0.05 * 0.95)


def test_payoff_matrix_short_and_long():
    grid = np.array([100.0, 200.0])
    P = payoff_matrix(grid, np.array([200.0, 100.0]), np.array([-1, 1]), np.array([0.05, 0.95]))
    assert P.shape == (3, 2)
    # short YES @0.05 on K=200: wins 0.05 unless settlement > 200
    assert P[:, 0] == pytest.approx([0.05 - fee(0.05), 0.05 - fee(0.05), 0.05 - 1 - fee(0.05)])
    # long YES @0.95 on K=100: loses 0.95 only when settlement <= 100
    assert P[:, 1] == pytest.approx([-0.95 - fee(0.95), 0.05 - fee(0.95), 0.05 - fee(0.95)])


def test_worst_case_with_base():
    P = np.array([[1.0], [-2.0]])
    assert worst_case(P, np.array([3.0])) == pytest.approx(-6.0)
    assert worst_case(P, np.array([3.0]), base=np.array([0.0, 4.0])) == pytest.approx(-2.0)
