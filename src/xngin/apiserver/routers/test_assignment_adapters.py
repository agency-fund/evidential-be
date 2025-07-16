"""Test assignment adapter conversion functions."""

import pytest

from xngin.apiserver.routers.assignment_adapters import (
    _make_balance_check,  # noqa: PLC2701
)
from xngin.apiserver.routers.common_api_types import BalanceCheck
from xngin.stats.balance import BalanceResult


def test_make_balance_check():
    """Test conversion from BalanceResult to BalanceCheck."""
    # Test with None input
    assert _make_balance_check(None, 0.5) is None

    # Test with actual BalanceResult
    balance_result = BalanceResult(
        f_statistic=1.234567890123456,
        f_pvalue=0.876543210987654,
        model_summary="test summary",
        is_balanced=True,
        numerator_df=5.0,
        denominator_df=100.0,
    )
    balance_check = _make_balance_check(balance_result, 0.5)

    assert isinstance(balance_check, BalanceCheck)
    assert balance_check.f_statistic == pytest.approx(1.234567890, abs=1e-9)
    assert balance_check.p_value == pytest.approx(0.876543211, abs=1e-9)
    assert balance_check.balance_ok is True
    assert balance_check.numerator_df == 5
    assert balance_check.denominator_df == 100


def test_make_balance_check_not_balanced():
    """Test conversion when balance is not OK."""
    balance_result = BalanceResult(
        f_statistic=2.5,
        f_pvalue=0.3,  # Less than threshold
        model_summary="test summary",
        is_balanced=False,
        numerator_df=3.0,
        denominator_df=50.0,
    )

    balance_check = _make_balance_check(balance_result, 0.5)

    assert balance_check.balance_ok is False
    assert balance_check.f_statistic == 2.5
    assert balance_check.p_value == 0.3
    assert balance_check.numerator_df == 3
    assert balance_check.denominator_df == 50
