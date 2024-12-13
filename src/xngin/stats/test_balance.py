import pytest
import pandas as pd
import numpy as np
from xngin.stats.balance import check_balance


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    data = {
        "treat": np.random.binomial(1, 0.5, n),
        "age": np.random.normal(30, 5, n),
        "income": np.random.lognormal(10, 1, n),
        "gender": np.random.binomial(1, 0.5, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    }
    return pd.DataFrame(data)


def test_check_balance(sample_data):
    result = check_balance(sample_data)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced
    assert result.model_summary is not None


def test_check_balance_with_missing_values(sample_data):
    # Add some missing values
    sample_data.loc[sample_data.index[:100], "income"] = np.nan

    result = check_balance(sample_data)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced


def test_check_balance_with_excluded_cols(sample_data):
    result = check_balance(data=sample_data, exclude_cols=["income", "region"])

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert isinstance(result.is_balanced, bool)


def test_check_balance_invalid_treatment(sample_data):
    invalid_data = sample_data.drop("treat", axis=1)

    with pytest.raises(KeyError):
        check_balance(invalid_data)


def test_check_balance_with_single_value_columns(sample_data):
    sample_data["constant_one"] = [1] * len(sample_data)
    sample_data["constant_none"] = [None] * len(sample_data)

    result = check_balance(sample_data)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced
    assert result.model_summary is not None


def test_check_balance_with_skewed_column():
    """
    If pd.qcut() used labels, this triggers the ValueError:
      Bin labels must be one fewer than the number of bin edges
    """
    data = {
        "treat": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        "skews": [0, 0, 0, 0, 0, 0, 4, 4, np.nan, np.nan],
    }
    df = pd.DataFrame(data)
    result = check_balance(df)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced is False
    assert result.model_summary is not None
