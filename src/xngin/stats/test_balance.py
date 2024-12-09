import pytest
import pandas as pd
import numpy as np
from patsy import PatsyError
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
