import pytest
import pandas as pd
import numpy as np
from xngin.apiserver.api_types import MetricType
from xngin.stats.assignment import assign_treatment

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    data = {
        'id': range(n),
        'age': np.round(np.random.normal(30, 5, n), 0),
        'income': np.round(np.float64(np.random.lognormal(10, 1, n)), 0),
        'gender': np.random.choice(['M', 'F'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n)
    }
    df = pd.DataFrame(data)
    #df.set_index('id', inplace=True)
    return df

def test_assign_treatment(sample_data):
    result = assign_treatment(
        data=sample_data,
        stratum_cols=['gender', 'region'],
        metric_cols=['age', 'income'],
        id_col='id',
        arm_names=['control', 'treatment'],
        experiment_id='test_exp_001',
        description='Test experiment'
    )
    
    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.balance_ok is not None
    assert result.experiment_id == 'test_exp_001'
    assert result.description == 'Test experiment'
    assert result.sample_size == len(sample_data)
    assert isinstance(result.assignments, pd.DataFrame)
    assert 'treat' in result.assignments.columns

def test_assign_treatment_multiple_arms(sample_data):
    result = assign_treatment(
        data=sample_data,
        stratum_cols=['gender', 'region'],
        metric_cols=['age', 'income'],
        id_col='id',
        arm_names=['control', 'treatment_a', 'treatment_b'],
        experiment_id='test_exp_002',
        description='Test multi-arm experiment'
    )
    
    assert len(result.assignments['treat'].unique()) == 3
    assert result.sample_size == len(sample_data)

def test_assign_treatment_reproducibility(sample_data):
    result1 = assign_treatment(
        data=sample_data,
        stratum_cols=['gender', 'region'],
        metric_cols=['age', 'income'],
        id_col='id',
        arm_names=['control', 'treatment'],
        experiment_id='test_exp_003',
        description='Test reproducibility',
        random_state=42
    )
    
    result2 = assign_treatment(
        data=sample_data,
        stratum_cols=['gender', 'region'],
        metric_cols=['age', 'income'],
        id_col='id',
        arm_names=['control', 'treatment'],
        experiment_id='test_exp_003',
        description='Test reproducibility',
        random_state=42
    )
    
    pd.testing.assert_frame_equal(result1.assignments, result2.assignments)

def test_assign_treatment_with_missing_values(sample_data):
    # Add some missing values
    sample_data.loc[sample_data.index[:100], 'income'] = np.nan
    
    result = assign_treatment(
        data=sample_data,
        stratum_cols=['gender', 'region'],
        metric_cols=['age', 'income'],
        id_col='id',
        arm_names=['control', 'treatment'],
        experiment_id='test_exp_004',
        description='Test with missing values'
    )
    
    assert result.sample_size == len(sample_data)
    assert not result.assignments['treat'].isna().any() 