"""Tests for cluster ICC calculations."""

import pandas as pd

from xngin.stats.cluster_icc import calculate_icc_from_dataframe


class TestICCFromDataFrame:
    def test_basic_icc_calculation(self):
        """Test ICC calculation with simple DataFrame."""
        # Create test data with known structure
        data = {
            "cluster_id": [1, 1, 1, 2, 2, 2],
            "outcome": [10, 12, 11, 20, 22, 21],
        }
        df = pd.DataFrame(data)

        icc = calculate_icc_from_dataframe(df)

        # ICC should be between 0 and 1
        assert 0 <= icc <= 1
        # With clear cluster separation, ICC should be substantial
        assert icc > 0.5
