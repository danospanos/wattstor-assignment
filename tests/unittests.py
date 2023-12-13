import unittest
import pandas as pd
import numpy as np
from data_modeling.model_pipeline import DataProcessingMixin 


def sample_data():
    dates = pd.date_range(start='2021-01-01', end='2021-02-01', freq='30M')
    data = pd.DataFrame({'value': np.random.rand(len(dates))}, index=dates)
    data.iloc[2:5]['value'] = np.nan
    return data

class TestDataProcessingMixin(unittest.TestCase):
    def test_fill_missing_data(self):
        data_processor = DataProcessingMixin()
        data_processor.data = sample_data()

        data_processor.fill_missing_data()

        self.assertFalse(
            data_processor.data.isnull().values.any(),
            "Data still contains missing values")

if __name__ == '__main__':
    unittest.main(verbosity=2)

