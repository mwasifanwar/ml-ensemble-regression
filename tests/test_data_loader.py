import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import AdvancedDataLoader

class TestAdvancedDataLoader:
    
    def setup_method(self):
        self.loader = AdvancedDataLoader()
        # Create sample data
        self.sample_data = pd.DataFrame({
            'YearsExperience': [1, 2, 3, 4, 5],
            'Salary': [50000, 60000, 70000, 80000, 90000]
        })
    
    def test_load_data(self, tmp_path):
    
        file_path = tmp_path / "test_data.csv"
        self.sample_data.to_csv(file_path, index=False)
        
     
        data = self.loader.load_data(str(file_path))
        assert data is not None
        assert len(data) == 5
        assert 'YearsExperience' in data.columns
        assert 'Salary' in data.columns
    
    def test_validate_data(self):
        self.loader.data = self.sample_data
        assert self.loader.validate_data() == True
    
    def test_validate_data_with_nulls(self):
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[0, 'Salary'] = np.nan
        self.loader.data = data_with_nulls
        assert self.loader.validate_data() == False
    
    def test_exploratory_data_analysis(self):
        self.loader.data = self.sample_data
        eda_results = self.loader.exploratory_data_analysis()
        
        assert 'basic_stats' in eda_results
        assert 'correlation_matrix' in eda_results
        assert 'skewness' in eda_results
        assert 'kurtosis' in eda_results