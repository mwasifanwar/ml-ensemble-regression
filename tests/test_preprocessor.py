import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessor import AdvancedPreprocessor, ModelConfig

class TestAdvancedPreprocessor:
    
    def setup_method(self):
        self.config = ModelConfig()
        self.preprocessor = AdvancedPreprocessor(self.config)
        
     
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        self.y = pd.Series([100, 200, 300, 400, 500])
    
    def test_create_features(self):
        X_poly = self.preprocessor.create_features(self.X)
        
  
        assert X_poly.shape[1] > self.X.shape[1]
        assert 'feature1 feature2' in X_poly.columns  # Interaction term
    
    def test_scale_features_standard(self):
        X_scaled, scaler = self.preprocessor.scale_features(self.X, 'standard')
        
        assert X_scaled.shape == self.X.shape
        assert scaler is not None
   
        assert abs(X_scaled.mean().mean()) < 1e-10
        assert abs(X_scaled.std().mean() - 1) < 1e-10
    
    def test_prepare_data(self):
        X_train, X_test, y_train, y_test, scaler = self.preprocessor.prepare_data(
            self.X, self.y, create_poly_features=True
        )
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert scaler is not None
        assert len(X_train) + len(X_test) == len(self.X)