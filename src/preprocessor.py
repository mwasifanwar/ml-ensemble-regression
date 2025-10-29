import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from typing import Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_trials: int = 100

class AdvancedPreprocessor:
    """Advanced data preprocessing with multiple scaling options"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scalers = {}
        self.feature_names = None
        
    def create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial and interaction features"""
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)
        
        
        feature_names = poly.get_feature_names_out(X.columns)
        self.feature_names = feature_names
        
        return pd.DataFrame(X_poly, columns=feature_names)
    
    def scale_features(self, X: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """Scale features using different methods"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaling method")
            
        X_scaled = scaler.fit_transform(X)
        self.scalers[method] = scaler
        
        return pd.DataFrame(X_scaled, columns=X.columns), scaler
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, create_poly_features: bool = True) -> Tuple:
        """Complete data preparation pipeline"""
        
        
        if create_poly_features:
            X_processed = self.create_features(X)
        else:
            X_processed = X.copy()
            
       
        X_scaled, scaler = self.scale_features(X_processed)
        
      
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        logger.info(f"Data prepared: X_train {X_train.shape}, X_test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, scaler