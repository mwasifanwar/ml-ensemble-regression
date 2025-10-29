import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedDataLoader:
    """Enhanced data loading and validation class"""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data with comprehensive validation"""
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def validate_data(self) -> bool:
        """Validate data quality and structure"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        checks = {
            'has_nulls': self.data.isnull().sum().sum() == 0,
            'has_duplicates': self.data.duplicated().sum() == 0,
            'has_infinite': np.isfinite(self.data.select_dtypes(include=[np.number])).all().all(),
            'sufficient_rows': len(self.data) > 10,
            'sufficient_features': len(self.data.columns) >= 2
        }
        
        for check, result in checks.items():
            if not result:
                logger.warning(f"Data validation failed: {check}")
                
        return all(checks.values())
    
    def exploratory_data_analysis(self) -> Dict[str, Any]:
        """Comprehensive EDA"""
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        eda_results = {}
        
        
        eda_results['basic_stats'] = self.data.describe()
        eda_results['correlation_matrix'] = self.data.corr()
        eda_results['skewness'] = self.data.skew()
        eda_results['kurtosis'] = self.data.kurtosis()
        
       
        self._generate_eda_plots()
        
        return eda_results
    
    def _generate_eda_plots(self):
        """Generate EDA plots"""
        import seaborn as sns
        import matplotlib.pyplot as plt
        
     
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
     
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            self.data[col].hist(bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            self.data.boxplot(column=col)
            plt.title(f'Boxplot of {col}')
            
            plt.tight_layout()
            plt.show()