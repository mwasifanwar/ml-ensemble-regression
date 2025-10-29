import numpy as np
import logging
from typing import Dict, Any, List
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_trials: int = 100

class AdvancedModelTrainer:
    """Advanced model training with multiple algorithms"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.results = {}
        
    def create_models(self, optimized_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create multiple regression models for comparison"""
        
        if optimized_params is None:
            optimized_params = {}
        
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        
        models = {
            'KNN': KNeighborsRegressor(**optimized_params.get('knn', {})),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.config.random_state),
            'Lasso Regression': Lasso(random_state=self.config.random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=self.config.random_state),
            'Random Forest': RandomForestRegressor(random_state=self.config.random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.config.random_state),
            'SVR': SVR()
        }
        
        return models
    
    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray,
                          optimized_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train and evaluate all models"""
        
        models = self.create_models(optimized_params)
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred)
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=self.config.cv_folds, 
                                          scoring='neg_mean_squared_error')
                cv_rmse = (-cv_scores) ** 0.5
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'cv_rmse_mean': cv_rmse.mean(),
                    'cv_rmse_std': cv_rmse.std(),
                    'predictions': y_pred,
                    'feature_importance': self._get_feature_importance(model, X_train.columns)
                }
                
                self.models[name] = model
                
                logger.info(f"{name} trained successfully. RMSE: {metrics['rmse']:.2f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {
                    'model': None,
                    'metrics': {},
                    'error': str(e)
                }
            
        self.results = results
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max(abs(y_true - y_pred))
        }
    
    def _get_feature_importance(self, model, feature_names):
        """Get feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                return dict(zip(feature_names, model.coef_))
            else:
                return None
        except:
            return None
    
    def get_best_model(self) -> tuple:
        """Get the best model based on RMSE"""
        if not self.results:
            raise ValueError("No models trained yet")
            
        valid_results = {k: v for k, v in self.results.items() if 'metrics' in v and v['metrics']}
        
        if not valid_results:
            raise ValueError("No valid model results")
            
        best_model_name = min(valid_results.keys(), 
                            key=lambda x: valid_results[x]['metrics']['rmse'])
        
        return best_model_name, valid_results[best_model_name]
    
    def print_comparison(self):
        """Print model comparison results"""
        if not self.results:
            logger.warning("No results to display")
            return
            
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        for model_name, result in self.results.items():
            if 'metrics' in result and result['metrics']:
                metrics = result['metrics']
                print(f"\n{model_name}:")
                print(f"  RMSE: {metrics['rmse']:.2f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  MAE: {metrics['mae']:.2f}")
                print(f"  Cross-Val RMSE: {result['cv_rmse_mean']:.2f} ± {result['cv_rmse_std']:.2f}")