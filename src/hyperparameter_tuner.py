import optuna
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_trials: int = 100

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def optimize_knn(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize KNN hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, min(50, len(X_train))),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'p': trial.suggest_int('p', 1, 3),
                'metric': trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan'])
            }
            
            
            if params['algorithm'] in ['ball_tree', 'kd_tree']:
                params['leaf_size'] = trial.suggest_int('leaf_size', 10, 50)
                
            model = KNeighborsRegressor(**params)
            score = cross_val_score(model, X_train, y_train, 
                                  cv=self.config.cv_folds, 
                                  scoring='neg_mean_squared_error').mean()
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials)
        
        logger.info(f"KNN optimization completed. Best score: {study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def optimize_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for multiple models"""
        
        def knn_objective(trial):
            return self._optimize_knn(trial, X_train, y_train)
        
        def rf_objective(trial):
            return self._optimize_random_forest(trial, X_train, y_train)
        
       
        knn_study = optuna.create_study(direction='maximize')
        knn_study.optimize(knn_objective, n_trials=self.config.n_trials)
        
        rf_study = optuna.create_study(direction='maximize')
        rf_study.optimize(rf_objective, n_trials=50)  # Fewer trials for RF
        
        return {
            'knn': {
                'best_params': knn_study.best_params,
                'best_value': knn_study.best_value
            },
            'random_forest': {
                'best_params': rf_study.best_params,
                'best_value': rf_study.best_value
            }
        }
    
    def _optimize_knn(self, trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """KNN optimization objective"""
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, min(50, len(X_train))),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 3),
        }
        
        model = KNeighborsRegressor(**params)
        return cross_val_score(model, X_train, y_train, 
                             cv=self.config.cv_folds, 
                             scoring='neg_mean_squared_error').mean()
    
    def _optimize_random_forest(self, trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Random Forest optimization objective"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**params, random_state=42)
        return cross_val_score(model, X_train, y_train, 
                             cv=self.config.cv_folds, 
                             scoring='neg_mean_squared_error').mean()