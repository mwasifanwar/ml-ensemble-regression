import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedVisualization:
    """Advanced visualization capabilities"""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_model_comparison(self, results: Dict[str, Any], metric: str = 'rmse', 
                            save_path: str = None):
        """Compare model performance"""
        
        # Filter out models with errors
        valid_results = {k: v for k, v in results.items() 
                        if 'metrics' in v and v['metrics'] and metric in v['metrics']}
        
        if not valid_results:
            logger.warning("No valid results to plot")
            return
            
        models = list(valid_results.keys())
        metrics = [valid_results[model]['metrics'][metric] for model in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, metrics, color=sns.color_palette("husl", len(models)))
        plt.title(f'Model Comparison - {metric.upper()}', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.upper())
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
            
        plt.show()
    
    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str, save_path: str = None):
        """Comprehensive residual analysis"""
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.7, color='steelblue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontweight='bold')
        axes[0, 0].set_title('Residuals vs Predicted', fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Distribution of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.7, color='green')
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual', fontweight='bold')
        axes[1, 1].set_ylabel('Predicted', fontweight='bold')
        axes[1, 1].set_title('Actual vs Predicted', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual analysis plot saved to {save_path}")
            
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              model_name: str, save_path: str = None):
        """Plot feature importance if available"""
        
        if not feature_importance:
            logger.warning("No feature importance data available")
            return
            
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importance, color=sns.color_palette("coolwarm", len(features)))
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name}', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_learning_curves(self, models_results: Dict[str, Any], save_path: str = None):
        """Plot learning curves for model comparison"""
        
        plt.figure(figsize=(12, 8))
        
        for model_name, results in models_results.items():
            if 'cv_rmse_mean' in results:
                plt.bar(model_name, results['cv_rmse_mean'], 
                       yerr=results['cv_rmse_std'], 
                       capsize=5, alpha=0.7, 
                       label=model_name)
        
        plt.title('Model Comparison - Cross-Validated RMSE', fontsize=16, fontweight='bold')
        plt.ylabel('RMSE (Cross-Validation)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()