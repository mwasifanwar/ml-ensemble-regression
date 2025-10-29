import pandas as pd
import numpy as np
import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from .data_loader import AdvancedDataLoader
from .preprocessor import AdvancedPreprocessor, ModelConfig
from .hyperparameter_tuner import HyperparameterOptimizer
from .model_trainer import AdvancedModelTrainer
from .visualization import AdvancedVisualization

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPipeline:
    """Complete ML pipeline"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_loader = AdvancedDataLoader()
        self.preprocessor = AdvancedPreprocessor(config)
        self.optimizer = HyperparameterOptimizer(config)
        self.trainer = AdvancedModelTrainer(config)
        self.visualizer = AdvancedVisualization()
        
    def run(self, file_path: str, create_poly_features: bool = True):
        """Execute complete pipeline"""
        
        logger.info("Starting Advanced KNN Regression Pipeline")
        
        
        logger.info("Step 1: Loading and validating data")
        data = self.data_loader.load_data(file_path)
        
        if not self.data_loader.validate_data():
            logger.warning("Data validation issues found")
        
        logger.info("Step 2: Performing exploratory data analysis")
        eda_results = self.data_loader.exploratory_data_analysis()
        
        X = data[['YearsExperience']]
        y = data['Salary']
        
        
        logger.info("Step 3: Preparing data (feature engineering and scaling)")
        X_train, X_test, y_train, y_test, scaler = self.preprocessor.prepare_data(
            X, y, create_poly_features=create_poly_features
        )
        
    
        logger.info("Step 4: Optimizing KNN hyperparameters")
        optimization_results = self.optimizer.optimize_knn(X_train, y_train)
        logger.info(f"Best KNN params: {optimization_results['best_params']}")
        
        
        optimized_params = {
            'knn': optimization_results['best_params']
        }
        
        
        logger.info("Step 5: Training and evaluating models")
        results = self.trainer.train_and_evaluate(
            X_train, X_test, y_train, y_test, optimized_params
        )
        
        
        self.trainer.print_comparison()
        
       
        logger.info("Step 6: Generating visualizations")
        
       
        self.visualizer.plot_model_comparison(
            results, 
            save_path="models/model_comparison/model_comparison.png"
        )
        
       
        best_model_name, best_results = self.trainer.get_best_model()
        best_predictions = best_results['predictions']
        self.visualizer.plot_residual_analysis(
            y_test, best_predictions, best_model_name,
            save_path="models/model_comparison/residual_analysis.png"
        )
        
        
        self.visualizer.plot_learning_curves(
            results,
            save_path="models/model_comparison/learning_curves.png"
        )
        
        # Feature importance if available
        if best_results.get('feature_importance'):
            self.visualizer.plot_feature_importance(
                best_results['feature_importance'],
                best_model_name,
                save_path="models/model_comparison/feature_importance.png"
            )
        
       
        self.save_artifacts(results, scaler)
        
        logger.info("Pipeline completed successfully")
        return results
    
    def save_artifacts(self, results: Dict[str, Any], scaler: Any):
        """Save models, scaler, and results"""
        
     
        Path("models/saved_models").mkdir(parents=True, exist_ok=True)
        Path("models/model_comparison").mkdir(parents=True, exist_ok=True)
        
    
        best_model_name, best_results = self.trainer.get_best_model()
        best_model = best_results['model']
        
        joblib.dump(best_model, f"models/saved_models/best_model_{best_model_name}.pkl")
        joblib.dump(scaler, "models/saved_models/scaler.pkl")
        
        
        for name, result in results.items():
            if result['model'] is not None:
                joblib.dump(result['model'], f"models/saved_models/{name}_model.pkl")
        
      
        results_summary = {
            name: {
                'metrics': result['metrics'],
                'cv_rmse_mean': result['cv_rmse_mean'],
                'cv_rmse_std': result['cv_rmse_std']
            }
            for name, result in results.items()
            if 'metrics' in result and result['metrics']
        }
        
        with open("models/model_comparison/results.json", "w") as f:
            json.dump(results_summary, f, indent=2)
       
        best_model_info = {
            'best_model': best_model_name,
            'best_metrics': best_results['metrics'],
            'feature_importance': best_results.get('feature_importance')
        }
        
        with open("models/model_comparison/best_model_info.json", "w") as f:
            json.dump(best_model_info, f, indent=2)
        
        logger.info(f"Artifacts saved successfully. Best model: {best_model_name}")

def main():
    """Main execution function"""
    

    config = ModelConfig()
    

    pipeline = ModelPipeline(config)
 
    try:
        results = pipeline.run('Salary_dataset.csv')
        
       
        best_model_name, best_results = pipeline.trainer.get_best_model()
        print("\n" + "="*50)
        print("🎯 PIPELINE EXECUTION COMPLETE")
        print("="*50)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"📊 Best RMSE: {best_results['metrics']['rmse']:.2f}")
        print(f"📈 R² Score: {best_results['metrics']['r2']:.4f}")
        print(f"🔍 Cross-Val RMSE: {best_results['cv_rmse_mean']:.2f} ± {best_results['cv_rmse_std']:.2f}")
        print(f"💾 Models saved to: models/saved_models/")
        print(f"📋 Results saved to: models/model_comparison/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()