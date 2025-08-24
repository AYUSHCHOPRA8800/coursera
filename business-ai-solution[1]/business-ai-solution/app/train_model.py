import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

from .logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """Model training and comparison for economic indicators"""
    
    def __init__(self, data_dir: str = "../data", models_dir: str = "../models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Define models to compare
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(),
            'Neural Network': MLPRegressor(random_state=42, max_iter=1000)
        }
        
        # Hyperparameter grids for optimization
        self.param_grids = {
            'Ridge Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVR': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training data"""
        try:
            # Try to load combined dataset first
            combined_path = self.data_dir / "combined_data.csv"
            if combined_path.exists():
                df = pd.read_csv(combined_path)
                logger.info(f"Loaded combined dataset with {len(df)} records")
            else:
                # Load individual country data and combine
                df_list = []
                for file_path in self.data_dir.glob("*_data.csv"):
                    country_df = pd.read_csv(file_path)
                    country_df['country'] = file_path.stem.replace('_data', '').upper()
                    df_list.append(country_df)
                
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    logger.info(f"Combined {len(df_list)} country datasets with {len(df)} total records")
                else:
                    raise FileNotFoundError("No data files found")
            
            # Prepare features and target
            feature_columns = [col for col in df.columns if col not in ['target', 'date', 'country', 'country_code']]
            X = df[feature_columns].fillna(0)
            y = df['target'].fillna(df['target'].mean())
            
            logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_baseline_model(self, y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """Create and evaluate baseline model (simple moving average)"""
        logger.info("Creating baseline model")
        
        # Simple moving average baseline
        baseline_predictions = y_train.rolling(window=3, min_periods=1).mean().iloc[-len(y_test):]
        
        # Calculate metrics
        mse = mean_squared_error(y_test, baseline_predictions)
        mae = mean_absolute_error(y_test, baseline_predictions)
        r2 = r2_score(y_test, baseline_predictions)
        
        baseline_metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }
        
        self.results['Baseline'] = baseline_metrics
        logger.info(f"Baseline model - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        return baseline_metrics
    
    def train_and_evaluate_model(self, model_name: str, model, X_train: pd.DataFrame, 
                                X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """Train and evaluate a single model"""
        try:
            logger.info(f"Training {model_name}")
            
            # Create pipeline with scaling for models that need it
            if model_name in ['SVR', 'Neural Network']:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
            else:
                pipeline = Pipeline([
                    ('model', model)
                ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
                'model': pipeline
            }
            
            self.results[model_name] = metrics
            logger.info(f"{model_name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return None
    
    def optimize_hyperparameters(self, model_name: str, model, X_train: pd.DataFrame, 
                               y_train: pd.Series) -> Any:
        """Optimize hyperparameters using GridSearchCV"""
        if model_name not in self.param_grids:
            return model
        
        try:
            logger.info(f"Optimizing hyperparameters for {model_name}")
            
            # Create pipeline with scaling if needed
            if model_name in ['SVR', 'Neural Network']:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                param_grid = {f'model__{k}': v for k, v in self.param_grids[model_name].items()}
            else:
                pipeline = Pipeline([('model', model)])
                param_grid = {f'model__{k}': v for k, v in self.param_grids[model_name].items()}
            
            # Grid search
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}")
            return model
    
    def train_all_models(self) -> Dict[str, Dict]:
        """Train and compare all models"""
        logger.info("Starting model training and comparison")
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Create baseline model
        self.create_baseline_model(y_train, y_test)
        
        # Train and evaluate all models
        for model_name, model in self.models.items():
            # Optimize hyperparameters
            optimized_model = self.optimize_hyperparameters(model_name, model, X_train, y_train)
            
            # Train and evaluate
            metrics = self.train_and_evaluate_model(
                model_name, optimized_model, X_train, X_test, y_train, y_test
            )
            
            if metrics:
                self.results[model_name] = metrics
        
        # Find best model
        self.find_best_model()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def find_best_model(self):
        """Find the best performing model based on R² score"""
        best_r2 = -float('inf')
        best_model_name = None
        
        for model_name, metrics in self.results.items():
            if 'r2' in metrics and metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model_name = model_name
        
        if best_model_name:
            self.best_model_name = best_model_name
            self.best_model = self.results[best_model_name]['model']
            logger.info(f"Best model: {best_model_name} with R² = {best_r2:.4f}")
        else:
            logger.error("No best model found")
    
    def save_results(self):
        """Save training results and best model"""
        # Save results summary
        results_summary = {}
        for model_name, metrics in self.results.items():
            if isinstance(metrics, dict) and 'r2' in metrics:
                results_summary[model_name] = {
                    'r2': metrics['r2'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae']
                }
        
        results_path = self.models_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save best model
        if self.best_model:
            model_path = self.models_dir / "best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logger.info(f"Best model saved to {model_path}")
        
        # Save feature names for later use
        X, _ = self.load_data()
        feature_names = list(X.columns)
        features_path = self.models_dir / "feature_names.json"
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        
        logger.info("Training results saved")
    
    def create_model_comparison_plot(self):
        """Create visualization comparing model performance"""
        try:
            # Prepare data for plotting
            model_names = []
            r2_scores = []
            rmse_scores = []
            
            for model_name, metrics in self.results.items():
                if isinstance(metrics, dict) and 'r2' in metrics:
                    model_names.append(model_name)
                    r2_scores.append(metrics['r2'])
                    rmse_scores.append(metrics['rmse'])
            
            # Create comparison plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² comparison
            bars1 = ax1.bar(model_names, r2_scores, color='skyblue')
            ax1.set_title('Model Performance Comparison - R² Score')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Highlight best model
            best_idx = r2_scores.index(max(r2_scores))
            bars1[best_idx].set_color('red')
            
            # RMSE comparison
            bars2 = ax2.bar(model_names, rmse_scores, color='lightcoral')
            ax2.set_title('Model Performance Comparison - RMSE')
            ax2.set_ylabel('RMSE')
            ax2.tick_params(axis='x', rotation=45)
            
            # Highlight best model (lowest RMSE)
            best_rmse_idx = rmse_scores.index(min(rmse_scores))
            bars2[best_rmse_idx].set_color('red')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.models_dir / "model_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'total_models_tested': len(self.results),
            'model_performance': {},
            'recommendations': []
        }
        
        # Add performance metrics
        for model_name, metrics in self.results.items():
            if isinstance(metrics, dict) and 'r2' in metrics:
                report['model_performance'][model_name] = {
                    'r2': round(metrics['r2'], 4),
                    'rmse': round(metrics['rmse'], 4),
                    'mae': round(metrics['mae'], 4)
                }
        
        # Add recommendations
        if self.best_model_name:
            best_r2 = self.results[self.best_model_name]['r2']
            if best_r2 > 0.8:
                report['recommendations'].append("Excellent model performance achieved")
            elif best_r2 > 0.6:
                report['recommendations'].append("Good model performance, consider feature engineering")
            else:
                report['recommendations'].append("Model performance needs improvement, consider data quality")
        
        # Save report
        report_path = self.models_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Training report generated")
        return report

def main():
    """Main function for model training"""
    logger.info("Starting model training process")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    results = trainer.train_all_models()
    
    # Create comparison plot
    trainer.create_model_comparison_plot()
    
    # Generate report
    report = trainer.generate_training_report()
    
    logger.info("Model training process completed")
    
    return {
        'results': results,
        'best_model': trainer.best_model_name,
        'report': report
    }

if __name__ == "__main__":
    main()
