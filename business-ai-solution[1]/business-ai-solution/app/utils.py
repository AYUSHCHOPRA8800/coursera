import pickle
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from .logger import get_logger

logger = get_logger(__name__)

def load_model(model_path: str):
    """Load a trained model from file"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def save_model(model, model_path: str):
    """Save a trained model to file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved successfully to {model_path}")
        
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")
        raise

def validate_input_data(features: Dict[str, float]) -> bool:
    """Validate input features for prediction"""
    try:
        # Check if features is a dictionary
        if not isinstance(features, dict):
            raise ValueError("Features must be a dictionary")
        
        # Check if features are not empty
        if not features:
            raise ValueError("Features dictionary cannot be empty")
        
        # Check if all values are numeric
        for key, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                raise ValueError(f"Feature {key} must be a valid numeric value")
        
        # Check for reasonable value ranges (economic indicators)
        for key, value in features.items():
            if 'gdp' in key.lower() and (value < -50 or value > 50):
                logger.warning(f"GDP growth value {value} seems unusual for {key}")
            elif 'inflation' in key.lower() and (value < -50 or value > 100):
                logger.warning(f"Inflation value {value} seems unusual for {key}")
            elif 'unemployment' in key.lower() and (value < 0 or value > 100):
                logger.warning(f"Unemployment value {value} seems unusual for {key}")
        
        return True
        
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise

def normalize_features(features: Dict[str, float], 
                      feature_stats: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """Normalize features using z-score normalization"""
    try:
        if feature_stats is None:
            # Use default statistics if not provided
            feature_stats = {
                'gdp_growth': {'mean': 2.5, 'std': 1.5},
                'inflation': {'mean': 2.0, 'std': 1.0},
                'unemployment': {'mean': 5.0, 'std': 2.0},
                'interest_rate': {'mean': 3.0, 'std': 1.5},
                'trade_balance': {'mean': 0.0, 'std': 3.0},
                'government_debt': {'mean': 60.0, 'std': 15.0},
                'foreign_investment': {'mean': 2.0, 'std': 1.5},
                'population_growth': {'mean': 1.0, 'std': 0.5}
            }
        
        normalized_features = {}
        
        for key, value in features.items():
            if key in feature_stats:
                mean = feature_stats[key]['mean']
                std = feature_stats[key]['std']
                normalized_features[key] = (value - mean) / std
            else:
                # If no statistics available, use original value
                normalized_features[key] = value
                logger.warning(f"No normalization statistics for feature: {key}")
        
        return normalized_features
        
    except Exception as e:
        logger.error(f"Error normalizing features: {e}")
        raise

def calculate_prediction_confidence(prediction: float, 
                                  historical_predictions: List[float],
                                  model_accuracy: float = 0.85) -> float:
    """Calculate prediction confidence based on historical data and model accuracy"""
    try:
        if not historical_predictions:
            return model_accuracy
        
        # Calculate how far the prediction is from historical mean
        historical_mean = np.mean(historical_predictions)
        historical_std = np.std(historical_predictions)
        
        if historical_std == 0:
            return model_accuracy
        
        # Z-score of current prediction
        z_score = abs(prediction - historical_mean) / historical_std
        
        # Confidence decreases as prediction deviates from historical patterns
        confidence_factor = max(0.1, 1.0 - (z_score * 0.1))
        
        # Combine with model accuracy
        confidence = model_accuracy * confidence_factor
        
        return min(1.0, max(0.0, confidence))
        
    except Exception as e:
        logger.error(f"Error calculating prediction confidence: {e}")
        return 0.5  # Default confidence

def create_sample_data(countries: List[str] = None, 
                      num_samples: int = 100) -> List[Dict[str, float]]:
    """Create sample data for testing"""
    if countries is None:
        countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR']
    
    np.random.seed(42)  # For reproducible results
    
    sample_data = []
    
    for _ in range(num_samples):
        country = np.random.choice(countries)
        
        features = {
            'gdp_growth': np.random.normal(2.5, 1.5),
            'inflation': np.random.normal(2.0, 1.0),
            'unemployment': np.random.normal(5.0, 2.0),
            'interest_rate': np.random.normal(3.0, 1.5),
            'trade_balance': np.random.normal(0.0, 3.0),
            'government_debt': np.random.normal(60.0, 15.0),
            'foreign_investment': np.random.normal(2.0, 1.5),
            'population_growth': np.random.normal(1.0, 0.5)
        }
        
        # Add some country-specific variations
        if country == 'USA':
            features['gdp_growth'] += 0.5
        elif country == 'CHN':
            features['gdp_growth'] += 1.0
        elif country == 'JPN':
            features['inflation'] -= 0.5
        
        sample_data.append({
            'country': country,
            'features': features
        })
    
    return sample_data

def load_feature_names(models_dir: str = "../models") -> List[str]:
    """Load feature names from saved file"""
    try:
        features_path = os.path.join(models_dir, "feature_names.json")
        
        if not os.path.exists(features_path):
            logger.warning(f"Feature names file not found: {features_path}")
            return []
        
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        logger.info(f"Loaded {len(feature_names)} feature names")
        return feature_names
        
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        return []

def save_feature_names(feature_names: List[str], models_dir: str = "../models"):
    """Save feature names to file"""
    try:
        os.makedirs(models_dir, exist_ok=True)
        features_path = os.path.join(models_dir, "feature_names.json")
        
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        
        logger.info(f"Saved {len(feature_names)} feature names to {features_path}")
        
    except Exception as e:
        logger.error(f"Error saving feature names: {e}")
        raise

def format_prediction_response(prediction: float, 
                             confidence: float,
                             model_name: str,
                             country: Optional[str] = None,
                             additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format prediction response with additional metadata"""
    response = {
        'prediction': round(float(prediction), 4),
        'confidence': round(float(confidence), 4),
        'model_used': model_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'country': country
    }
    
    if additional_info:
        response.update(additional_info)
    
    return response

def calculate_model_metrics(y_true: List[float], 
                          y_pred: List[float]) -> Dict[str, float]:
    """Calculate model performance metrics"""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
        
        return {
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'r2': round(r2, 4),
            'mape': round(mape, 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating model metrics: {e}")
        return {}

def create_model_comparison_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a DataFrame for model comparison"""
    try:
        comparison_data = []
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'r2' in metrics:
                row = {'Model': model_name}
                row.update(metrics)
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by R² score (descending)
        if not df.empty:
            df = df.sort_values('r2', ascending=False)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating model comparison DataFrame: {e}")
        return pd.DataFrame()

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality and return summary statistics"""
    try:
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Check for outliers in numeric columns
        outliers = {}
        for col in quality_report['numeric_columns']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            outliers[col] = outlier_count
        
        quality_report['outliers'] = outliers
        
        # Calculate basic statistics for numeric columns
        if quality_report['numeric_columns']:
            quality_report['numeric_stats'] = df[quality_report['numeric_columns']].describe().to_dict()
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Error checking data quality: {e}")
        return {}

def ensure_directory_exists(directory_path: str):
    """Ensure a directory exists, create if it doesn't"""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        raise

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            return round(size_bytes / (1024 * 1024), 2)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0.0

def cleanup_old_files(directory: str, pattern: str, max_age_days: int = 30):
    """Clean up old files in a directory"""
    try:
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        directory_path = Path(directory)
        
        if not directory_path.exists():
            return
        
        files_removed = 0
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age < cutoff_date:
                    file_path.unlink()
                    files_removed += 1
                    logger.info(f"Removed old file: {file_path}")
        
        if files_removed > 0:
            logger.info(f"Cleaned up {files_removed} old files from {directory}")
            
    except Exception as e:
        logger.error(f"Error cleaning up old files: {e}")

# Configuration utilities
def load_config(config_path: str = "../config/config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str = "../config/config.json"):
    """Save configuration to JSON file"""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise
