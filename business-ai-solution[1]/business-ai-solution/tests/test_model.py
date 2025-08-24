import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from train_model import ModelTrainer
from utils import (
    load_model, save_model, validate_input_data, normalize_features,
    calculate_prediction_confidence, create_sample_data, calculate_model_metrics
)

class TestModelTraining:
    """Test cases for model training functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 8
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 10 + 100  # Target values around 100
        
        # Create feature names
        feature_names = [
            'gdp_growth', 'inflation', 'unemployment', 'interest_rate',
            'trade_balance', 'government_debt', 'foreign_investment', 'population_growth'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['country'] = np.random.choice(['USA', 'CHN', 'JPN'], n_samples)
        df['date'] = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        
        return df
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_model_trainer_initialization(self, temp_data_dir, temp_models_dir):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        assert trainer.data_dir == Path(temp_data_dir)
        assert trainer.models_dir == Path(temp_models_dir)
        assert len(trainer.models) > 0
        assert len(trainer.param_grids) > 0
        assert trainer.results == {}
        assert trainer.best_model is None
    
    def test_load_data_with_combined_dataset(self, temp_data_dir, temp_models_dir, sample_data):
        """Test loading data from combined dataset"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Save sample data as combined dataset
        combined_path = Path(temp_data_dir) / "combined_data.csv"
        sample_data.to_csv(combined_path, index=False)
        
        X, y = trainer.load_data()
        
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert len(X.columns) == 8  # Number of features
        assert 'target' not in X.columns
        assert 'country' not in X.columns
        assert 'date' not in X.columns
    
    def test_load_data_with_individual_files(self, temp_data_dir, temp_models_dir, sample_data):
        """Test loading data from individual country files"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Save individual country files
        for country in ['USA', 'CHN', 'JPN']:
            country_data = sample_data[sample_data['country'] == country]
            if len(country_data) > 0:
                country_path = Path(temp_data_dir) / f"{country.lower()}_data.csv"
                country_data.to_csv(country_path, index=False)
        
        X, y = trainer.load_data()
        
        assert len(X) > 0
        assert len(y) > 0
        assert len(X.columns) == 8
    
    def test_load_data_no_files(self, temp_data_dir, temp_models_dir):
        """Test loading data when no files exist"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        with pytest.raises(FileNotFoundError):
            trainer.load_data()
    
    def test_create_baseline_model(self, temp_data_dir, temp_models_dir, sample_data):
        """Test baseline model creation"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Split data
        train_size = int(0.8 * len(sample_data))
        y_train = sample_data['target'].iloc[:train_size]
        y_test = sample_data['target'].iloc[train_size:]
        
        metrics = trainer.create_baseline_model(y_train, y_test)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert isinstance(metrics['r2'], float)
        assert 'Baseline' in trainer.results
    
    def test_train_and_evaluate_model(self, temp_data_dir, temp_models_dir, sample_data):
        """Test training and evaluating a single model"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Prepare data
        feature_cols = [col for col in sample_data.columns if col not in ['target', 'country', 'date']]
        X = sample_data[feature_cols]
        y = sample_data['target']
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        
        # Test with Linear Regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        metrics = trainer.train_and_evaluate_model(
            'Linear Regression', model, X_train, X_test, y_train, y_test
        )
        
        assert metrics is not None
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'model' in metrics
        assert 'Linear Regression' in trainer.results
    
    def test_optimize_hyperparameters(self, temp_data_dir, temp_models_dir, sample_data):
        """Test hyperparameter optimization"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Prepare data
        feature_cols = [col for col in sample_data.columns if col not in ['target', 'country', 'date']]
        X = sample_data[feature_cols]
        y = sample_data['target']
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        # Test with Ridge Regression
        from sklearn.linear_model import Ridge
        model = Ridge()
        
        optimized_model = trainer.optimize_hyperparameters(
            'Ridge Regression', model, X_train, y_train
        )
        
        assert optimized_model is not None
        assert hasattr(optimized_model, 'fit')
        assert hasattr(optimized_model, 'predict')
    
    def test_find_best_model(self, temp_data_dir, temp_models_dir):
        """Test finding the best model"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Add some mock results
        trainer.results = {
            'Model A': {'r2': 0.8, 'rmse': 2.0, 'mae': 1.5},
            'Model B': {'r2': 0.9, 'rmse': 1.5, 'mae': 1.2},
            'Model C': {'r2': 0.7, 'rmse': 2.5, 'mae': 1.8}
        }
        
        trainer.find_best_model()
        
        assert trainer.best_model_name == 'Model B'
        assert trainer.best_model is not None
    
    def test_save_results(self, temp_data_dir, temp_models_dir, sample_data):
        """Test saving training results"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Mock best model
        mock_model = Mock()
        trainer.best_model = mock_model
        trainer.best_model_name = 'Test Model'
        
        # Add some results
        trainer.results = {
            'Test Model': {
                'r2': 0.85,
                'rmse': 1.5,
                'mae': 1.2,
                'model': mock_model
            }
        }
        
        trainer.save_results()
        
        # Check if files were created
        results_file = Path(temp_models_dir) / "training_results.json"
        model_file = Path(temp_models_dir) / "best_model.pkl"
        features_file = Path(temp_models_dir) / "feature_names.json"
        
        assert results_file.exists()
        assert model_file.exists()
        assert features_file.exists()
    
    def test_create_model_comparison_plot(self, temp_data_dir, temp_models_dir):
        """Test creating model comparison plot"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Add mock results
        trainer.results = {
            'Model A': {'r2': 0.8, 'rmse': 2.0},
            'Model B': {'r2': 0.9, 'rmse': 1.5},
            'Model C': {'r2': 0.7, 'rmse': 2.5}
        }
        
        trainer.create_model_comparison_plot()
        
        plot_file = Path(temp_models_dir) / "model_comparison.png"
        assert plot_file.exists()
    
    def test_generate_training_report(self, temp_data_dir, temp_models_dir):
        """Test generating training report"""
        trainer = ModelTrainer(data_dir=temp_data_dir, models_dir=temp_models_dir)
        
        # Add mock results
        trainer.results = {
            'Model A': {'r2': 0.8, 'rmse': 2.0, 'mae': 1.5},
            'Model B': {'r2': 0.9, 'rmse': 1.5, 'mae': 1.2}
        }
        trainer.best_model_name = 'Model B'
        
        report = trainer.generate_training_report()
        
        assert 'training_date' in report
        assert 'best_model' in report
        assert 'total_models_tested' in report
        assert 'model_performance' in report
        assert 'recommendations' in report
        assert report['best_model'] == 'Model B'
        assert report['total_models_tested'] == 2

class TestModelUtils:
    """Test cases for model utility functions"""
    
    def test_load_model(self, temp_models_dir):
        """Test loading a model from file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock model
            mock_model = Mock()
            mock_model.predict.return_value = [95.5]
            
            # Save model
            model_path = os.path.join(temp_dir, "test_model.pkl")
            save_model(mock_model, model_path)
            
            # Load model
            loaded_model = load_model(model_path)
            
            assert loaded_model is not None
            assert hasattr(loaded_model, 'predict')
            assert loaded_model.predict([1, 2, 3]) == [95.5]
    
    def test_load_model_file_not_found(self):
        """Test loading model from non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_model("non_existent_model.pkl")
    
    def test_save_model(self, temp_models_dir):
        """Test saving a model to file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock model
            mock_model = Mock()
            
            # Save model
            model_path = os.path.join(temp_dir, "test_model.pkl")
            save_model(mock_model, model_path)
            
            # Check if file exists
            assert os.path.exists(model_path)
    
    def test_validate_input_data_valid(self):
        """Test input validation with valid data"""
        features = {
            'gdp_growth': 2.5,
            'inflation': 2.0,
            'unemployment': 5.0
        }
        
        assert validate_input_data(features) is True
    
    def test_validate_input_data_invalid_type(self):
        """Test input validation with invalid data type"""
        features = "not a dict"
        
        with pytest.raises(ValueError, match="Features must be a dictionary"):
            validate_input_data(features)
    
    def test_validate_input_data_empty(self):
        """Test input validation with empty dictionary"""
        features = {}
        
        with pytest.raises(ValueError, match="Features dictionary cannot be empty"):
            validate_input_data(features)
    
    def test_validate_input_data_non_numeric(self):
        """Test input validation with non-numeric values"""
        features = {
            'gdp_growth': "invalid",
            'inflation': 2.0
        }
        
        with pytest.raises(ValueError, match="must be a valid numeric value"):
            validate_input_data(features)
    
    def test_validate_input_data_nan(self):
        """Test input validation with NaN values"""
        features = {
            'gdp_growth': float('nan'),
            'inflation': 2.0
        }
        
        with pytest.raises(ValueError, match="must be a valid numeric value"):
            validate_input_data(features)
    
    def test_validate_input_data_inf(self):
        """Test input validation with infinite values"""
        features = {
            'gdp_growth': float('inf'),
            'inflation': 2.0
        }
        
        with pytest.raises(ValueError, match="must be a valid numeric value"):
            validate_input_data(features)
    
    def test_normalize_features(self):
        """Test feature normalization"""
        features = {
            'gdp_growth': 2.5,
            'inflation': 2.0,
            'unemployment': 5.0
        }
        
        normalized = normalize_features(features)
        
        assert isinstance(normalized, dict)
        assert 'gdp_growth' in normalized
        assert 'inflation' in normalized
        assert 'unemployment' in normalized
        assert all(isinstance(v, float) for v in normalized.values())
    
    def test_normalize_features_with_stats(self):
        """Test feature normalization with custom statistics"""
        features = {
            'gdp_growth': 2.5,
            'inflation': 2.0
        }
        
        feature_stats = {
            'gdp_growth': {'mean': 2.0, 'std': 1.0},
            'inflation': {'mean': 1.5, 'std': 0.5}
        }
        
        normalized = normalize_features(features, feature_stats)
        
        # Check that normalization was applied correctly
        expected_gdp = (2.5 - 2.0) / 1.0
        expected_inflation = (2.0 - 1.5) / 0.5
        
        assert abs(normalized['gdp_growth'] - expected_gdp) < 1e-6
        assert abs(normalized['inflation'] - expected_inflation) < 1e-6
    
    def test_calculate_prediction_confidence(self):
        """Test prediction confidence calculation"""
        prediction = 95.5
        historical_predictions = [94.0, 95.0, 96.0, 97.0]
        model_accuracy = 0.85
        
        confidence = calculate_prediction_confidence(
            prediction, historical_predictions, model_accuracy
        )
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_calculate_prediction_confidence_no_history(self):
        """Test prediction confidence with no historical data"""
        prediction = 95.5
        historical_predictions = []
        model_accuracy = 0.85
        
        confidence = calculate_prediction_confidence(
            prediction, historical_predictions, model_accuracy
        )
        
        assert confidence == model_accuracy
    
    def test_create_sample_data(self):
        """Test sample data creation"""
        countries = ['USA', 'CHN', 'JPN']
        num_samples = 50
        
        sample_data = create_sample_data(countries, num_samples)
        
        assert len(sample_data) == num_samples
        assert all('country' in item for item in sample_data)
        assert all('features' in item for item in sample_data)
        assert all(item['country'] in countries for item in sample_data)
    
    def test_calculate_model_metrics(self):
        """Test model metrics calculation"""
        y_true = [100, 101, 102, 103, 104]
        y_pred = [99, 101, 102, 104, 105]
        
        metrics = calculate_model_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_calculate_model_metrics_perfect_prediction(self):
        """Test model metrics with perfect predictions"""
        y_true = [100, 101, 102, 103, 104]
        y_pred = [100, 101, 102, 103, 104]
        
        metrics = calculate_model_metrics(y_true, y_pred)
        
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['r2'] == 1.0
        assert metrics['mape'] == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
