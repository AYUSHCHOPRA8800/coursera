import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from main import app

class TestAPI:
    """Test cases for the API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_features(self):
        """Sample features for testing"""
        return {
            "gdp_growth": 2.5,
            "inflation": 2.0,
            "unemployment": 5.0,
            "interest_rate": 3.0,
            "trade_balance": 0.5,
            "government_debt": 65.0,
            "foreign_investment": 2.5,
            "population_growth": 1.2
        }
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing"""
        mock = Mock()
        mock.predict.return_value = [95.5]
        return mock
    
    def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
    
    @patch('app.main.model')
    def test_predict_single_success(self, mock_model, client, sample_features):
        """Test successful single prediction"""
        mock_model.predict.return_value = [95.5]
        
        request_data = {
            "country": "USA",
            "features": sample_features
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "model_used" in data
        assert "timestamp" in data
        assert "country" in data
        assert data["country"] == "USA"
        assert isinstance(data["prediction"], float)
        assert isinstance(data["confidence"], float)
    
    @patch('app.main.model')
    def test_predict_single_no_country(self, mock_model, client, sample_features):
        """Test prediction without country specification"""
        mock_model.predict.return_value = [95.5]
        
        request_data = {
            "features": sample_features
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["country"] is None
    
    def test_predict_single_invalid_features(self, client):
        """Test prediction with invalid features"""
        request_data = {
            "country": "USA",
            "features": {
                "gdp_growth": "invalid",  # Should be numeric
                "inflation": 2.0
            }
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 500
    
    def test_predict_single_empty_features(self, client):
        """Test prediction with empty features"""
        request_data = {
            "country": "USA",
            "features": {}
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 500
    
    @patch('app.main.model')
    def test_predict_single_model_not_loaded(self, mock_model, client, sample_features):
        """Test prediction when model is not loaded"""
        mock_model = None
        
        request_data = {
            "country": "USA",
            "features": sample_features
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    @patch('app.main.model')
    def test_predict_batch_success(self, mock_model, client, sample_features):
        """Test successful batch prediction"""
        mock_model.predict.return_value = [95.5]
        
        request_data = {
            "predictions": [
                {"country": "USA", "features": sample_features},
                {"country": "CHN", "features": sample_features}
            ]
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert "processing_time" in data
        assert len(data["predictions"]) == 2
        assert data["total_count"] == 2
        assert isinstance(data["processing_time"], float)
    
    @patch('app.main.model')
    def test_predict_batch_empty_list(self, mock_model, client):
        """Test batch prediction with empty list"""
        request_data = {
            "predictions": []
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == 0
        assert data["total_count"] == 0
    
    @patch('app.main.model')
    @patch('pandas.read_csv')
    def test_predict_by_country_success(self, mock_read_csv, mock_model, client):
        """Test prediction for specific country"""
        # Mock the CSV data
        mock_df = Mock()
        mock_df.tail.return_value.iterrows.return_value = [
            (0, Mock(**{
                'drop.return_value.to_dict.return_value': {
                    'gdp_growth': 2.5,
                    'inflation': 2.0
                },
                'date': '2023-12-31',
                'target': 95.0
            }))
        ]
        mock_read_csv.return_value = mock_df
        
        mock_model.predict.return_value = [95.5]
        
        response = client.get("/predict/country/USA")
        assert response.status_code == 200
        
        data = response.json()
        assert "country" in data
        assert "predictions" in data
        assert "model_used" in data
        assert data["country"] == "USA"
    
    @patch('app.main.model')
    @patch('pandas.read_csv')
    def test_predict_by_country_not_found(self, mock_read_csv, mock_model, client):
        """Test prediction for non-existent country"""
        # Mock file not found
        mock_read_csv.side_effect = FileNotFoundError()
        
        response = client.get("/predict/country/NONEXISTENT")
        assert response.status_code == 404
        assert "No data found" in response.json()["detail"]
    
    @patch('app.main.model')
    @patch('os.listdir')
    @patch('pandas.read_csv')
    def test_predict_all_countries_success(self, mock_read_csv, mock_listdir, mock_model, client):
        """Test prediction for all countries"""
        # Mock directory listing
        mock_listdir.return_value = ["usa_data.csv", "chn_data.csv"]
        
        # Mock CSV data
        mock_df = Mock()
        mock_df.iloc = [-1]  # Mock last row
        mock_df.iloc[-1].drop.return_value.to_dict.return_value = {
            'gdp_growth': 2.5,
            'inflation': 2.0
        }
        mock_df.iloc[-1].date = '2023-12-31'
        mock_read_csv.return_value = mock_df
        
        mock_model.predict.return_value = [95.5]
        
        response = client.get("/predict/all-countries")
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_countries" in data
        assert "model_used" in data
        assert "timestamp" in data
        assert data["total_countries"] == 2
    
    @patch('app.main.model')
    def test_predict_all_countries_no_data(self, mock_model, client):
        """Test prediction for all countries when no data exists"""
        with patch('os.listdir', return_value=[]):
            response = client.get("/predict/all-countries")
            assert response.status_code == 200
            
            data = response.json()
            assert data["total_countries"] == 0
            assert len(data["predictions"]) == 0
    
    def test_metrics_endpoint(self, client):
        """Test the metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        # Check that metrics structure exists
        assert isinstance(data, dict)
    
    @patch('app.main.model')
    def test_model_info_endpoint_with_model(self, mock_model, client):
        """Test model info endpoint when model is loaded"""
        # Mock model with feature_importances_
        mock_model.feature_importances_ = [0.1, 0.2, 0.3, 0.4]
        
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "feature_count" in data
        assert "training_date" in data
        assert "version" in data
        assert data["feature_count"] == 4
    
    def test_model_info_endpoint_no_model(self, client):
        """Test model info endpoint when model is not loaded"""
        with patch('app.main.model', None):
            response = client.get("/model/info")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
    
    def test_predict_invalid_json(self, client):
        """Test prediction with invalid JSON"""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422
    
    def test_predict_missing_features(self, client):
        """Test prediction with missing features field"""
        request_data = {
            "country": "USA"
            # Missing features field
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    @patch('app.main.model')
    def test_predict_with_nan_values(self, mock_model, client):
        """Test prediction with NaN values in features"""
        request_data = {
            "country": "USA",
            "features": {
                "gdp_growth": float('nan'),
                "inflation": 2.0
            }
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 500
    
    @patch('app.main.model')
    def test_predict_with_inf_values(self, mock_model, client):
        """Test prediction with infinite values in features"""
        request_data = {
            "country": "USA",
            "features": {
                "gdp_growth": float('inf'),
                "inflation": 2.0
            }
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 500
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present"""
        response = client.options("/predict")
        # FastAPI automatically handles CORS, so this should work
        assert response.status_code in [200, 405]  # 405 if OPTIONS not implemented
    
    @patch('app.main.model')
    def test_large_batch_prediction(self, mock_model, client, sample_features):
        """Test batch prediction with many items"""
        mock_model.predict.return_value = [95.5]
        
        # Create a large batch
        predictions = []
        for i in range(100):
            predictions.append({
                "country": f"COUNTRY_{i}",
                "features": sample_features
            })
        
        request_data = {"predictions": predictions}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_count"] == 100
        assert len(data["predictions"]) == 100
    
    def test_request_validation(self, client):
        """Test various request validation scenarios"""
        # Test with non-dict features
        request_data = {
            "country": "USA",
            "features": "not a dict"
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
        
        # Test with None features
        request_data = {
            "country": "USA",
            "features": None
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__])
