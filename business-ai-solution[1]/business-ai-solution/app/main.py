from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import pickle
import os
import logging
from datetime import datetime
import time

from .logger import get_logger
from .monitor import PerformanceMonitor
from .utils import load_model, validate_input_data

# Initialize FastAPI app
app = FastAPI(
    title="Economic Indicator Prediction API",
    description="AI-powered economic indicator prediction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger and monitor
logger = get_logger(__name__)
monitor = PerformanceMonitor()

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        model = load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    country: Optional[str] = None
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_used: str
    timestamp: str
    country: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    processing_time: float

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Economic Indicator Prediction API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make prediction for a single data point"""
    start_time = time.time()
    
    try:
        # Validate input data
        validate_input_data(request.features)
        
        # Make prediction
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        prediction = model.predict([list(request.features.values())])[0]
        confidence = 0.85  # Mock confidence score
        
        # Log prediction
        logger.info(f"Prediction made for country: {request.country}, value: {prediction}")
        
        # Monitor performance
        processing_time = time.time() - start_time
        monitor.record_prediction_time(processing_time)
        monitor.record_prediction_accuracy(0.85)  # Mock accuracy
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=confidence,
            model_used="RandomForest",
            timestamp=datetime.now().isoformat(),
            country=request.country
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        monitor.record_error()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple data points"""
    start_time = time.time()
    
    try:
        predictions = []
        
        for pred_request in request.predictions:
            # Validate input data
            validate_input_data(pred_request.features)
            
            # Make prediction
            if model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            prediction = model.predict([list(pred_request.features.values())])[0]
            confidence = 0.85  # Mock confidence score
            
            predictions.append(PredictionResponse(
                prediction=float(prediction),
                confidence=confidence,
                model_used="RandomForest",
                timestamp=datetime.now().isoformat(),
                country=pred_request.country
            ))
        
        processing_time = time.time() - start_time
        
        # Log batch prediction
        logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        
        # Monitor performance
        monitor.record_batch_prediction_time(processing_time)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        monitor.record_error()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/country/{country}")
async def predict_by_country(country: str):
    """Get predictions for a specific country using historical data"""
    try:
        # Load country-specific data
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{country.lower()}_data.csv")
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
        
        # Load and preprocess data
        df = pd.read_csv(data_path)
        
        # Make predictions for the country
        predictions = []
        for _, row in df.tail(10).iterrows():  # Last 10 data points
            features = row.drop(['target', 'date']).to_dict()
            prediction = model.predict([list(features.values())])[0]
            
            predictions.append({
                "date": row['date'],
                "prediction": float(prediction),
                "actual": float(row['target']) if 'target' in row else None
            })
        
        return {
            "country": country,
            "predictions": predictions,
            "model_used": "RandomForest",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Country prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/all-countries")
async def predict_all_countries():
    """Get predictions for all available countries"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        countries = []
        
        # Find all country data files
        for file in os.listdir(data_dir):
            if file.endswith("_data.csv"):
                country = file.replace("_data.csv", "").title()
                countries.append(country)
        
        all_predictions = {}
        
        for country in countries:
            try:
                data_path = os.path.join(data_dir, f"{country.lower()}_data.csv")
                df = pd.read_csv(data_path)
                
                # Make prediction for latest data point
                latest_row = df.iloc[-1]
                features = latest_row.drop(['target', 'date']).to_dict()
                prediction = model.predict([list(features.values())])[0]
                
                all_predictions[country] = {
                    "prediction": float(prediction),
                    "confidence": 0.85,
                    "last_updated": latest_row['date']
                }
                
            except Exception as e:
                logger.warning(f"Failed to predict for {country}: {e}")
                all_predictions[country] = {"error": str(e)}
        
        return {
            "predictions": all_predictions,
            "total_countries": len(countries),
            "model_used": "RandomForest",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"All countries prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_performance_metrics():
    """Get performance monitoring metrics"""
    return monitor.get_metrics()

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "RandomForest",
        "feature_count": len(model.feature_importances_) if hasattr(model, 'feature_importances_') else "Unknown",
        "training_date": "2024-01-01",  # Mock date
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
