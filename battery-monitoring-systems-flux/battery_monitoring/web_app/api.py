"""
Web API module for battery monitoring system.

This module provides REST API endpoints for battery monitoring
data access, ML predictions, and system management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.database import get_database_manager
from ..data.loader import DataLoader
from ..ml.anomaly_detector import AnomalyDetector
from ..ml.cell_predictor import CellPredictor
from ..ml.forecaster import Forecaster
from ..llm.chatbot import BatteryChatbot


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    device_id: Optional[int] = None
    cell_number: Optional[int] = None
    analysis_type: str = "all"  # "anomaly", "prediction", "forecast", "all"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    logger = get_logger("web_api")
    
    # Create FastAPI app
    app = FastAPI(
        title=config.web_app.api_title,
        description=config.web_app.api_description,
        version=config.web_app.api_version,
        docs_url=config.web_app.docs_url,
        redoc_url=config.web_app.redoc_url
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins,
        allow_credentials=True,
        allow_methods=config.security.cors_methods,
        allow_headers=config.security.cors_headers,
    )
    
    # Initialize components
    db_manager = get_database_manager()
    data_loader = DataLoader()
    anomaly_detector = AnomalyDetector()
    cell_predictor = CellPredictor()
    forecaster = Forecaster()
    chatbot = BatteryChatbot()
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": config.app_version
        }
    
    # System status endpoint
    @app.get("/status")
    async def system_status():
        """Get system status and component health."""
        try:
            # Check database
            db_stats = db_manager.get_database_stats()
            
            # Check ML models
            model_status = {
                'anomaly_detection': anomaly_detector.isolation_forest is not None,
                'cell_prediction': cell_predictor.classifier is not None,
                'forecasting': (forecaster.voltage_model is not None or 
                               forecaster.temperature_model is not None or 
                               forecaster.gravity_model is not None)
            }
            
            return {
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "database": db_stats,
                "models": model_status,
                "llm_provider": chatbot.provider,
                "llm_model": chatbot.model
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Data endpoints
    @app.get("/data/battery")
    async def get_battery_data(
        device_id: Optional[int] = None,
        cell_number: Optional[int] = None,
        limit: int = 100
    ):
        """Get battery data with optional filtering."""
        try:
            df = data_loader.load_from_database(
                device_id=device_id,
                cell_number=cell_number,
                limit=limit
            )
            
            return {
                "data": df.to_dict(orient='records'),
                "total_records": len(df),
                "columns": df.columns.tolist()
            }
        except Exception as e:
            logger.error(f"Error loading battery data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/data/summary")
    async def get_data_summary():
        """Get data summary statistics."""
        try:
            df = data_loader.load_from_database(limit=1000)
            summary = data_loader.get_data_summary(df)
            
            return summary
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ML analysis endpoints
    @app.post("/analysis")
    async def run_analysis(request: AnalysisRequest):
        """Run ML analysis on battery data."""
        try:
            # Load data
            df = data_loader.load_from_database(
                device_id=request.device_id,
                cell_number=request.cell_number,
                limit=1000
            )
            
            if len(df) == 0:
                raise HTTPException(status_code=404, detail="No data found for the specified criteria")
            
            results = {}
            
            # Run anomaly detection
            if request.analysis_type in ["anomaly", "all"]:
                try:
                    anomaly_results = anomaly_detector.detect_anomalies(df)
                    results["anomaly_detection"] = {
                        "anomaly_count": int(anomaly_results['is_anomaly'].sum()),
                        "total_samples": len(anomaly_results),
                        "anomaly_rate": float(anomaly_results['is_anomaly'].mean()),
                        "avg_anomaly_score": float(anomaly_results['anomaly_score'].mean())
                    }
                except Exception as e:
                    results["anomaly_detection"] = {"error": str(e)}
            
            # Run cell prediction
            if request.analysis_type in ["prediction", "all"]:
                try:
                    prediction_results = cell_predictor.predict(df)
                    results["cell_prediction"] = {
                        "total_samples": len(prediction_results),
                        "alive_predictions": int(prediction_results['is_alive'].sum()),
                        "dead_predictions": int(prediction_results['is_dead'].sum()),
                        "avg_confidence": float(prediction_results['prediction_confidence'].mean())
                    }
                except Exception as e:
                    results["cell_prediction"] = {"error": str(e)}
            
            # Run forecasting
            if request.analysis_type in ["forecast", "all"]:
                try:
                    forecast_results = forecaster.forecast(df)
                    results["forecasting"] = forecast_results
                except Exception as e:
                    results["forecasting"] = {"error": str(e)}
            
            return {
                "analysis_type": request.analysis_type,
                "data_shape": df.shape,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # LLM chat endpoint
    @app.post("/chat")
    async def chat(request: ChatRequest):
        """Chat with the LLM about battery monitoring."""
        try:
            response = chatbot.chat(request.message, request.context)
            
            return {
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "model": chatbot.model,
                "provider": chatbot.provider
            }
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Model management endpoints
    @app.post("/models/train")
    async def train_models(background_tasks: BackgroundTasks):
        """Train all ML models in the background."""
        try:
            # Load training data
            df = data_loader.load_from_database(limit=5000)
            
            if len(df) == 0:
                raise HTTPException(status_code=404, detail="No training data available")
            
            # Start background training
            background_tasks.add_task(train_models_background, df)
            
            return {
                "message": "Model training started in background",
                "data_shape": df.shape,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error starting model training: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/status")
    async def get_model_status():
        """Get status of all ML models."""
        try:
            return {
                "anomaly_detection": {
                    "trained": anomaly_detector.isolation_forest is not None,
                    "model_type": "isolation_forest" if anomaly_detector.isolation_forest else None
                },
                "cell_prediction": {
                    "trained": cell_predictor.classifier is not None,
                    "model_type": "random_forest_classifier" if cell_predictor.classifier else None
                },
                "forecasting": {
                    "trained": (forecaster.voltage_model is not None or 
                               forecaster.temperature_model is not None or 
                               forecaster.gravity_model is not None),
                    "models": {
                        "voltage": forecaster.voltage_model is not None,
                        "temperature": forecaster.temperature_model is not None,
                        "specific_gravity": forecaster.gravity_model is not None
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Performance metrics endpoints
    @app.get("/metrics")
    async def get_performance_metrics():
        """Get performance metrics for all components."""
        try:
            return {
                "anomaly_detection": anomaly_detector.get_performance_metrics(),
                "cell_prediction": cell_predictor.get_performance_metrics(),
                "forecasting": forecaster.get_performance_metrics(),
                "chatbot": chatbot.get_performance_metrics(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
        )
    
    logger.info("FastAPI application created successfully")
    return app


def train_models_background(df):
    """Background task for training models."""
    logger = get_logger("web_api")
    
    try:
        logger.info("Starting background model training")
        
        # Train anomaly detection
        anomaly_detector = AnomalyDetector()
        anomaly_results = anomaly_detector.train(df)
        anomaly_detector.save_model()
        logger.info("Anomaly detection model trained")
        
        # Train cell prediction
        cell_predictor = CellPredictor()
        cell_results = cell_predictor.train(df)
        cell_predictor.save_model()
        logger.info("Cell prediction model trained")
        
        # Train forecasting
        forecaster = Forecaster()
        forecast_results = forecaster.train(df)
        forecaster.save_model()
        logger.info("Forecasting models trained")
        
        logger.info("Background model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background model training: {e}") 