"""
Model Trainer module for battery monitoring system.

This module provides centralized model training capabilities for all ML models
including anomaly detection, cell prediction, and forecasting.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .cell_predictor import CellPredictor
from .forecaster import Forecaster

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.exceptions import ModelTrainingError


class ModelTrainer:
    """
    Centralized model trainer for battery monitoring system.
    
    Handles training of all ML models including anomaly detection,
    cell health prediction, and time series forecasting.
    """
    
    def __init__(self, config=None):
        """Initialize the model trainer."""
        self.config = config or get_config()
        self.logger = get_logger("model_trainer")
        self.performance_logger = get_performance_logger("model_trainer")
        
        # Initialize models
        self.anomaly_detector = AnomalyDetector(config)
        self.cell_predictor = CellPredictor(config)
        self.forecaster = Forecaster(config)
        
        # Training history
        self.training_history = []
        
        self.logger.info("ModelTrainer initialized")
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all ML models with the provided data.
        
        Args:
            df: Training DataFrame with battery data
            
        Returns:
            Dictionary with training results for all models
        """
        self.performance_logger.start_timer("train_all_models")
        
        try:
            self.logger.info("Starting training for all ML models")
            
            results = {}
            
            # Train anomaly detection model
            try:
                self.logger.info("Training anomaly detection model")
                anomaly_results = self.anomaly_detector.train(df)
                results['anomaly_detection'] = anomaly_results
                self.logger.info("Anomaly detection model training completed")
            except Exception as e:
                self.logger.error(f"Error training anomaly detection model: {e}")
                results['anomaly_detection'] = {'error': str(e)}
            
            # Train cell prediction model
            try:
                self.logger.info("Training cell prediction model")
                cell_results = self.cell_predictor.train(df)
                results['cell_prediction'] = cell_results
                self.logger.info("Cell prediction model training completed")
            except Exception as e:
                self.logger.error(f"Error training cell prediction model: {e}")
                results['cell_prediction'] = {'error': str(e)}
            
            # Train forecasting models
            try:
                self.logger.info("Training forecasting models")
                forecast_results = self.forecaster.train(df)
                results['forecasting'] = forecast_results
                self.logger.info("Forecasting models training completed")
            except Exception as e:
                self.logger.error(f"Error training forecasting models: {e}")
                results['forecasting'] = {'error': str(e)}
            
            # Save all models
            try:
                self.logger.info("Saving trained models")
                self.save_all_models()
                results['models_saved'] = True
            except Exception as e:
                self.logger.error(f"Error saving models: {e}")
                results['models_saved'] = False
                results['save_error'] = str(e)
            
            # Record training session
            training_session = {
                'timestamp': datetime.now(),
                'data_shape': df.shape,
                'results': results
            }
            self.training_history.append(training_session)
            
            self.performance_logger.end_timer("train_all_models", True)
            self.logger.info("All model training completed")
            
            return results
            
        except Exception as e:
            self.performance_logger.end_timer("train_all_models", False)
            raise ModelTrainingError(f"Error in model training: {e}")
    
    def train_anomaly_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train only the anomaly detection model."""
        try:
            self.logger.info("Training anomaly detection model")
            results = self.anomaly_detector.train(df)
            self.anomaly_detector.save_model()
            return results
        except Exception as e:
            raise ModelTrainingError(f"Error training anomaly detection model: {e}")
    
    def train_cell_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train only the cell prediction model."""
        try:
            self.logger.info("Training cell prediction model")
            results = self.cell_predictor.train(df)
            self.cell_predictor.save_model()
            return results
        except Exception as e:
            raise ModelTrainingError(f"Error training cell prediction model: {e}")
    
    def train_forecasting(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train only the forecasting models."""
        try:
            self.logger.info("Training forecasting models")
            results = self.forecaster.train(df)
            self.forecaster.save_model()
            return results
        except Exception as e:
            raise ModelTrainingError(f"Error training forecasting models: {e}")
    
    def save_all_models(self) -> Dict[str, str]:
        """Save all trained models to disk."""
        try:
            saved_paths = {}
            
            # Save anomaly detection model
            if self.anomaly_detector.isolation_forest is not None:
                path = self.anomaly_detector.save_model()
                saved_paths['anomaly_detection'] = path
            
            # Save cell prediction model
            if self.cell_predictor.classifier is not None:
                path = self.cell_predictor.save_model()
                saved_paths['cell_prediction'] = path
            
            # Save forecasting models
            if (self.forecaster.voltage_model is not None or 
                self.forecaster.temperature_model is not None or 
                self.forecaster.gravity_model is not None):
                path = self.forecaster.save_model()
                saved_paths['forecasting'] = path
            
            self.logger.info(f"All models saved: {list(saved_paths.keys())}")
            return saved_paths
            
        except Exception as e:
            raise ModelTrainingError(f"Error saving models: {e}")
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all trained models from disk."""
        try:
            load_results = {}
            
            # Load anomaly detection model
            try:
                success = self.anomaly_detector.load_model()
                load_results['anomaly_detection'] = success
            except Exception as e:
                self.logger.warning(f"Could not load anomaly detection model: {e}")
                load_results['anomaly_detection'] = False
            
            # Load cell prediction model
            try:
                success = self.cell_predictor.load_model()
                load_results['cell_prediction'] = success
            except Exception as e:
                self.logger.warning(f"Could not load cell prediction model: {e}")
                load_results['cell_prediction'] = False
            
            # Load forecasting models
            try:
                success = self.forecaster.load_model()
                load_results['forecasting'] = success
            except Exception as e:
                self.logger.warning(f"Could not load forecasting models: {e}")
                load_results['forecasting'] = False
            
            self.logger.info(f"Model loading completed: {load_results}")
            return load_results
            
        except Exception as e:
            raise ModelTrainingError(f"Error loading models: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        return {
            'anomaly_detection': {
                'trained': self.anomaly_detector.isolation_forest is not None,
                'model_type': 'isolation_forest' if self.anomaly_detector.isolation_forest else None
            },
            'cell_prediction': {
                'trained': self.cell_predictor.classifier is not None,
                'model_type': 'random_forest_classifier' if self.cell_predictor.classifier else None
            },
            'forecasting': {
                'trained': (self.forecaster.voltage_model is not None or 
                           self.forecaster.temperature_model is not None or 
                           self.forecaster.gravity_model is not None),
                'models': {
                    'voltage': self.forecaster.voltage_model is not None,
                    'temperature': self.forecaster.temperature_model is not None,
                    'specific_gravity': self.forecaster.gravity_model is not None
                }
            }
        }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        return {
            'anomaly_detection': self.anomaly_detector.get_performance_metrics(),
            'cell_prediction': self.cell_predictor.get_performance_metrics(),
            'forecasting': self.forecaster.get_performance_metrics()
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics for all models."""
        self.anomaly_detector.reset_metrics()
        self.cell_predictor.reset_metrics()
        self.forecaster.reset_metrics()
        self.logger.info("All model performance metrics reset") 