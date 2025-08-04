"""
Anomaly Detection module for battery monitoring system.

This module provides anomaly detection capabilities for battery cell data
including voltage, temperature, and specific gravity anomalies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.exceptions import ModelError, ModelTrainingError, ModelInferenceError


class AnomalyDetector:
    """
    Anomaly detection for battery cell data.
    
    Detects anomalies in voltage, temperature, and specific gravity readings
    using isolation forest and statistical methods.
    """
    
    def __init__(self, config=None):
        """Initialize the anomaly detector."""
        self.config = config or get_config()
        self.logger = get_logger("anomaly_detection")
        self.performance_logger = get_performance_logger("anomaly_detection")
        
        # Models
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.pca = None
        
        # Training data
        self.training_data = None
        self.feature_columns = None
        
        # Performance metrics
        self.anomaly_scores = []
        self.detection_times = []
        
        self.logger.info("AnomalyDetector initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection.
        
        Args:
            df: Input DataFrame with battery data
            
        Returns:
            DataFrame with prepared features
        """
        self.performance_logger.start_timer("prepare_features")
        
        try:
            # Select key columns
            key_columns = self.config.data.key_columns
            available_columns = [col for col in key_columns if col in df.columns]
            
            # Debug logging
            self.logger.info(f"Available columns in DataFrame: {list(df.columns)}")
            self.logger.info(f"Key columns from config: {key_columns}")
            self.logger.info(f"Available key columns: {available_columns}")
            
            if len(available_columns) < 3:
                raise ModelError(f"Insufficient columns for anomaly detection. Need at least 3, got {len(available_columns)}")
            
            # Create features - only use numeric columns
            numeric_columns = df[available_columns].select_dtypes(include=[np.number]).columns
            features = df[numeric_columns].copy()
            
            # Add derived features for numeric columns only
            if 'cell_voltage' in features.columns:
                features['voltage_diff'] = features['cell_voltage'].diff()
                features['voltage_rolling_mean'] = features['cell_voltage'].rolling(window=5, min_periods=1).mean()
                features['voltage_rolling_std'] = features['cell_voltage'].rolling(window=5, min_periods=1).std()
            
            if 'cell_temperature' in features.columns:
                features['temp_diff'] = features['cell_temperature'].diff()
                features['temp_rolling_mean'] = features['cell_temperature'].rolling(window=5, min_periods=1).mean()
                features['temp_rolling_std'] = features['cell_temperature'].rolling(window=5, min_periods=1).std()
            
            if 'cell_specific_gravity' in features.columns:
                features['gravity_diff'] = features['cell_specific_gravity'].diff()
                features['gravity_rolling_mean'] = features['cell_specific_gravity'].rolling(window=5, min_periods=1).mean()
                features['gravity_rolling_std'] = features['cell_specific_gravity'].rolling(window=5, min_periods=1).std()
            
            # Fill NaN values
            features = features.ffill().bfill().fillna(0)
            
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            self.feature_columns = features.columns.tolist()
            
            self.performance_logger.end_timer("prepare_features", True)
            self.logger.info(f"Prepared {len(features)} features for anomaly detection")
            
            return features
            
        except Exception as e:
            self.performance_logger.end_timer("prepare_features", False)
            raise ModelError(f"Error preparing features: {e}")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the anomaly detection model.
        
        Args:
            df: Training DataFrame with battery data
            
        Returns:
            Dictionary with training results
        """
        self.performance_logger.start_timer("train_anomaly_model")
        
        try:
            self.logger.info("Starting anomaly detection model training")
            
            # Prepare features
            features = self.prepare_features(df)
            self.training_data = features
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Initialize and train isolation forest
            self.isolation_forest = IsolationForest(
                contamination=self.config.ml.contamination,
                n_estimators=self.config.ml.n_estimators,
                max_samples=self.config.ml.max_samples,
                random_state=self.config.ml.random_state
            )
            
            # Optional: Add PCA for dimensionality reduction
            if features.shape[1] > 10:
                self.pca = PCA(n_components=min(10, features.shape[1]))
                pca_features = self.pca.fit_transform(scaled_features)
                self.logger.info(f"Applied PCA: {features.shape[1]} -> {pca_features.shape[1]} features")
                # Train the model on PCA-transformed features
                self.isolation_forest.fit(pca_features)
            else:
                # Train the model on original scaled features
                self.isolation_forest.fit(scaled_features)
            
            # Calculate training metrics
            if self.pca is not None:
                train_scores = self.isolation_forest.decision_function(pca_features)
                train_predictions = self.isolation_forest.predict(pca_features)
            else:
                train_scores = self.isolation_forest.decision_function(scaled_features)
                train_predictions = self.isolation_forest.predict(scaled_features)
            
            # Training results
            results = {
                'model_type': 'isolation_forest',
                'n_samples': len(features),
                'n_features': features.shape[1],
                'contamination': self.config.ml.contamination,
                'n_estimators': self.config.ml.n_estimators,
                'training_date': datetime.now(),
                'feature_columns': self.feature_columns,
                'anomaly_count': np.sum(train_predictions == -1),
                'normal_count': np.sum(train_predictions == 1),
                'avg_anomaly_score': np.mean(train_scores),
                'std_anomaly_score': np.std(train_scores)
            }
            
            self.performance_logger.end_timer("train_anomaly_model", True)
            self.logger.info(f"Anomaly detection model trained successfully: {results['anomaly_count']} anomalies detected")
            
            return results
            
        except Exception as e:
            self.performance_logger.end_timer("train_anomaly_model", False)
            raise ModelTrainingError(f"Error training anomaly detection model: {e}")
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in battery data.
        
        Args:
            df: DataFrame with battery data to analyze
            
        Returns:
            DataFrame with anomaly detection results
        """
        self.performance_logger.start_timer("detect_anomalies")
        
        try:
            if self.isolation_forest is None:
                raise ModelError("Model not trained. Call train() first.")
            
            self.logger.info(f"Detecting anomalies in {len(df)} samples")
            
            # Prepare features
            features = self.prepare_features(df)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Apply PCA if available
            if self.pca is not None:
                scaled_features = self.pca.transform(scaled_features)
            
            # Detect anomalies
            anomaly_scores = self.isolation_forest.decision_function(scaled_features)
            anomaly_predictions = self.isolation_forest.predict(scaled_features)
            
            # Create results DataFrame
            results = df.copy()
            results['anomaly_score'] = anomaly_scores
            results['is_anomaly'] = anomaly_predictions == -1
            results['anomaly_confidence'] = np.abs(anomaly_scores)
            
            # Add specific anomaly flags
            if 'CellVoltage' in df.columns:
                voltage_range = self.config.data.voltage_range
                results['voltage_anomaly'] = (
                    (df['CellVoltage'] < voltage_range[0]) | 
                    (df['CellVoltage'] > voltage_range[1])
                )
            
            if 'CellTemperature' in df.columns:
                temp_range = self.config.data.temperature_range
                results['temperature_anomaly'] = (
                    (df['CellTemperature'] < temp_range[0]) | 
                    (df['CellTemperature'] > temp_range[1])
                )
            
            if 'CellSpecificGravity' in df.columns:
                gravity_range = self.config.data.specific_gravity_range
                results['specific_gravity_anomaly'] = (
                    (df['CellSpecificGravity'] < gravity_range[0]) | 
                    (df['CellSpecificGravity'] > gravity_range[1])
                )
            
            # Store performance metrics
            self.anomaly_scores.extend(anomaly_scores.tolist())
            self.detection_times.append(datetime.now())
            
            anomaly_count = np.sum(anomaly_predictions == -1)
            self.performance_logger.end_timer("detect_anomalies", True)
            self.logger.info(f"Anomaly detection completed: {anomaly_count} anomalies found")
            
            return results
            
        except Exception as e:
            self.performance_logger.end_timer("detect_anomalies", False)
            raise ModelInferenceError(f"Error detecting anomalies: {e}")
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        try:
            if self.isolation_forest is None:
                raise ModelError("No trained model to save")
            
            model_path = model_path or self.config.ml.anomaly_model_path
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model components
            model_data = {
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_columns': self.feature_columns,
                'training_data_shape': self.training_data.shape if self.training_data is not None else None,
                'config': {
                    'contamination': self.config.ml.contamination,
                    'n_estimators': self.config.ml.n_estimators,
                    'max_samples': self.config.ml.max_samples
                }
            }
            
            joblib.dump(model_data, model_path)
            
            self.logger.info(f"Anomaly detection model saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            raise ModelError(f"Error saving model: {e}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            model_path = model_path or self.config.ml.anomaly_model_path
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            # Load model components
            model_data = joblib.load(model_path)
            
            self.isolation_forest = model_data['isolation_forest']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.feature_columns = model_data['feature_columns']
            
            self.logger.info(f"Anomaly detection model loaded from {model_path}")
            return True
            
        except Exception as e:
            raise ModelError(f"Error loading model: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the anomaly detector.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.anomaly_scores:
            return {}
        
        return {
            'total_detections': len(self.anomaly_scores),
            'avg_anomaly_score': np.mean(self.anomaly_scores),
            'std_anomaly_score': np.std(self.anomaly_scores),
            'min_anomaly_score': np.min(self.anomaly_scores),
            'max_anomaly_score': np.max(self.anomaly_scores),
            'detection_times': self.detection_times
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.anomaly_scores = []
        self.detection_times = []
        self.logger.info("Performance metrics reset") 