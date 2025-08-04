"""
Cell Health Prediction module for battery monitoring system.

This module provides cell health prediction capabilities to determine
if battery cells are likely to be dead or alive based on their patterns.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.exceptions import ModelError, ModelTrainingError, ModelInferenceError


class CellPredictor:
    """
    Cell health prediction for battery monitoring.
    
    Predicts whether battery cells are likely to be dead or alive
    based on voltage, temperature, and specific gravity patterns.
    """
    
    def __init__(self, config=None):
        """Initialize the cell predictor."""
        self.config = config or get_config()
        self.logger = get_logger("cell_prediction")
        self.performance_logger = get_performance_logger("cell_prediction")
        
        # Models
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training data
        self.training_data = None
        self.feature_columns = None
        
        # Performance metrics
        self.predictions = []
        self.confidences = []
        self.prediction_times = []
        
        self.logger.info("CellPredictor initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for cell health prediction.
        
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
            
            if len(available_columns) < 3:
                raise ModelError(f"Insufficient columns for cell prediction. Need at least 3, got {len(available_columns)}")
            
            # Create features - only use numeric columns
            numeric_columns = df[available_columns].select_dtypes(include=[np.number]).columns
            features = df[numeric_columns].copy()
            
            # Add derived features for numeric columns only
            if 'CellVoltage' in features.columns:
                features['voltage_mean'] = features['CellVoltage'].mean()
                features['voltage_std'] = features['CellVoltage'].std()
                features['voltage_min'] = features['CellVoltage'].min()
                features['voltage_max'] = features['CellVoltage'].max()
                features['voltage_range'] = features['CellVoltage'].max() - features['CellVoltage'].min()
                features['voltage_trend'] = features['CellVoltage'].diff().mean()
            
            if 'CellTemperature' in features.columns:
                features['temp_mean'] = features['CellTemperature'].mean()
                features['temp_std'] = features['CellTemperature'].std()
                features['temp_min'] = features['CellTemperature'].min()
                features['temp_max'] = features['CellTemperature'].max()
                features['temp_range'] = features['CellTemperature'].max() - features['CellTemperature'].min()
                features['temp_trend'] = features['CellTemperature'].diff().mean()
            
            if 'CellSpecificGravity' in features.columns:
                features['gravity_mean'] = features['CellSpecificGravity'].mean()
                features['gravity_std'] = features['CellSpecificGravity'].std()
                features['gravity_min'] = features['CellSpecificGravity'].min()
                features['gravity_max'] = features['CellSpecificGravity'].max()
                features['gravity_range'] = features['CellSpecificGravity'].max() - features['CellSpecificGravity'].min()
                features['gravity_trend'] = features['CellSpecificGravity'].diff().mean()
            
            # Add time-based features if available
            if 'PacketDateTime' in df.columns:
                try:
                    dt = pd.to_datetime(df['PacketDateTime'])
                    features['hour'] = dt.dt.hour
                    features['day_of_week'] = dt.dt.dayofweek
                    features['is_weekend'] = dt.dt.dayofweek >= 5
                except:
                    pass
            
            # Fill NaN values
            features = features.ffill().bfill().fillna(0)
            
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            self.feature_columns = features.columns.tolist()
            
            self.performance_logger.end_timer("prepare_features", True)
            self.logger.info(f"Prepared {len(features)} features for cell prediction")
            
            return features
            
        except Exception as e:
            self.performance_logger.end_timer("prepare_features", False)
            raise ModelError(f"Error preparing features: {e}")
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create labels for training based on voltage thresholds.
        
        Args:
            df: DataFrame with battery data
            
        Returns:
            Array of labels ('alive' or 'dead')
        """
        try:
            if 'CellVoltage' not in df.columns:
                raise ModelError("CellVoltage column required for label creation")
            
            # Create labels based on voltage thresholds
            dead_threshold = self.config.ml.dead_cell_threshold
            alive_threshold = self.config.ml.alive_cell_threshold
            
            # Group by device and cell to get average voltage per cell
            if 'DeviceID' in df.columns and 'CellNumber' in df.columns:
                cell_voltages = df.groupby(['DeviceID', 'CellNumber'])['CellVoltage'].mean()
            else:
                cell_voltages = df.groupby('CellNumber')['CellVoltage'].mean()
            
            # Create labels
            labels = []
            for voltage in cell_voltages:
                if voltage < dead_threshold:
                    labels.append('dead')
                elif voltage > alive_threshold:
                    labels.append('alive')
                else:
                    # For intermediate values, use the closer threshold
                    if abs(voltage - dead_threshold) < abs(voltage - alive_threshold):
                        labels.append('dead')
                    else:
                        labels.append('alive')
            
            return np.array(labels)
            
        except Exception as e:
            raise ModelError(f"Error creating labels: {e}")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the cell health prediction model.
        
        Args:
            df: Training DataFrame with battery data
            
        Returns:
            Dictionary with training results
        """
        self.performance_logger.start_timer("train_cell_model")
        
        try:
            self.logger.info("Starting cell health prediction model training")
            
            # Create labels first
            labels = self.create_labels(df)
            
            # Prepare features for each cell (group by device and cell)
            if 'DeviceID' in df.columns and 'CellNumber' in df.columns:
                # Group by device and cell, then prepare features
                cell_features = []
                cell_labels = []
                
                for (device_id, cell_num), group in df.groupby(['DeviceID', 'CellNumber']):
                    if len(group) > 0:
                        # Prepare features for this cell group
                        cell_feat = self.prepare_features(group)
                        if len(cell_feat) > 0:
                            # Use the first row of features for this cell
                            cell_features.append(cell_feat.iloc[0])
                            cell_labels.append(labels[len(cell_features) - 1])
                
                # Convert to DataFrame
                features_df = pd.DataFrame(cell_features)
            else:
                # Fallback: use individual rows
                features = self.prepare_features(df)
                features_df = features
                cell_labels = labels[:len(features)]
            
            self.training_data = features_df
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(cell_labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, encoded_labels,
                test_size=self.config.ml.test_size,
                random_state=self.config.ml.random_state,
                stratify=encoded_labels
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train classifier
            self.classifier = RandomForestClassifier(
                n_estimators=self.config.ml.n_estimators,
                random_state=self.config.ml.random_state,
                n_jobs=-1
            )
            
            # Train the model
            self.classifier.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.classifier.predict(X_test_scaled)
            y_pred_proba = self.classifier.predict_proba(X_test_scaled)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Training results
            results = {
                'model_type': 'random_forest_classifier',
                'n_samples': len(features_df),
                'n_features': features_df.shape[1],
                'n_estimators': self.config.ml.n_estimators,
                'training_date': datetime.now(),
                'feature_columns': list(features_df.columns),
                'label_classes': self.label_encoder.classes_.tolist(),
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'confusion_matrix': conf_matrix.tolist(),
                'class_distribution': {
                    'alive': np.sum(labels == 'alive'),
                    'dead': np.sum(labels == 'dead')
                }
            }
            
            self.performance_logger.end_timer("train_cell_model", True)
            self.logger.info(f"Cell prediction model trained successfully: accuracy={results['accuracy']:.3f}")
            
            return results
            
        except Exception as e:
            self.performance_logger.end_timer("train_cell_model", False)
            raise ModelTrainingError(f"Error training cell prediction model: {e}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cell health for battery data.
        
        Args:
            df: DataFrame with battery data to analyze
            
        Returns:
            DataFrame with prediction results
        """
        self.performance_logger.start_timer("predict_cell_health")
        
        try:
            if self.classifier is None:
                raise ModelError("Model not trained. Call train() first.")
            
            self.logger.info(f"Predicting cell health for {len(df)} samples")
            
            # Prepare features
            features = self.prepare_features(df)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.classifier.predict(scaled_features)
            prediction_proba = self.classifier.predict_proba(scaled_features)
            
            # Decode predictions
            decoded_predictions = self.label_encoder.inverse_transform(predictions)
            
            # Get confidence scores
            confidences = np.max(prediction_proba, axis=1)
            
            # Create results DataFrame
            results = df.copy()
            results['predicted_health'] = decoded_predictions
            results['prediction_confidence'] = confidences
            
            # Handle probability assignment based on number of classes
            if len(self.label_encoder.classes_) == 2:
                # Binary classification
                results['alive_probability'] = prediction_proba[:, 1]
                results['dead_probability'] = prediction_proba[:, 0]
            else:
                # Single class or multi-class
                results['alive_probability'] = prediction_proba[:, 0]
                results['dead_probability'] = 1.0 - prediction_proba[:, 0]
            
            # Add prediction flags
            results['is_alive'] = decoded_predictions == 'alive'
            results['is_dead'] = decoded_predictions == 'dead'
            
            # Store performance metrics
            self.predictions.extend(decoded_predictions.tolist())
            self.confidences.extend(confidences.tolist())
            self.prediction_times.append(datetime.now())
            
            alive_count = np.sum(decoded_predictions == 'alive')
            dead_count = np.sum(decoded_predictions == 'dead')
            
            self.performance_logger.end_timer("predict_cell_health", True)
            self.logger.info(f"Cell health prediction completed: {alive_count} alive, {dead_count} dead")
            
            return results
            
        except Exception as e:
            self.performance_logger.end_timer("predict_cell_health", False)
            raise ModelInferenceError(f"Error predicting cell health: {e}")
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        try:
            if self.classifier is None:
                raise ModelError("No trained model to save")
            
            model_path = model_path or self.config.ml.prediction_model_path
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model components
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'training_data_shape': self.training_data.shape if self.training_data is not None else None,
                'config': {
                    'n_estimators': self.config.ml.n_estimators,
                    'dead_cell_threshold': self.config.ml.dead_cell_threshold,
                    'alive_cell_threshold': self.config.ml.alive_cell_threshold
                }
            }
            
            joblib.dump(model_data, model_path)
            
            self.logger.info(f"Cell prediction model saved to {model_path}")
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
            model_path = model_path or self.config.ml.prediction_model_path
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            # Load model components
            model_data = joblib.load(model_path)
            
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            
            self.logger.info(f"Cell prediction model loaded from {model_path}")
            return True
            
        except Exception as e:
            raise ModelError(f"Error loading model: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the cell predictor.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.predictions:
            return {}
        
        return {
            'total_predictions': len(self.predictions),
            'alive_predictions': self.predictions.count('alive'),
            'dead_predictions': self.predictions.count('dead'),
            'avg_confidence': np.mean(self.confidences),
            'std_confidence': np.std(self.confidences),
            'min_confidence': np.min(self.confidences),
            'max_confidence': np.max(self.confidences),
            'prediction_times': self.prediction_times
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.predictions = []
        self.confidences = []
        self.prediction_times = []
        self.logger.info("Performance metrics reset") 