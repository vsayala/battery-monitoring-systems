"""
Model Evaluator module for battery monitoring system.

This module provides model evaluation and performance assessment capabilities
for all ML models in the battery monitoring system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from .anomaly_detector import AnomalyDetector
from .cell_predictor import CellPredictor
from .forecaster import Forecaster

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.exceptions import ModelError


class ModelEvaluator:
    """
    Model evaluator for battery monitoring system.
    
    Provides comprehensive evaluation capabilities for all ML models
    including performance metrics, visualizations, and model comparison.
    """
    
    def __init__(self, config=None):
        """Initialize the model evaluator."""
        self.config = config or get_config()
        self.logger = get_logger("model_evaluator")
        self.performance_logger = get_performance_logger("model_evaluator")
        
        # Initialize models
        self.anomaly_detector = AnomalyDetector(config)
        self.cell_predictor = CellPredictor(config)
        self.forecaster = Forecaster(config)
        
        # Evaluation results
        self.evaluation_history = []
        
        self.logger.info("ModelEvaluator initialized")
    
    def evaluate_all_models(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate all ML models with test data.
        
        Args:
            test_df: Test DataFrame with battery data
            
        Returns:
            Dictionary with evaluation results for all models
        """
        self.performance_logger.start_timer("evaluate_all_models")
        
        try:
            self.logger.info("Starting evaluation for all ML models")
            
            results = {}
            
            # Load models if not already loaded
            self._load_models_if_needed()
            
            # Evaluate anomaly detection model
            try:
                self.logger.info("Evaluating anomaly detection model")
                anomaly_results = self._evaluate_anomaly_detection(test_df)
                results['anomaly_detection'] = anomaly_results
            except Exception as e:
                self.logger.error(f"Error evaluating anomaly detection model: {e}")
                results['anomaly_detection'] = {'error': str(e)}
            
            # Evaluate cell prediction model
            try:
                self.logger.info("Evaluating cell prediction model")
                cell_results = self._evaluate_cell_prediction(test_df)
                results['cell_prediction'] = cell_results
            except Exception as e:
                self.logger.error(f"Error evaluating cell prediction model: {e}")
                results['cell_prediction'] = {'error': str(e)}
            
            # Evaluate forecasting models
            try:
                self.logger.info("Evaluating forecasting models")
                forecast_results = self._evaluate_forecasting(test_df)
                results['forecasting'] = forecast_results
            except Exception as e:
                self.logger.error(f"Error evaluating forecasting models: {e}")
                results['forecasting'] = {'error': str(e)}
            
            # Record evaluation session
            evaluation_session = {
                'timestamp': datetime.now(),
                'test_data_shape': test_df.shape,
                'results': results
            }
            self.evaluation_history.append(evaluation_session)
            
            self.performance_logger.end_timer("evaluate_all_models", True)
            self.logger.info("All model evaluation completed")
            
            return results
            
        except Exception as e:
            self.performance_logger.end_timer("evaluate_all_models", False)
            raise ModelError(f"Error in model evaluation: {e}")
    
    def _load_models_if_needed(self) -> None:
        """Load models if they haven't been loaded yet."""
        try:
            # Try to load models
            self.anomaly_detector.load_model()
            self.cell_predictor.load_model()
            self.forecaster.load_model()
        except Exception as e:
            self.logger.warning(f"Some models could not be loaded: {e}")
    
    def _evaluate_anomaly_detection(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate anomaly detection model."""
        try:
            if self.anomaly_detector.isolation_forest is None:
                return {'error': 'Model not trained'}
            
            # Detect anomalies
            results = self.anomaly_detector.detect_anomalies(test_df)
            
            # Calculate metrics
            anomaly_count = results['is_anomaly'].sum()
            total_count = len(results)
            anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
            
            # Calculate average anomaly score
            avg_score = results['anomaly_score'].mean()
            std_score = results['anomaly_score'].std()
            
            # Specific anomaly rates
            voltage_anomaly_rate = results.get('voltage_anomaly', pd.Series([False] * len(results))).mean()
            temperature_anomaly_rate = results.get('temperature_anomaly', pd.Series([False] * len(results))).mean()
            gravity_anomaly_rate = results.get('specific_gravity_anomaly', pd.Series([False] * len(results))).mean()
            
            return {
                'total_samples': total_count,
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_rate),
                'avg_anomaly_score': float(avg_score),
                'std_anomaly_score': float(std_score),
                'voltage_anomaly_rate': float(voltage_anomaly_rate),
                'temperature_anomaly_rate': float(temperature_anomaly_rate),
                'specific_gravity_anomaly_rate': float(gravity_anomaly_rate),
                'evaluation_date': datetime.now()
            }
            
        except Exception as e:
            raise ModelError(f"Error evaluating anomaly detection: {e}")
    
    def _evaluate_cell_prediction(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate cell prediction model."""
        try:
            if self.cell_predictor.classifier is None:
                return {'error': 'Model not trained'}
            
            # Make predictions
            results = self.cell_predictor.predict(test_df)
            
            # Create ground truth labels (simplified approach)
            if 'CellVoltage' in test_df.columns:
                # Create labels based on voltage thresholds
                dead_threshold = self.config.ml.dead_cell_threshold
                alive_threshold = self.config.ml.alive_cell_threshold
                
                ground_truth = []
                for voltage in test_df['CellVoltage']:
                    if voltage < dead_threshold:
                        ground_truth.append('dead')
                    elif voltage > alive_threshold:
                        ground_truth.append('alive')
                    else:
                        # For intermediate values, use the closer threshold
                        if abs(voltage - dead_threshold) < abs(voltage - alive_threshold):
                            ground_truth.append('dead')
                        else:
                            ground_truth.append('alive')
                
                # Calculate metrics
                y_true = ground_truth
                y_pred = results['predicted_health'].tolist()
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Class distribution
                alive_predictions = results['is_alive'].sum()
                dead_predictions = results['is_dead'].sum()
                
                return {
                    'total_samples': len(results),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'alive_predictions': int(alive_predictions),
                    'dead_predictions': int(dead_predictions),
                    'avg_confidence': float(results['prediction_confidence'].mean()),
                    'confusion_matrix': cm.tolist(),
                    'evaluation_date': datetime.now()
                }
            else:
                return {'error': 'CellVoltage column required for evaluation'}
            
        except Exception as e:
            raise ModelError(f"Error evaluating cell prediction: {e}")
    
    def _evaluate_forecasting(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate forecasting models."""
        try:
            results = {}
            
            # Evaluate voltage forecasting
            if self.forecaster.voltage_model is not None and 'CellVoltage' in test_df.columns:
                voltage_metrics = self._evaluate_forecast_parameter(test_df, 'CellVoltage', self.forecaster.voltage_model)
                results['voltage'] = voltage_metrics
            
            # Evaluate temperature forecasting
            if self.forecaster.temperature_model is not None and 'CellTemperature' in test_df.columns:
                temp_metrics = self._evaluate_forecast_parameter(test_df, 'CellTemperature', self.forecaster.temperature_model)
                results['temperature'] = temp_metrics
            
            # Evaluate specific gravity forecasting
            if self.forecaster.gravity_model is not None and 'CellSpecificGravity' in test_df.columns:
                gravity_metrics = self._evaluate_forecast_parameter(test_df, 'CellSpecificGravity', self.forecaster.gravity_model)
                results['specific_gravity'] = gravity_metrics
            
            return {
                'models_evaluated': len(results),
                'model_results': results,
                'evaluation_date': datetime.now()
            }
            
        except Exception as e:
            raise ModelError(f"Error evaluating forecasting: {e}")
    
    def _evaluate_forecast_parameter(self, test_df: pd.DataFrame, target_column: str, model) -> Dict[str, Any]:
        """Evaluate forecasting for a specific parameter."""
        try:
            # Prepare features
            features, target = self.forecaster.prepare_time_series_features(test_df, target_column)
            
            # Remove rows with NaN targets
            valid_indices = ~target.isna()
            X = features[valid_indices]
            y = target[valid_indices]
            
            if len(X) == 0:
                return {'error': 'No valid data for evaluation'}
            
            # Scale features
            X_scaled = self.forecaster.scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100 if np.any(y != 0) else 0
            
            return {
                'n_samples': len(y),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'mape': float(mape),
                'target_mean': float(y.mean()),
                'target_std': float(y.std())
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results from evaluate_all_models
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        try:
            if output_path is None:
                output_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(output_path, 'w') as f:
                f.write("Battery Monitoring System - Model Evaluation Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                
                # Anomaly Detection Results
                f.write("ANOMALY DETECTION EVALUATION\n")
                f.write("-" * 30 + "\n")
                if 'anomaly_detection' in results:
                    anomaly = results['anomaly_detection']
                    if 'error' not in anomaly:
                        f.write(f"Total Samples: {anomaly['total_samples']}\n")
                        f.write(f"Anomalies Detected: {anomaly['anomaly_count']}\n")
                        f.write(f"Anomaly Rate: {anomaly['anomaly_rate']:.3f}\n")
                        f.write(f"Average Anomaly Score: {anomaly['avg_anomaly_score']:.3f}\n")
                        f.write(f"Voltage Anomaly Rate: {anomaly['voltage_anomaly_rate']:.3f}\n")
                        f.write(f"Temperature Anomaly Rate: {anomaly['temperature_anomaly_rate']:.3f}\n")
                        f.write(f"Specific Gravity Anomaly Rate: {anomaly['specific_gravity_anomaly_rate']:.3f}\n")
                    else:
                        f.write(f"Error: {anomaly['error']}\n")
                f.write("\n")
                
                # Cell Prediction Results
                f.write("CELL PREDICTION EVALUATION\n")
                f.write("-" * 30 + "\n")
                if 'cell_prediction' in results:
                    cell = results['cell_prediction']
                    if 'error' not in cell:
                        f.write(f"Total Samples: {cell['total_samples']}\n")
                        f.write(f"Accuracy: {cell['accuracy']:.3f}\n")
                        f.write(f"Precision: {cell['precision']:.3f}\n")
                        f.write(f"Recall: {cell['recall']:.3f}\n")
                        f.write(f"F1 Score: {cell['f1_score']:.3f}\n")
                        f.write(f"Alive Predictions: {cell['alive_predictions']}\n")
                        f.write(f"Dead Predictions: {cell['dead_predictions']}\n")
                        f.write(f"Average Confidence: {cell['avg_confidence']:.3f}\n")
                    else:
                        f.write(f"Error: {cell['error']}\n")
                f.write("\n")
                
                # Forecasting Results
                f.write("FORECASTING EVALUATION\n")
                f.write("-" * 30 + "\n")
                if 'forecasting' in results:
                    forecast = results['forecasting']
                    if 'error' not in forecast:
                        f.write(f"Models Evaluated: {forecast['models_evaluated']}\n")
                        for param, metrics in forecast['model_results'].items():
                            f.write(f"\n{param.upper()}:\n")
                            if 'error' not in metrics:
                                f.write(f"  MAE: {metrics['mae']:.3f}\n")
                                f.write(f"  RMSE: {metrics['rmse']:.3f}\n")
                                f.write(f"  R² Score: {metrics['r2_score']:.3f}\n")
                                f.write(f"  MAPE: {metrics['mape']:.3f}%\n")
                            else:
                                f.write(f"  Error: {metrics['error']}\n")
                    else:
                        f.write(f"Error: {forecast['error']}\n")
                f.write("\n")
                
                # Summary
                f.write("SUMMARY\n")
                f.write("-" * 30 + "\n")
                successful_models = sum(1 for model_results in results.values() if 'error' not in model_results)
                total_models = len(results)
                f.write(f"Successfully Evaluated Models: {successful_models}/{total_models}\n")
                f.write(f"Evaluation Date: {datetime.now()}\n")
            
            self.logger.info(f"Evaluation report generated: {output_path}")
            return output_path
            
        except Exception as e:
            raise ModelError(f"Error generating evaluation report: {e}")
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.evaluation_history
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of model performance metrics."""
        if not self.evaluation_history:
            return {}
        
        latest_evaluation = self.evaluation_history[-1]['results']
        
        summary = {
            'last_evaluation_date': self.evaluation_history[-1]['timestamp'],
            'models_evaluated': len(latest_evaluation),
            'successful_evaluations': sum(1 for model_results in latest_evaluation.values() if 'error' not in model_results)
        }
        
        # Add key metrics
        if 'anomaly_detection' in latest_evaluation and 'error' not in latest_evaluation['anomaly_detection']:
            summary['anomaly_rate'] = latest_evaluation['anomaly_detection']['anomaly_rate']
        
        if 'cell_prediction' in latest_evaluation and 'error' not in latest_evaluation['cell_prediction']:
            summary['cell_prediction_accuracy'] = latest_evaluation['cell_prediction']['accuracy']
        
        if 'forecasting' in latest_evaluation and 'error' not in latest_evaluation['forecasting']:
            summary['forecasting_models'] = latest_evaluation['forecasting']['models_evaluated']
        
        return summary 