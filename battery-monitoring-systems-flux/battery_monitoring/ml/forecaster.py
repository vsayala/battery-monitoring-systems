"""
Time Series Forecasting module for battery monitoring system.

This module provides forecasting capabilities for battery cell parameters
including voltage, temperature, and specific gravity predictions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.exceptions import ModelError, ModelTrainingError, ModelInferenceError


class Forecaster:
    """
    Time series forecasting for battery monitoring.
    
    Forecasts future values of voltage, temperature, and specific gravity
    using linear regression and time series analysis.
    """
    
    def __init__(self, config=None):
        """Initialize the forecaster."""
        self.config = config or get_config()
        self.logger = get_logger("forecasting")
        self.performance_logger = get_performance_logger("forecasting")
        
        # Models
        self.voltage_model = None
        self.temperature_model = None
        self.gravity_model = None
        self.scaler = StandardScaler()
        
        # Training data
        self.training_data = None
        self.feature_columns = None
        
        # Performance metrics
        self.forecasts = []
        self.forecast_times = []
        
        self.logger.info("Forecaster initialized")
    
    def prepare_time_series_features(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare time series features for forecasting.
        
        Args:
            df: Input DataFrame with battery data
            target_column: Column to forecast
            
        Returns:
            Tuple of features DataFrame and target series
        """
        try:
            if target_column not in df.columns:
                raise ModelError(f"Target column {target_column} not found in data")
            
            # Sort by time if available
            if 'PacketDateTime' in df.columns:
                df = df.sort_values('PacketDateTime')
            
            # Create lag features
            features = pd.DataFrame()
            target = df[target_column].copy()
            
            # Add lag features
            for lag in range(1, self.config.ml.lookback_window + 1):
                features[f'{target_column}_lag_{lag}'] = target.shift(lag)
            
            # Add rolling statistics
            features[f'{target_column}_rolling_mean'] = target.rolling(window=5, min_periods=1).mean()
            features[f'{target_column}_rolling_std'] = target.rolling(window=5, min_periods=1).std()
            features[f'{target_column}_rolling_min'] = target.rolling(window=5, min_periods=1).min()
            features[f'{target_column}_rolling_max'] = target.rolling(window=5, min_periods=1).max()
            
            # Add time-based features
            if 'PacketDateTime' in df.columns:
                try:
                    dt = pd.to_datetime(df['PacketDateTime'])
                    features['hour'] = dt.dt.hour
                    features['day_of_week'] = dt.dt.dayofweek
                    features['is_weekend'] = dt.dt.dayofweek >= 5
                except:
                    pass
            
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
            target = target.fillna(method='bfill').fillna(method='ffill')
            
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            return features, target
            
        except Exception as e:
            raise ModelError(f"Error preparing time series features: {e}")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the forecasting models.
        
        Args:
            df: Training DataFrame with battery data
            
        Returns:
            Dictionary with training results
        """
        self.performance_logger.start_timer("train_forecasting_models")
        
        try:
            self.logger.info("Starting forecasting model training")
            
            results = {}
            
            # Train voltage forecasting model
            if 'CellVoltage' in df.columns:
                self.logger.info("Training voltage forecasting model")
                voltage_features, voltage_target = self.prepare_time_series_features(df, 'CellVoltage')
                
                # Remove rows with NaN targets
                valid_indices = ~voltage_target.isna()
                X = voltage_features[valid_indices]
                y = voltage_target[valid_indices]
                
                if len(X) > self.config.ml.min_samples:
                    # Scale features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Train model
                    self.voltage_model = LinearRegression()
                    self.voltage_model.fit(X_scaled, y)
                    
                    # Evaluate
                    y_pred = self.voltage_model.predict(X_scaled)
                    mae = mean_absolute_error(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    
                    results['voltage'] = {
                        'model_type': 'linear_regression',
                        'n_samples': len(X),
                        'n_features': X.shape[1],
                        'mae': mae,
                        'rmse': rmse,
                        'training_date': datetime.now()
                    }
            
            # Train temperature forecasting model
            if 'CellTemperature' in df.columns:
                self.logger.info("Training temperature forecasting model")
                temp_features, temp_target = self.prepare_time_series_features(df, 'CellTemperature')
                
                valid_indices = ~temp_target.isna()
                X = temp_features[valid_indices]
                y = temp_target[valid_indices]
                
                if len(X) > self.config.ml.min_samples:
                    X_scaled = self.scaler.fit_transform(X)
                    self.temperature_model = LinearRegression()
                    self.temperature_model.fit(X_scaled, y)
                    
                    y_pred = self.temperature_model.predict(X_scaled)
                    mae = mean_absolute_error(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    
                    results['temperature'] = {
                        'model_type': 'linear_regression',
                        'n_samples': len(X),
                        'n_features': X.shape[1],
                        'mae': mae,
                        'rmse': rmse,
                        'training_date': datetime.now()
                    }
            
            # Train specific gravity forecasting model
            if 'CellSpecificGravity' in df.columns:
                self.logger.info("Training specific gravity forecasting model")
                gravity_features, gravity_target = self.prepare_time_series_features(df, 'CellSpecificGravity')
                
                valid_indices = ~gravity_target.isna()
                X = gravity_features[valid_indices]
                y = gravity_target[valid_indices]
                
                if len(X) > self.config.ml.min_samples:
                    X_scaled = self.scaler.fit_transform(X)
                    self.gravity_model = LinearRegression()
                    self.gravity_model.fit(X_scaled, y)
                    
                    y_pred = self.gravity_model.predict(X_scaled)
                    mae = mean_absolute_error(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    
                    results['specific_gravity'] = {
                        'model_type': 'linear_regression',
                        'n_samples': len(X),
                        'n_features': X.shape[1],
                        'mae': mae,
                        'rmse': rmse,
                        'training_date': datetime.now()
                    }
            
            self.performance_logger.end_timer("train_forecasting_models", True)
            self.logger.info(f"Forecasting models trained successfully: {len(results)} models")
            
            return results
            
        except Exception as e:
            self.performance_logger.end_timer("train_forecasting_models", False)
            raise ModelTrainingError(f"Error training forecasting models: {e}")
    
    def forecast(self, df: pd.DataFrame, forecast_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate forecasts for battery parameters.
        
        Args:
            df: DataFrame with recent battery data
            forecast_steps: Number of steps to forecast
            
        Returns:
            Dictionary with forecast results
        """
        self.performance_logger.start_timer("generate_forecasts")
        
        try:
            forecast_steps = forecast_steps or self.config.ml.forecast_steps
            
            self.logger.info(f"Generating forecasts for {forecast_steps} steps")
            
            forecasts = {}
            
            # Forecast voltage
            if self.voltage_model is not None and 'CellVoltage' in df.columns:
                voltage_forecast = self._forecast_parameter(df, 'CellVoltage', self.voltage_model, forecast_steps)
                forecasts['voltage'] = voltage_forecast
            
            # Forecast temperature
            if self.temperature_model is not None and 'CellTemperature' in df.columns:
                temp_forecast = self._forecast_parameter(df, 'CellTemperature', self.temperature_model, forecast_steps)
                forecasts['temperature'] = temp_forecast
            
            # Forecast specific gravity
            if self.gravity_model is not None and 'CellSpecificGravity' in df.columns:
                gravity_forecast = self._forecast_parameter(df, 'CellSpecificGravity', self.gravity_model, forecast_steps)
                forecasts['specific_gravity'] = gravity_forecast
            
            # Store performance metrics
            self.forecasts.append(forecasts)
            self.forecast_times.append(datetime.now())
            
            self.performance_logger.end_timer("generate_forecasts", True)
            self.logger.info(f"Forecasts generated successfully for {len(forecasts)} parameters")
            
            return forecasts
            
        except Exception as e:
            self.performance_logger.end_timer("generate_forecasts", False)
            raise ModelInferenceError(f"Error generating forecasts: {e}")
    
    def _forecast_parameter(self, df: pd.DataFrame, target_column: str, model, forecast_steps: int) -> Dict[str, Any]:
        """Helper method to forecast a specific parameter."""
        try:
            # Prepare features for the last available data
            features, target = self.prepare_time_series_features(df, target_column)
            
            if len(features) == 0:
                return {}
            
            # Get the most recent feature vector
            latest_features = features.iloc[-1:].values
            
            # Generate forecasts
            forecast_values = []
            current_features = latest_features.copy()
            
            for step in range(forecast_steps):
                # Scale features
                scaled_features = self.scaler.transform(current_features)
                
                # Make prediction
                prediction = model.predict(scaled_features)[0]
                forecast_values.append(prediction)
                
                # Update features for next step (simple approach)
                # In a more sophisticated implementation, you'd update the lag features
                current_features[0, 0] = prediction  # Update the first lag
            
            # Generate forecast dates
            if 'PacketDateTime' in df.columns:
                last_date = pd.to_datetime(df['PacketDateTime'].iloc[-1])
                forecast_dates = [last_date + timedelta(hours=i+1) for i in range(forecast_steps)]
            else:
                forecast_dates = [datetime.now() + timedelta(hours=i+1) for i in range(forecast_steps)]
            
            return {
                'forecast_values': forecast_values,
                'forecast_dates': [d.isoformat() for d in forecast_dates],
                'forecast_steps': forecast_steps,
                'last_actual_value': target.iloc[-1] if len(target) > 0 else None,
                'forecast_type': target_column
            }
            
        except Exception as e:
            self.logger.error(f"Error forecasting {target_column}: {e}")
            return {}
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """Save the trained models to disk."""
        try:
            model_path = model_path or self.config.ml.forecasting_model_path
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'voltage_model': self.voltage_model,
                'temperature_model': self.temperature_model,
                'gravity_model': self.gravity_model,
                'scaler': self.scaler,
                'config': {
                    'forecast_steps': self.config.ml.forecast_steps,
                    'lookback_window': self.config.ml.lookback_window,
                    'min_samples': self.config.ml.min_samples
                }
            }
            
            joblib.dump(model_data, model_path)
            self.logger.info(f"Forecasting models saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            raise ModelError(f"Error saving models: {e}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained models from disk."""
        try:
            model_path = model_path or self.config.ml.forecasting_model_path
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            model_data = joblib.load(model_path)
            
            self.voltage_model = model_data['voltage_model']
            self.temperature_model = model_data['temperature_model']
            self.gravity_model = model_data['gravity_model']
            self.scaler = model_data['scaler']
            
            self.logger.info(f"Forecasting models loaded from {model_path}")
            return True
            
        except Exception as e:
            raise ModelError(f"Error loading models: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the forecaster."""
        if not self.forecasts:
            return {}
        
        return {
            'total_forecasts': len(self.forecasts),
            'forecast_times': self.forecast_times,
            'last_forecast': self.forecasts[-1] if self.forecasts else None
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.forecasts = []
        self.forecast_times = []
        self.logger.info("Performance metrics reset") 