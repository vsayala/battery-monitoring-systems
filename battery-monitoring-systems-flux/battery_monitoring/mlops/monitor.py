"""
MLOps Monitor module for battery monitoring system.

This module provides monitoring and alerting capabilities for
ML models, data drift detection, and system performance.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.database import get_database_manager
from ..data.loader import DataLoader
from ..ml.anomaly_detector import AnomalyDetector
from ..ml.cell_predictor import CellPredictor
from ..ml.forecaster import Forecaster


class MLOpsMonitor:
    """
    MLOps monitor for battery monitoring system.
    
    Provides continuous monitoring of ML models, data drift detection,
    performance tracking, and alerting capabilities.
    """
    
    def __init__(self, config=None):
        """Initialize the MLOps monitor."""
        self.config = config or get_config()
        self.logger = get_logger("mlops_monitor")
        self.performance_logger = get_performance_logger("mlops_monitor")
        
        # MLOps configuration
        self.drift_threshold = self.config.mlops.drift_threshold
        self.performance_threshold = self.config.mlops.performance_threshold
        self.alert_interval = self.config.mlops.alert_interval
        self.retention_days = self.config.mlops.retention_days
        self.metrics = self.config.mlops.metrics
        
        # Components
        self.db_manager = get_database_manager()
        self.data_loader = DataLoader()
        self.anomaly_detector = AnomalyDetector()
        self.cell_predictor = CellPredictor()
        self.forecaster = Forecaster()
        
        # Monitoring state
        self.monitoring_active = False
        self.last_alert_time = datetime.now()
        self.alert_history = []
        self.performance_history = []
        
        # Background tasks
        self.monitoring_task = None
        
        self.logger.info("MLOpsMonitor initialized")
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        try:
            if self.monitoring_active:
                self.logger.warning("Monitoring is already active")
                return
            
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("MLOps monitoring started")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("MLOps monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    def stop_monitoring_sync(self):
        """Synchronous stop method for compatibility."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            self.logger.info("MLOps monitoring stopped (sync)")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                try:
                    # Run monitoring checks
                    await self._check_data_drift()
                    await self._check_model_performance()
                    await self._check_system_health()
                    await self._cleanup_old_data()
                    
                    # Wait before next check
                    await asyncio.sleep(self.alert_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
    
    async def _check_data_drift(self):
        """Check for data drift in incoming data."""
        try:
            # Get recent data
            recent_df = self.data_loader.load_from_database(limit=1000)
            
            if len(recent_df) == 0:
                return
            
            # Get historical data for comparison
            historical_df = self.data_loader.load_from_database(limit=5000)
            
            if len(historical_df) == 0:
                return
            
            drift_detected = {}
            
            # Check voltage drift
            if 'CellVoltage' in recent_df.columns and 'CellVoltage' in historical_df.columns:
                recent_mean = recent_df['CellVoltage'].mean()
                historical_mean = historical_df['CellVoltage'].mean()
                voltage_drift = abs(recent_mean - historical_mean) / historical_mean
                
                if voltage_drift > self.drift_threshold:
                    drift_detected['voltage'] = {
                        'drift_value': voltage_drift,
                        'threshold': self.drift_threshold,
                        'recent_mean': recent_mean,
                        'historical_mean': historical_mean
                    }
            
            # Check temperature drift
            if 'CellTemperature' in recent_df.columns and 'CellTemperature' in historical_df.columns:
                recent_mean = recent_df['CellTemperature'].mean()
                historical_mean = historical_df['CellTemperature'].mean()
                temp_drift = abs(recent_mean - historical_mean) / historical_mean
                
                if temp_drift > self.drift_threshold:
                    drift_detected['temperature'] = {
                        'drift_value': temp_drift,
                        'threshold': self.drift_threshold,
                        'recent_mean': recent_mean,
                        'historical_mean': historical_mean
                    }
            
            # Check specific gravity drift
            if 'CellSpecificGravity' in recent_df.columns and 'CellSpecificGravity' in historical_df.columns:
                recent_mean = recent_df['CellSpecificGravity'].mean()
                historical_mean = historical_df['CellSpecificGravity'].mean()
                gravity_drift = abs(recent_mean - historical_mean) / historical_mean
                
                if gravity_drift > self.drift_threshold:
                    drift_detected['specific_gravity'] = {
                        'drift_value': gravity_drift,
                        'threshold': self.drift_threshold,
                        'recent_mean': recent_mean,
                        'historical_mean': historical_mean
                    }
            
            # Send alert if drift detected
            if drift_detected:
                await self._send_drift_alert(drift_detected)
                
        except Exception as e:
            self.logger.error(f"Error checking data drift: {e}")
    
    async def _check_model_performance(self):
        """Check ML model performance."""
        try:
            # Get recent data for performance evaluation
            recent_df = self.data_loader.load_from_database(limit=500)
            
            if len(recent_df) == 0:
                return
            
            performance_issues = {}
            
            # Check anomaly detection performance
            try:
                if self.anomaly_detector.isolation_forest is not None:
                    anomaly_results = self.anomaly_detector.detect_anomalies(recent_df)
                    anomaly_rate = anomaly_results['is_anomaly'].mean()
                    
                    if anomaly_rate > 0.5:  # High anomaly rate might indicate model issues
                        performance_issues['anomaly_detection'] = {
                            'issue': 'high_anomaly_rate',
                            'anomaly_rate': anomaly_rate,
                            'threshold': 0.5
                        }
            except Exception as e:
                performance_issues['anomaly_detection'] = {
                    'issue': 'model_error',
                    'error': str(e)
                }
            
            # Check cell prediction performance
            try:
                if self.cell_predictor.classifier is not None:
                    prediction_results = self.cell_predictor.predict(recent_df)
                    avg_confidence = prediction_results['prediction_confidence'].mean()
                    
                    if avg_confidence < self.performance_threshold:
                        performance_issues['cell_prediction'] = {
                            'issue': 'low_confidence',
                            'avg_confidence': avg_confidence,
                            'threshold': self.performance_threshold
                        }
            except Exception as e:
                performance_issues['cell_prediction'] = {
                    'issue': 'model_error',
                    'error': str(e)
                }
            
            # Send alert if performance issues detected
            if performance_issues:
                await self._send_performance_alert(performance_issues)
                
        except Exception as e:
            self.logger.error(f"Error checking model performance: {e}")
    
    async def _check_system_health(self):
        """Check overall system health."""
        try:
            # Get database stats
            db_stats = self.db_manager.get_database_stats()
            
            # Check for system issues
            system_issues = {}
            
            # Check database health
            if db_stats['total_records'] == 0:
                system_issues['database'] = {
                    'issue': 'no_data',
                    'message': 'No battery data available'
                }
            
            # Check model availability
            model_status = {
                'anomaly_detection': self.anomaly_detector.isolation_forest is not None,
                'cell_prediction': self.cell_predictor.classifier is not None,
                'forecasting': (self.forecaster.voltage_model is not None or 
                               self.forecaster.temperature_model is not None or 
                               self.forecaster.gravity_model is not None)
            }
            
            untrained_models = [model for model, trained in model_status.items() if not trained]
            if untrained_models:
                system_issues['models'] = {
                    'issue': 'untrained_models',
                    'models': untrained_models
                }
            
            # Send alert if system issues detected
            if system_issues:
                await self._send_system_alert(system_issues)
                
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        try:
            # Clean up old database records
            deleted_count = self.db_manager.cleanup_old_data(self.retention_days)
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old records")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def _send_drift_alert(self, drift_data: Dict[str, Any]):
        """Send data drift alert."""
        try:
            alert = {
                'type': 'data_drift',
                'severity': 'warning',
                'message': 'Data drift detected in battery monitoring data',
                'drift_data': drift_data,
                'timestamp': datetime.now().isoformat(),
                'threshold': self.drift_threshold
            }
            
            await self._send_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error sending drift alert: {e}")
    
    async def _send_performance_alert(self, performance_data: Dict[str, Any]):
        """Send model performance alert."""
        try:
            alert = {
                'type': 'model_performance',
                'severity': 'warning',
                'message': 'ML model performance issues detected',
                'performance_data': performance_data,
                'timestamp': datetime.now().isoformat(),
                'threshold': self.performance_threshold
            }
            
            await self._send_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error sending performance alert: {e}")
    
    async def _send_system_alert(self, system_data: Dict[str, Any]):
        """Send system health alert."""
        try:
            alert = {
                'type': 'system_health',
                'severity': 'error',
                'message': 'System health issues detected',
                'system_data': system_data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._send_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error sending system alert: {e}")
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels."""
        try:
            # Store alert in history
            self.alert_history.append(alert)
            
            # Check if enough time has passed since last alert
            if datetime.now() - self.last_alert_time < timedelta(minutes=5):
                return  # Rate limit alerts
            
            self.last_alert_time = datetime.now()
            
            # Log alert
            self.logger.warning(f"Alert: {alert['message']}")
            
            # Save alert to database
            self.db_manager.save_system_metric(
                f"alert_{alert['type']}",
                1.0,
                "count"
            )
            
            # TODO: Send alerts through configured channels (email, Slack, etc.)
            # For now, just log the alert
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'monitoring_active': self.monitoring_active,
            'last_alert_time': self.last_alert_time.isoformat(),
            'total_alerts': len(self.alert_history),
            'drift_threshold': self.drift_threshold,
            'performance_threshold': self.performance_threshold,
            'alert_interval': self.alert_interval,
            'retention_days': self.retention_days
        }
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get MLOps performance metrics."""
        return {
            'monitoring_active': self.monitoring_active,
            'total_alerts': len(self.alert_history),
            'last_alert_time': self.last_alert_time.isoformat(),
            'drift_threshold': self.drift_threshold,
            'performance_threshold': self.performance_threshold,
            'metrics_tracked': self.metrics
        }
    
    def clear_alert_history(self) -> None:
        """Clear alert history."""
        self.alert_history = []
        self.logger.info("Alert history cleared") 