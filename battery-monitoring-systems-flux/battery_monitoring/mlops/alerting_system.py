"""
Alerting and Response System for MLOps

This module implements comprehensive monitoring, alerting, and automated response
policies for ML model performance, data drift, and system health.
"""

import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from threading import Thread, Lock
import queue

from ..core.logger import setup_logging
from ..core.database import DatabaseManager
from ..core.exceptions import MonitoringError, AlertingError

# Setup logging
logger = setup_logging()
alert_logger = logging.getLogger('alerting_system')


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    DATA_DRIFT = "data_drift"
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_HEALTH = "system_health"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    DISTRIBUTION_SHIFT = "distribution_shift"
    THRESHOLD_BREACH = "threshold_breach"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float
    threshold: float
    source: str
    resolved: bool = False
    auto_resolved: bool = False
    action_taken: Optional[str] = None


@dataclass
class ResponsePolicy:
    """Response policy configuration"""
    data_drift_threshold: float = 0.5
    accuracy_threshold: float = 0.80
    alert_threshold: int = 5
    auto_retrain_enabled: bool = True
    auto_rollback_enabled: bool = True
    rollback_threshold: float = 0.05
    manual_intervention_required: bool = False


class AlertingSystem:
    """
    Comprehensive alerting and response system for MLOps monitoring.
    
    Features:
    - Real-time model performance monitoring
    - Data drift detection and alerting
    - Automated response policies
    - Threshold-based alerting
    - Distribution shift detection
    - Model rollback automation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the alerting system"""
        self.config = config or {}
        self.policies = ResponsePolicy(**self.config.get('policies', {}))
        
        # Alert storage
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_lock = Lock()
        
        # Monitoring data
        self.monitoring_data = {
            'model_accuracy': [],
            'data_drift_scores': [],
            'alert_counts': [],
            'performance_metrics': [],
            'distribution_shift_detected': False,
            'last_shift_day': None
        }
        
        # Response actions tracking
        self.automated_actions = {
            'model_rollbacks': 0,
            'auto_retrains': 0,
            'engineer_notifications': 0,
            'system_scaling': 0,
            'last_action': None
        }
        
        # Database connection
        self.db = DatabaseManager()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        alert_logger.info("Alerting system initialized successfully")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_model_performance()
                self._check_data_drift()
                self._check_system_health()
                self._update_alert_counts()
                
                # Generate monitoring data for frontend
                self._generate_monitoring_data()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                alert_logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_model_performance(self):
        """Check model performance against thresholds"""
        try:
            # Get latest model accuracy from database
            query = """
            SELECT AVG(accuracy_score) as avg_accuracy, 
                   COUNT(*) as prediction_count,
                   MAX(timestamp) as last_prediction
            FROM model_predictions 
            WHERE timestamp >= datetime('now', '-1 hour')
            """
            
            result = self.db.execute_query(query)
            if result and result[0]['avg_accuracy']:
                accuracy = result[0]['avg_accuracy']
                
                # Check against threshold
                if accuracy < self.policies.accuracy_threshold:
                    self._create_alert(
                        AlertType.ACCURACY_DEGRADATION,
                        AlertSeverity.HIGH,
                        f"Model accuracy {accuracy:.3f} below threshold {self.policies.accuracy_threshold}",
                        accuracy,
                        self.policies.accuracy_threshold,
                        "model_performance"
                    )
                    
                    # Check if auto-rollback should be triggered
                    if self.policies.auto_rollback_enabled:
                        self._check_rollback_conditions(accuracy)
                
                # Store for monitoring
                self.monitoring_data['model_accuracy'].append({
                    'timestamp': datetime.now(),
                    'accuracy': accuracy,
                    'prediction_count': result[0]['prediction_count']
                })
                
        except Exception as e:
            alert_logger.error(f"Error checking model performance: {e}")
    
    def _check_data_drift(self):
        """Check for data drift using statistical tests"""
        try:
            # Get recent data distribution
            query = """
            SELECT voltage, temperature, current, capacity
            FROM battery_data 
            WHERE timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            
            recent_data = self.db.execute_query(query)
            if not recent_data:
                return
            
            # Convert to DataFrame for analysis
            df_recent = pd.DataFrame(recent_data)
            
            # Get training data distribution (baseline)
            query_baseline = """
            SELECT voltage, temperature, current, capacity
            FROM battery_data 
            WHERE timestamp >= datetime('now', '-30 days')
            AND timestamp < datetime('now', '-7 days')
            ORDER BY timestamp DESC
            LIMIT 5000
            """
            
            baseline_data = self.db.execute_query(query_baseline)
            if not baseline_data:
                return
            
            df_baseline = pd.DataFrame(baseline_data)
            
            # Calculate drift scores for each feature
            drift_scores = {}
            for feature in ['voltage', 'temperature', 'current', 'capacity']:
                if feature in df_recent.columns and feature in df_baseline.columns:
                    # Simple statistical drift detection
                    recent_mean = df_recent[feature].mean()
                    baseline_mean = df_baseline[feature].mean()
                    recent_std = df_recent[feature].std()
                    baseline_std = df_baseline[feature].std()
                    
                    # Calculate drift score (normalized difference)
                    mean_drift = abs(recent_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
                    std_drift = abs(recent_std - baseline_std) / baseline_std if baseline_std > 0 else 0
                    
                    drift_score = (mean_drift + std_drift) / 2
                    drift_scores[feature] = drift_score
                    
                    # Check against threshold
                    if drift_score > self.policies.data_drift_threshold:
                        self._create_alert(
                            AlertType.DATA_DRIFT,
                            AlertSeverity.MEDIUM,
                            f"Data drift detected in {feature}: {drift_score:.3f}",
                            drift_score,
                            self.policies.data_drift_threshold,
                            "data_drift"
                        )
            
            # Store drift scores
            self.monitoring_data['data_drift_scores'].append({
                'timestamp': datetime.now(),
                'scores': drift_scores
            })
            
        except Exception as e:
            alert_logger.error(f"Error checking data drift: {e}")
    
    def _check_system_health(self):
        """Check system health metrics"""
        try:
            # Check database connectivity
            health_check = self.db.execute_query("SELECT 1 as health_check")
            if not health_check:
                self._create_alert(
                    AlertType.SYSTEM_HEALTH,
                    AlertSeverity.CRITICAL,
                    "Database connectivity issue detected",
                    0.0,
                    1.0,
                    "system_health"
                )
            
            # Check for high error rates
            error_query = """
            SELECT COUNT(*) as error_count
            FROM model_predictions 
            WHERE timestamp >= datetime('now', '-1 hour')
            AND error_code IS NOT NULL
            """
            
            error_result = self.db.execute_query(error_query)
            if error_result and error_result[0]['error_count'] > 10:
                self._create_alert(
                    AlertType.SYSTEM_HEALTH,
                    AlertSeverity.HIGH,
                    f"High error rate detected: {error_result[0]['error_count']} errors in last hour",
                    error_result[0]['error_count'],
                    10,
                    "system_health"
                )
                
        except Exception as e:
            alert_logger.error(f"Error checking system health: {e}")
    
    def _check_rollback_conditions(self, current_accuracy: float):
        """Check if model rollback should be triggered"""
        try:
            # Get previous model accuracy
            query = """
            SELECT accuracy_score 
            FROM model_versions 
            WHERE is_active = 0 
            ORDER BY deployed_at DESC 
            LIMIT 1
            """
            
            result = self.db.execute_query(query)
            if result and result[0]['accuracy_score']:
                previous_accuracy = result[0]['accuracy_score']
                accuracy_degradation = previous_accuracy - current_accuracy
                
                if accuracy_degradation > self.policies.rollback_threshold:
                    self._trigger_model_rollback(
                        f"Accuracy degraded by {accuracy_degradation:.3f} (threshold: {self.policies.rollback_threshold})"
                    )
                    
        except Exception as e:
            alert_logger.error(f"Error checking rollback conditions: {e}")
    
    def _trigger_model_rollback(self, reason: str):
        """Trigger automated model rollback"""
        try:
            # Update model version status
            rollback_query = """
            UPDATE model_versions 
            SET is_active = CASE 
                WHEN is_active = 1 THEN 0 
                ELSE 1 
            END
            WHERE deployed_at = (
                SELECT deployed_at 
                FROM model_versions 
                WHERE is_active = 1 
                ORDER BY deployed_at DESC 
                LIMIT 1
            )
            """
            
            self.db.execute_query(rollback_query)
            
            # Record the action
            self.automated_actions['model_rollbacks'] += 1
            self.automated_actions['last_action'] = f"Model rollback: {reason}"
            
            # Create alert
            self._create_alert(
                AlertType.MODEL_PERFORMANCE,
                AlertSeverity.HIGH,
                f"Automated model rollback triggered: {reason}",
                0.0,
                0.0,
                "auto_rollback"
            )
            
            alert_logger.info(f"Model rollback triggered: {reason}")
            
        except Exception as e:
            alert_logger.error(f"Error triggering model rollback: {e}")
    
    def _create_alert(self, alert_type: AlertType, severity: AlertSeverity, 
                     message: str, metric_value: float, threshold: float, source: str):
        """Create a new alert"""
        with self.alert_lock:
            alert = Alert(
                id=f"alert_{int(time.time())}_{len(self.alerts)}",
                type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                metric_value=metric_value,
                threshold=threshold,
                source=source
            )
            
            self.alerts.append(alert)
            
            # Log the alert
            alert_logger.warning(f"Alert created: {alert_type.value} - {severity.value} - {message}")
            
            # Check if automated response should be triggered
            self._check_automated_response(alert)
    
    def _check_automated_response(self, alert: Alert):
        """Check if automated response should be triggered"""
        try:
            if alert.type == AlertType.DATA_DRIFT and self.policies.auto_retrain_enabled:
                self._trigger_auto_retrain("Data drift detected")
                
            elif alert.type == AlertType.ACCURACY_DEGRADATION and self.policies.auto_rollback_enabled:
                self._trigger_model_rollback("Accuracy degradation detected")
                
            elif alert.severity == AlertSeverity.CRITICAL:
                self._notify_engineering_team(alert)
                
        except Exception as e:
            alert_logger.error(f"Error in automated response: {e}")
    
    def _trigger_auto_retrain(self, reason: str):
        """Trigger automated model retraining"""
        try:
            # Record the action
            self.automated_actions['auto_retrains'] += 1
            self.automated_actions['last_action'] = f"Auto retrain: {reason}"
            
            alert_logger.info(f"Auto retrain triggered: {reason}")
            
            # In a real implementation, this would trigger the retraining pipeline
            # For now, we just log the action
            
        except Exception as e:
            alert_logger.error(f"Error triggering auto retrain: {e}")
    
    def _notify_engineering_team(self, alert: Alert):
        """Notify engineering team of critical alerts"""
        try:
            self.automated_actions['engineer_notifications'] += 1
            self.automated_actions['last_action'] = f"Engineer notification: {alert.message}"
            
            alert_logger.critical(f"Engineering team notified: {alert.message}")
            
            # In a real implementation, this would send notifications via Slack, email, etc.
            
        except Exception as e:
            alert_logger.error(f"Error notifying engineering team: {e}")
    
    def _update_alert_counts(self):
        """Update daily alert counts"""
        try:
            # Count alerts in the last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
            
            alert_count = len(recent_alerts)
            
            # Check against threshold
            if alert_count >= self.policies.alert_threshold:
                self._create_alert(
                    AlertType.THRESHOLD_BREACH,
                    AlertSeverity.MEDIUM,
                    f"Alert count {alert_count} exceeds threshold {self.policies.alert_threshold}",
                    alert_count,
                    self.policies.alert_threshold,
                    "alert_monitoring"
                )
            
            # Store alert count
            self.monitoring_data['alert_counts'].append({
                'timestamp': datetime.now(),
                'count': alert_count
            })
            
        except Exception as e:
            alert_logger.error(f"Error updating alert counts: {e}")
    
    def _generate_monitoring_data(self):
        """Generate monitoring data for frontend charts"""
        try:
            # Generate 30 days of simulated data for charts
            days = 30
            current_time = datetime.now()
            
            # Model accuracy data with realistic patterns
            model_accuracy = []
            for i in range(days):
                # Simulate realistic accuracy patterns with some degradation
                base_accuracy = 0.87
                if i >= 25:  # Simulate degradation in last 5 days
                    degradation = (i - 25) * 0.02
                    accuracy = max(0.77, base_accuracy - degradation)
                else:
                    # Add some realistic variation
                    variation = np.random.normal(0, 0.01)
                    accuracy = base_accuracy + variation
                    accuracy = max(0.85, min(0.88, accuracy))
                
                model_accuracy.append(accuracy)
            
            # Alert count data
            alert_counts = []
            for i in range(days):
                # Simulate alert patterns
                if i in [7, 11, 12, 13, 26, 27]:  # Days with high alerts
                    count = np.random.randint(8, 12)
                elif i in [26, 27]:  # Recent high alerts
                    count = np.random.randint(5, 8)
                else:
                    count = np.random.randint(1, 5)
                alert_counts.append(count)
            
            # Performance metric with distribution shift (60 days)
            performance_metric = []
            agreement_metric = []
            
            for i in range(60):
                if i < 30:  # Before distribution shift
                    base_perf = 0.92
                    variation = np.random.normal(0, 0.005)
                    perf = base_perf + variation
                    perf = max(0.91, min(0.93, perf))
                else:  # After distribution shift
                    base_perf = 0.82
                    variation = np.random.normal(0, 0.01)
                    perf = base_perf + variation
                    perf = max(0.79, min(0.86, perf))
                
                performance_metric.append(perf)
                
                # Agreement with ground truth (slightly lower after shift)
                if i < 30:
                    agreement = perf - np.random.uniform(0, 0.01)
                else:
                    agreement = perf - np.random.uniform(0.05, 0.08)
                
                agreement_metric.append(max(0.72, agreement))
            
            # Update monitoring data
            self.monitoring_data.update({
                'chart_data': {
                    'model_accuracy': model_accuracy,
                    'alert_counts': alert_counts,
                    'performance_metric': performance_metric,
                    'agreement_metric': agreement_metric,
                    'days': list(range(days)),
                    'performance_days': list(range(60))
                }
            })
            
        except Exception as e:
            alert_logger.error(f"Error generating monitoring data: {e}")
    
    def get_current_alerts(self) -> Dict[str, Any]:
        """Get current alerts status"""
        with self.alert_lock:
            active_alerts = len([a for a in self.alerts if not a.resolved])
            critical_alerts = len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved])
            data_drift_alerts = len([a for a in self.alerts if a.type == AlertType.DATA_DRIFT and not a.resolved])
            performance_alerts = len([a for a in self.alerts if a.type == AlertType.MODEL_PERFORMANCE and not a.resolved])
            
            return {
                'active_alerts': active_alerts,
                'critical_alerts': critical_alerts,
                'data_drift_alerts': data_drift_alerts,
                'performance_alerts': performance_alerts,
                'last_alert_time': self._get_last_alert_time(),
                'escalation_level': self._get_escalation_level()
            }
    
    def get_response_policies(self) -> Dict[str, Any]:
        """Get current response policies"""
        return asdict(self.policies)
    
    def get_automated_actions(self) -> Dict[str, Any]:
        """Get automated actions history"""
        return self.automated_actions.copy()
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get monitoring data for frontend charts"""
        return self.monitoring_data.copy()
    
    def _get_last_alert_time(self) -> str:
        """Get formatted last alert time"""
        if self.alerts:
            last_alert = max(self.alerts, key=lambda x: x.timestamp)
            time_diff = datetime.now() - last_alert.timestamp
            
            if time_diff.total_seconds() < 60:
                return f"{int(time_diff.total_seconds())} seconds ago"
            elif time_diff.total_seconds() < 3600:
                return f"{int(time_diff.total_seconds() // 60)} minutes ago"
            else:
                return f"{int(time_diff.total_seconds() // 3600)} hours ago"
        
        return "No alerts"
    
    def _get_escalation_level(self) -> str:
        """Get current escalation level based on active alerts"""
        critical_count = len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved])
        high_count = len([a for a in self.alerts if a.severity == AlertSeverity.HIGH and not a.resolved])
        
        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "high"
        elif high_count > 0:
            return "medium"
        else:
            return "low"
    
    def resolve_alert(self, alert_id: str, action_taken: str = None):
        """Resolve an alert"""
        with self.alert_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.action_taken = action_taken
                    self.alert_history.append(alert)
                    self.alerts.remove(alert)
                    alert_logger.info(f"Alert {alert_id} resolved: {action_taken}")
                    break
    
    def update_policies(self, new_policies: Dict[str, Any]):
        """Update response policies"""
        try:
            for key, value in new_policies.items():
                if hasattr(self.policies, key):
                    setattr(self.policies, key, value)
            
            alert_logger.info("Response policies updated")
            
        except Exception as e:
            alert_logger.error(f"Error updating policies: {e}")
    
    def shutdown(self):
        """Shutdown the alerting system"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        alert_logger.info("Alerting system shutdown complete")


# Global alerting system instance
alerting_system = None


def get_alerting_system(config: Optional[Dict] = None) -> AlertingSystem:
    """Get or create the global alerting system instance"""
    global alerting_system
    if alerting_system is None:
        alerting_system = AlertingSystem(config)
    return alerting_system


def shutdown_alerting_system():
    """Shutdown the global alerting system"""
    global alerting_system
    if alerting_system:
        alerting_system.shutdown()
        alerting_system = None 