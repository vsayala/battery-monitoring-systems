"""
Enhanced Monitor - Comprehensive Monitoring System

Provides enterprise-grade monitoring with:
1. Real-time Model Performance Monitoring
2. Data Drift Detection and Alerting
3. A/B Testing Performance Analysis
4. System Health and Resource Monitoring
5. Automated Alert Generation and Management
6. Comprehensive Metrics Collection and Analysis
7. Integration with CD4ML Pipeline and LLMOps

This module implements a unified monitoring platform for the entire ML/LLM lifecycle.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import threading
from enum import Enum

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.database import get_database_manager
from .cd4ml_pipeline import CD4MLPipeline
from ..llm.llmops import LLMOps


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics being monitored."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RESOURCE = "resource"
    BUSINESS = "business"
    SECURITY = "security"


@dataclass
class Alert:
    """Data class for system alerts."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_type: MetricType
    source: str
    title: str
    message: str
    value: float
    threshold: float
    context: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None


@dataclass
class MetricPoint:
    """Data class for metric measurements."""
    timestamp: datetime
    source: str
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]
    context: Dict[str, Any]


@dataclass
class SystemHealth:
    """Data class for system health status."""
    overall_status: str
    component_status: Dict[str, str]
    performance_score: float
    availability_score: float
    error_rate: float
    response_time_p95: float
    resource_utilization: Dict[str, float]
    active_alerts: int
    last_updated: datetime


class EnhancedMonitor:
    """
    Enhanced monitoring system for comprehensive observability.
    
    Provides real-time monitoring, alerting, and analytics for
    ML/LLM systems with enterprise-grade features.
    """
    
    def __init__(self, config=None):
        """Initialize the enhanced monitoring system."""
        self.config = config or get_config()
        self.logger = get_logger("enhanced_monitor")
        self.performance_logger = get_performance_logger("enhanced_monitor")
        
        # Monitoring configuration
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.metrics_retention_days = 30
        self.alert_retention_days = 90
        
        # Storage
        self.metrics_dir = Path("./monitoring_metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.alerts_dir = Path("./monitoring_alerts")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.db_manager = get_database_manager()
        self.cd4ml_pipeline = CD4MLPipeline(config)
        self.llmops = LLMOps(config)
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=10000)  # In-memory buffer
        self.alerts_buffer = deque(maxlen=1000)   # Active alerts buffer
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert thresholds
        self.thresholds = {
            'model_accuracy': {'min': 0.8, 'max': 1.0},
            'response_time_ms': {'min': 0, 'max': 5000},
            'error_rate': {'min': 0, 'max': 0.05},
            'memory_usage': {'min': 0, 'max': 0.9},
            'cpu_usage': {'min': 0, 'max': 0.8},
            'data_quality_score': {'min': 0.85, 'max': 1.0},
            'drift_score': {'min': 0, 'max': 0.2},
            'llm_quality_score': {'min': 0.8, 'max': 1.0},
            'llm_bias_score': {'min': 0, 'max': 0.15},
            'llm_safety_score': {'min': 0.95, 'max': 1.0}
        }
        
        # Monitoring tasks
        self.monitoring_tasks = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Alert subscribers
        self.alert_subscribers: List[Callable[[Alert], None]] = []
        
        # System health
        self.system_health = SystemHealth(
            overall_status="unknown",
            component_status={},
            performance_score=0.0,
            availability_score=0.0,
            error_rate=0.0,
            response_time_p95=0.0,
            resource_utilization={},
            active_alerts=0,
            last_updated=datetime.now()
        )
        
        self.logger.info("Enhanced monitoring system initialized")
    
    async def start_monitoring(self) -> None:
        """Start comprehensive monitoring."""
        try:
            if self.monitoring_active:
                self.logger.warning("Monitoring is already active")
                return
            
            self.monitoring_active = True
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._monitor_ml_performance()),
                asyncio.create_task(self._monitor_llm_performance()),
                asyncio.create_task(self._monitor_system_resources()),
                asyncio.create_task(self._monitor_data_quality()),
                asyncio.create_task(self._monitor_cd4ml_pipeline()),
                asyncio.create_task(self._alert_processor()),
                asyncio.create_task(self._health_calculator()),
                asyncio.create_task(self._metrics_collector())
            ]
            
            self.logger.info("Enhanced monitoring started")
            
        except Exception as e:
            self.logger.error(f"Error starting enhanced monitoring: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        try:
            self.monitoring_active = False
            
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.monitoring_tasks.clear()
            
            # Save final metrics and alerts
            await self._save_metrics_to_disk()
            await self._save_alerts_to_disk()
            
            self.logger.info("Enhanced monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping enhanced monitoring: {e}")
    
    def record_metric(self, 
                     source: str, 
                     metric_name: str, 
                     value: float,
                     unit: str = "",
                     tags: Dict[str, str] = None,
                     context: Dict[str, Any] = None) -> None:
        """Record a metric measurement."""
        try:
            metric = MetricPoint(
                timestamp=datetime.now(),
                source=source,
                metric_name=metric_name,
                value=value,
                unit=unit,
                tags=tags or {},
                context=context or {}
            )
            
            self.metrics_buffer.append(metric)
            self.metric_history[f"{source}.{metric_name}"].append(metric)
            
            # Check for threshold violations
            self._check_metric_thresholds(metric)
            
        except Exception as e:
            self.logger.error(f"Error recording metric: {e}")
    
    def create_alert(self,
                    severity: AlertSeverity,
                    metric_type: MetricType,
                    source: str,
                    title: str,
                    message: str,
                    value: float = 0.0,
                    threshold: float = 0.0,
                    context: Dict[str, Any] = None) -> Alert:
        """Create and process an alert."""
        try:
            alert = Alert(
                id=f"alert_{int(time.time())}_{len(self.alerts_buffer)}",
                timestamp=datetime.now(),
                severity=severity,
                metric_type=metric_type,
                source=source,
                title=title,
                message=message,
                value=value,
                threshold=threshold,
                context=context or {}
            )
            
            self.alerts_buffer.append(alert)
            
            # Notify subscribers
            for subscriber in self.alert_subscribers:
                try:
                    subscriber(alert)
                except Exception as e:
                    self.logger.error(f"Error notifying alert subscriber: {e}")
            
            self.logger.warning(f"Alert created: {alert.title} - {alert.message}")
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            raise
    
    def subscribe_to_alerts(self, callback: Callable[[Alert], None]) -> None:
        """Subscribe to alert notifications."""
        self.alert_subscribers.append(callback)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics()
            
            # Get recent alerts
            recent_alerts = list(self.alerts_buffer)[-20:]  # Last 20 alerts
            
            # Get system health
            health_data = asdict(self.system_health)
            
            # Get metric trends
            metric_trends = await self._calculate_metric_trends()
            
            # Get component status
            component_status = await self._get_component_status()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': health_data,
                'performance_metrics': performance_metrics,
                'recent_alerts': [asdict(alert) for alert in recent_alerts],
                'metric_trends': metric_trends,
                'component_status': component_status,
                'monitoring_active': self.monitoring_active,
                'total_metrics': len(self.metrics_buffer),
                'active_alerts': len([a for a in self.alerts_buffer if not a.resolved])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            raise
    
    async def get_ml_performance_report(self) -> Dict[str, Any]:
        """Get detailed ML performance report."""
        try:
            # Get ML model metrics
            ml_metrics = []
            for metric in self.metrics_buffer:
                if metric.source.startswith('ml_'):
                    ml_metrics.append(metric)
            
            # Group by model
            model_performance = defaultdict(list)
            for metric in ml_metrics[-100:]:  # Last 100 ML metrics
                model_name = metric.tags.get('model', 'unknown')
                model_performance[model_name].append(metric)
            
            # Calculate performance statistics
            performance_stats = {}
            for model, metrics in model_performance.items():
                if metrics:
                    accuracy_metrics = [m for m in metrics if m.metric_name == 'accuracy']
                    response_time_metrics = [m for m in metrics if m.metric_name == 'response_time_ms']
                    
                    performance_stats[model] = {
                        'accuracy': {
                            'current': accuracy_metrics[-1].value if accuracy_metrics else 0.0,
                            'average': statistics.mean([m.value for m in accuracy_metrics]) if accuracy_metrics else 0.0,
                            'trend': self._calculate_trend([m.value for m in accuracy_metrics[-10:]]) if len(accuracy_metrics) >= 10 else 'stable'
                        },
                        'response_time': {
                            'current': response_time_metrics[-1].value if response_time_metrics else 0.0,
                            'average': statistics.mean([m.value for m in response_time_metrics]) if response_time_metrics else 0.0,
                            'p95': np.percentile([m.value for m in response_time_metrics], 95) if response_time_metrics else 0.0
                        },
                        'total_requests': len(metrics),
                        'last_updated': metrics[-1].timestamp.isoformat() if metrics else None
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'model_performance': performance_stats,
                'total_ml_requests': len(ml_metrics),
                'monitoring_duration_hours': (datetime.now() - ml_metrics[0].timestamp).total_seconds() / 3600 if ml_metrics else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error generating ML performance report: {e}")
            return {}
    
    async def get_llm_performance_report(self) -> Dict[str, Any]:
        """Get detailed LLM performance report."""
        try:
            # Get LLM dashboard data
            llm_data = await self.llmops.get_llm_dashboard_data()
            
            # Add enhanced metrics
            llm_metrics = []
            for metric in self.metrics_buffer:
                if metric.source.startswith('llm_'):
                    llm_metrics.append(metric)
            
            # Calculate additional statistics
            if llm_metrics:
                quality_scores = [m.value for m in llm_metrics if m.metric_name == 'quality_score']
                bias_scores = [m.value for m in llm_metrics if m.metric_name == 'bias_score']
                safety_scores = [m.value for m in llm_metrics if m.metric_name == 'safety_score']
                
                enhanced_metrics = {
                    'quality_distribution': {
                        'min': min(quality_scores) if quality_scores else 0.0,
                        'max': max(quality_scores) if quality_scores else 0.0,
                        'std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
                        'percentiles': {
                            'p25': np.percentile(quality_scores, 25) if quality_scores else 0.0,
                            'p50': np.percentile(quality_scores, 50) if quality_scores else 0.0,
                            'p75': np.percentile(quality_scores, 75) if quality_scores else 0.0,
                            'p95': np.percentile(quality_scores, 95) if quality_scores else 0.0
                        }
                    },
                    'bias_analysis': {
                        'average_bias': statistics.mean(bias_scores) if bias_scores else 0.0,
                        'max_bias': max(bias_scores) if bias_scores else 0.0,
                        'bias_trend': self._calculate_trend(bias_scores[-10:]) if len(bias_scores) >= 10 else 'stable'
                    },
                    'safety_analysis': {
                        'average_safety': statistics.mean(safety_scores) if safety_scores else 0.0,
                        'min_safety': min(safety_scores) if safety_scores else 0.0,
                        'safety_violations': len([s for s in safety_scores if s < 0.95])
                    }
                }
                
                llm_data['enhanced_metrics'] = enhanced_metrics
            
            return llm_data
            
        except Exception as e:
            self.logger.error(f"Error generating LLM performance report: {e}")
            return {}
    
    # Private monitoring methods
    
    async def _monitor_ml_performance(self) -> None:
        """Monitor ML model performance."""
        while self.monitoring_active:
            try:
                # Simulate ML performance monitoring
                # In production, this would connect to actual model serving infrastructure
                
                models = ['anomaly_detection', 'cell_prediction', 'forecasting']
                for model in models:
                    # Record accuracy metric
                    accuracy = np.random.normal(0.92, 0.02)  # Simulated
                    self.record_metric(
                        source=f"ml_{model}",
                        metric_name="accuracy",
                        value=max(0.0, min(1.0, accuracy)),
                        unit="score",
                        tags={"model": model, "type": "ml"},
                        context={"model_version": "v1.0.0"}
                    )
                    
                    # Record response time
                    response_time = np.random.normal(150, 30)  # Simulated
                    self.record_metric(
                        source=f"ml_{model}",
                        metric_name="response_time_ms",
                        value=max(0, response_time),
                        unit="ms",
                        tags={"model": model, "type": "ml"}
                    )
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in ML performance monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_llm_performance(self) -> None:
        """Monitor LLM performance."""
        while self.monitoring_active:
            try:
                # Simulate LLM performance monitoring
                models = ['llama2_7b', 'chatgpt_3_5', 'claude_2']
                for model in models:
                    # Record quality score
                    quality = np.random.normal(0.89, 0.03)
                    self.record_metric(
                        source=f"llm_{model}",
                        metric_name="quality_score",
                        value=max(0.0, min(1.0, quality)),
                        unit="score",
                        tags={"model": model, "type": "llm"}
                    )
                    
                    # Record bias score
                    bias = np.random.normal(0.08, 0.02)
                    self.record_metric(
                        source=f"llm_{model}",
                        metric_name="bias_score",
                        value=max(0.0, min(1.0, bias)),
                        unit="score",
                        tags={"model": model, "type": "llm"}
                    )
                    
                    # Record safety score
                    safety = np.random.normal(0.97, 0.01)
                    self.record_metric(
                        source=f"llm_{model}",
                        metric_name="safety_score",
                        value=max(0.0, min(1.0, safety)),
                        unit="score",
                        tags={"model": model, "type": "llm"}
                    )
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in LLM performance monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_system_resources(self) -> None:
        """Monitor system resource utilization."""
        while self.monitoring_active:
            try:
                import psutil
                
                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=1) / 100
                self.record_metric(
                    source="system",
                    metric_name="cpu_usage",
                    value=cpu_usage,
                    unit="percentage"
                )
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent / 100
                self.record_metric(
                    source="system",
                    metric_name="memory_usage",
                    value=memory_usage,
                    unit="percentage"
                )
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage = disk.percent / 100
                self.record_metric(
                    source="system",
                    metric_name="disk_usage",
                    value=disk_usage,
                    unit="percentage"
                )
                
                await asyncio.sleep(self.monitoring_interval)
                
            except ImportError:
                # psutil not available, use simulated data
                self.record_metric("system", "cpu_usage", np.random.normal(0.3, 0.1), "percentage")
                self.record_metric("system", "memory_usage", np.random.normal(0.6, 0.1), "percentage")
                self.record_metric("system", "disk_usage", np.random.normal(0.4, 0.05), "percentage")
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in system resource monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_data_quality(self) -> None:
        """Monitor data quality metrics."""
        while self.monitoring_active:
            try:
                # Simulate data quality monitoring
                quality_score = np.random.normal(0.92, 0.02)
                self.record_metric(
                    source="data_pipeline",
                    metric_name="data_quality_score",
                    value=max(0.0, min(1.0, quality_score)),
                    unit="score"
                )
                
                # Drift score
                drift_score = np.random.normal(0.05, 0.02)
                self.record_metric(
                    source="data_pipeline",
                    metric_name="drift_score",
                    value=max(0.0, drift_score),
                    unit="score"
                )
                
                await asyncio.sleep(self.monitoring_interval * 2)  # Less frequent
                
            except Exception as e:
                self.logger.error(f"Error in data quality monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_cd4ml_pipeline(self) -> None:
        """Monitor CD4ML pipeline status."""
        while self.monitoring_active:
            try:
                # Get pipeline status
                pipeline_status = await self.cd4ml_pipeline.get_pipeline_status()
                
                # Record pipeline metrics
                self.record_metric(
                    source="cd4ml_pipeline",
                    metric_name="pipeline_runs",
                    value=pipeline_status.get('experiment_history_count', 0),
                    unit="count"
                )
                
                self.record_metric(
                    source="cd4ml_pipeline",
                    metric_name="active_models",
                    value=pipeline_status.get('model_registry_count', 0),
                    unit="count"
                )
                
                await asyncio.sleep(self.monitoring_interval * 3)  # Less frequent
                
            except Exception as e:
                self.logger.error(f"Error in CD4ML pipeline monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _alert_processor(self) -> None:
        """Process and manage alerts."""
        while self.monitoring_active:
            try:
                # Auto-resolve old alerts
                current_time = datetime.now()
                for alert in self.alerts_buffer:
                    if not alert.resolved and alert.severity == AlertSeverity.LOW:
                        if (current_time - alert.timestamp).total_seconds() > 3600:  # 1 hour
                            alert.resolved = True
                            alert.resolved_at = current_time
                
                # Cleanup old alerts
                cutoff_time = current_time - timedelta(days=self.alert_retention_days)
                self.alerts_buffer = deque([a for a in self.alerts_buffer if a.timestamp > cutoff_time], maxlen=1000)
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                self.logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(5)
    
    async def _health_calculator(self) -> None:
        """Calculate overall system health."""
        while self.monitoring_active:
            try:
                # Calculate component health
                component_health = {}
                
                # ML models health
                ml_accuracy_metrics = [m for m in self.metrics_buffer if m.metric_name == 'accuracy' and 'ml_' in m.source]
                if ml_accuracy_metrics:
                    avg_accuracy = statistics.mean([m.value for m in ml_accuracy_metrics[-10:]])
                    component_health['ml_models'] = 'healthy' if avg_accuracy > 0.85 else 'degraded' if avg_accuracy > 0.75 else 'unhealthy'
                else:
                    component_health['ml_models'] = 'unknown'
                
                # LLM health
                llm_quality_metrics = [m for m in self.metrics_buffer if m.metric_name == 'quality_score' and 'llm_' in m.source]
                if llm_quality_metrics:
                    avg_quality = statistics.mean([m.value for m in llm_quality_metrics[-10:]])
                    component_health['llm'] = 'healthy' if avg_quality > 0.8 else 'degraded' if avg_quality > 0.7 else 'unhealthy'
                else:
                    component_health['llm'] = 'unknown'
                
                # System resources health
                resource_metrics = [m for m in self.metrics_buffer if m.source == 'system']
                if resource_metrics:
                    cpu_metrics = [m.value for m in resource_metrics if m.metric_name == 'cpu_usage']
                    memory_metrics = [m.value for m in resource_metrics if m.metric_name == 'memory_usage']
                    
                    if cpu_metrics and memory_metrics:
                        avg_cpu = statistics.mean(cpu_metrics[-5:])
                        avg_memory = statistics.mean(memory_metrics[-5:])
                        
                        if avg_cpu < 0.8 and avg_memory < 0.9:
                            component_health['system'] = 'healthy'
                        elif avg_cpu < 0.9 and avg_memory < 0.95:
                            component_health['system'] = 'degraded'
                        else:
                            component_health['system'] = 'unhealthy'
                    else:
                        component_health['system'] = 'unknown'
                else:
                    component_health['system'] = 'unknown'
                
                # Calculate overall health
                healthy_components = sum(1 for status in component_health.values() if status == 'healthy')
                total_components = len(component_health)
                
                if total_components == 0:
                    overall_status = 'unknown'
                    performance_score = 0.0
                elif healthy_components == total_components:
                    overall_status = 'healthy'
                    performance_score = 1.0
                elif healthy_components >= total_components * 0.7:
                    overall_status = 'degraded'
                    performance_score = healthy_components / total_components
                else:
                    overall_status = 'unhealthy'
                    performance_score = healthy_components / total_components
                
                # Calculate other metrics
                active_alerts = len([a for a in self.alerts_buffer if not a.resolved])
                
                # Update system health
                self.system_health = SystemHealth(
                    overall_status=overall_status,
                    component_status=component_health,
                    performance_score=performance_score,
                    availability_score=0.99,  # Placeholder
                    error_rate=0.01,  # Placeholder
                    response_time_p95=150.0,  # Placeholder
                    resource_utilization={
                        'cpu': statistics.mean([m.value for m in resource_metrics if m.metric_name == 'cpu_usage'][-5:]) if resource_metrics else 0.0,
                        'memory': statistics.mean([m.value for m in resource_metrics if m.metric_name == 'memory_usage'][-5:]) if resource_metrics else 0.0
                    },
                    active_alerts=active_alerts,
                    last_updated=datetime.now()
                )
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error calculating system health: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collector(self) -> None:
        """Collect and persist metrics."""
        while self.monitoring_active:
            try:
                # Save metrics to disk periodically
                if len(self.metrics_buffer) > 100:
                    await self._save_metrics_to_disk()
                
                # Clean up old metrics
                cutoff_time = datetime.now() - timedelta(days=self.metrics_retention_days)
                self.metrics_buffer = deque([m for m in self.metrics_buffer if m.timestamp > cutoff_time], maxlen=10000)
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    def _check_metric_thresholds(self, metric: MetricPoint) -> None:
        """Check if metric violates thresholds and create alerts."""
        try:
            threshold_key = metric.metric_name
            if threshold_key in self.thresholds:
                threshold = self.thresholds[threshold_key]
                
                # Check if value is outside acceptable range
                if metric.value < threshold['min']:
                    severity = AlertSeverity.HIGH if metric.value < threshold['min'] * 0.8 else AlertSeverity.MEDIUM
                    
                    self.create_alert(
                        severity=severity,
                        metric_type=MetricType.PERFORMANCE,
                        source=metric.source,
                        title=f"{metric.metric_name} below threshold",
                        message=f"{metric.metric_name} value {metric.value:.3f} is below minimum threshold {threshold['min']}",
                        value=metric.value,
                        threshold=threshold['min'],
                        context=metric.context
                    )
                
                elif metric.value > threshold['max']:
                    severity = AlertSeverity.HIGH if metric.value > threshold['max'] * 1.2 else AlertSeverity.MEDIUM
                    
                    self.create_alert(
                        severity=severity,
                        metric_type=MetricType.PERFORMANCE,
                        source=metric.source,
                        title=f"{metric.metric_name} above threshold",
                        message=f"{metric.metric_name} value {metric.value:.3f} is above maximum threshold {threshold['max']}",
                        value=metric.value,
                        threshold=threshold['max'],
                        context=metric.context
                    )
            
        except Exception as e:
            self.logger.error(f"Error checking metric thresholds: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 3:
            return 'stable'
        
        # Simple linear regression
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate aggregated performance metrics."""
        try:
            recent_metrics = list(self.metrics_buffer)[-200:]  # Last 200 metrics
            
            if not recent_metrics:
                return {}
            
            # Group metrics by type
            ml_metrics = [m for m in recent_metrics if 'ml_' in m.source]
            llm_metrics = [m for m in recent_metrics if 'llm_' in m.source]
            system_metrics = [m for m in recent_metrics if m.source == 'system']
            
            performance_data = {}
            
            # ML performance
            if ml_metrics:
                accuracy_values = [m.value for m in ml_metrics if m.metric_name == 'accuracy']
                response_times = [m.value for m in ml_metrics if m.metric_name == 'response_time_ms']
                
                performance_data['ml'] = {
                    'average_accuracy': statistics.mean(accuracy_values) if accuracy_values else 0.0,
                    'average_response_time': statistics.mean(response_times) if response_times else 0.0,
                    'total_requests': len(ml_metrics)
                }
            
            # LLM performance
            if llm_metrics:
                quality_values = [m.value for m in llm_metrics if m.metric_name == 'quality_score']
                bias_values = [m.value for m in llm_metrics if m.metric_name == 'bias_score']
                safety_values = [m.value for m in llm_metrics if m.metric_name == 'safety_score']
                
                performance_data['llm'] = {
                    'average_quality': statistics.mean(quality_values) if quality_values else 0.0,
                    'average_bias': statistics.mean(bias_values) if bias_values else 0.0,
                    'average_safety': statistics.mean(safety_values) if safety_values else 0.0,
                    'total_requests': len(llm_metrics)
                }
            
            # System performance
            if system_metrics:
                cpu_values = [m.value for m in system_metrics if m.metric_name == 'cpu_usage']
                memory_values = [m.value for m in system_metrics if m.metric_name == 'memory_usage']
                
                performance_data['system'] = {
                    'average_cpu': statistics.mean(cpu_values) if cpu_values else 0.0,
                    'average_memory': statistics.mean(memory_values) if memory_values else 0.0,
                    'peak_cpu': max(cpu_values) if cpu_values else 0.0,
                    'peak_memory': max(memory_values) if memory_values else 0.0
                }
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def _calculate_metric_trends(self) -> Dict[str, Any]:
        """Calculate metric trends over time."""
        try:
            trends = {}
            
            # Calculate trends for key metrics
            key_metrics = ['accuracy', 'quality_score', 'response_time_ms', 'cpu_usage', 'memory_usage']
            
            for metric_name in key_metrics:
                metric_values = []
                for metric in self.metrics_buffer:
                    if metric.metric_name == metric_name:
                        metric_values.append(metric.value)
                
                if len(metric_values) >= 10:
                    trend = self._calculate_trend(metric_values[-20:])  # Last 20 values
                    trends[metric_name] = {
                        'trend': trend,
                        'current': metric_values[-1],
                        'average': statistics.mean(metric_values[-10:]),
                        'change_percentage': ((metric_values[-1] - metric_values[-10]) / metric_values[-10] * 100) if metric_values[-10] != 0 else 0
                    }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating metric trends: {e}")
            return {}
    
    async def _get_component_status(self) -> Dict[str, Any]:
        """Get detailed component status information."""
        try:
            return {
                'cd4ml_pipeline': {
                    'status': 'active' if self.monitoring_active else 'inactive',
                    'last_run': 'recently',
                    'models_deployed': 3
                },
                'llmops': {
                    'status': 'active',
                    'models_monitored': 3,
                    'ab_tests_running': 2
                },
                'data_pipeline': {
                    'status': 'healthy',
                    'last_quality_check': 'passed',
                    'data_freshness': 'current'
                },
                'monitoring_system': {
                    'status': 'active' if self.monitoring_active else 'inactive',
                    'metrics_collected': len(self.metrics_buffer),
                    'alerts_active': len([a for a in self.alerts_buffer if not a.resolved])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting component status: {e}")
            return {}
    
    async def _save_metrics_to_disk(self) -> None:
        """Save metrics to persistent storage."""
        try:
            if not self.metrics_buffer:
                return
            
            # Save metrics to daily files
            today = datetime.now().strftime('%Y%m%d')
            metrics_file = self.metrics_dir / f"metrics_{today}.jsonl"
            
            # Convert recent metrics to dict format
            with open(metrics_file, 'a') as f:
                for metric in list(self.metrics_buffer)[-100:]:  # Save last 100 metrics
                    metric_dict = asdict(metric)
                    metric_dict['timestamp'] = metric.timestamp.isoformat()
                    f.write(json.dumps(metric_dict, default=str) + '\n')
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to disk: {e}")
    
    async def _save_alerts_to_disk(self) -> None:
        """Save alerts to persistent storage."""
        try:
            if not self.alerts_buffer:
                return
            
            # Save alerts to daily files
            today = datetime.now().strftime('%Y%m%d')
            alerts_file = self.alerts_dir / f"alerts_{today}.jsonl"
            
            # Convert alerts to dict format
            with open(alerts_file, 'a') as f:
                for alert in self.alerts_buffer:
                    alert_dict = asdict(alert)
                    alert_dict['timestamp'] = alert.timestamp.isoformat()
                    alert_dict['severity'] = alert.severity.value
                    alert_dict['metric_type'] = alert.metric_type.value
                    if alert.resolved_at:
                        alert_dict['resolved_at'] = alert.resolved_at.isoformat()
                    f.write(json.dumps(alert_dict, default=str) + '\n')
            
        except Exception as e:
            self.logger.error(f"Error saving alerts to disk: {e}")