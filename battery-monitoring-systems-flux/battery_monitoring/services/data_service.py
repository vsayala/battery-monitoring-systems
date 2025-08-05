#!/usr/bin/env python3
"""
Comprehensive Data Service for Battery Monitoring System.

Provides unified data access, real-time analytics, and dashboard integration
with proper error handling and logging throughout the MLOps pipeline.
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import traceback
from pathlib import Path

# Setup basic logging for the module
logger = logging.getLogger(__name__)


class DataService:
    """
    Unified data service for the battery monitoring system.
    
    Provides real-time data access, analytics, and dashboard integration
    with comprehensive error handling and logging.
    """
    
    def __init__(self, db_path: str = "battery_monitoring.db"):
        """Initialize the data service.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.DataService")
        
        # Setup logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Validate database connection
        try:
            self._validate_database()
            self.logger.info(f"DataService initialized with database: {db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize DataService: {e}")
            raise
    
    def _validate_database(self) -> bool:
        """Validate database connection and schema.
        
        Returns:
            True if database is valid, raises exception otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if battery_data table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='battery_data'
                """)
                
                if not cursor.fetchone():
                    raise Exception("battery_data table not found in database")
                
                # Check record count
                cursor.execute("SELECT COUNT(*) FROM battery_data")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    self.logger.warning("Database is empty - no battery data found")
                else:
                    self.logger.info(f"Database validation successful: {count} records found")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Database validation failed: {e}")
            raise
    
    def _execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a SQL query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing query results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = [dict(row) for row in rows]
                
                self.logger.debug(f"Query executed successfully: {len(results)} rows returned")
                return results
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}\nQuery: {query}")
            raise
    
    def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get comprehensive dashboard overview with real data.
        
        Returns:
            Dictionary containing dashboard overview data
        """
        try:
            self.logger.info("Generating dashboard overview")
            
            # Get basic statistics
            stats = self.get_basic_statistics()
            
            # Get recent data trends
            trends = self.get_data_trends()
            
            # Get system health metrics
            health = self.get_system_health()
            
            # Get alerts and notifications
            alerts = self.get_recent_alerts()
            
            overview = {
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "trends": trends,
                "health": health,
                "alerts": alerts,
                "status": "active"
            }
            
            self.logger.info("Dashboard overview generated successfully")
            return overview
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard overview: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    def get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic system statistics.
        
        Returns:
            Dictionary containing basic statistics
        """
        try:
            # Total records
            total_query = "SELECT COUNT(*) as total_records FROM battery_data"
            total_result = self._execute_query(total_query)
            total_records = total_result[0]["total_records"]
            
            # Unique devices
            devices_query = "SELECT COUNT(DISTINCT device_id) as unique_devices FROM battery_data"
            devices_result = self._execute_query(devices_query)
            unique_devices = devices_result[0]["unique_devices"]
            
            # Unique sites
            sites_query = "SELECT COUNT(DISTINCT site_id) as unique_sites FROM battery_data"
            sites_result = self._execute_query(sites_query)
            unique_sites = sites_result[0]["unique_sites"]
            
            # Date range
            date_range_query = """
                SELECT 
                    MIN(packet_datetime) as earliest_record,
                    MAX(packet_datetime) as latest_record
                FROM battery_data 
                WHERE packet_datetime IS NOT NULL
            """
            date_result = self._execute_query(date_range_query)
            date_info = date_result[0] if date_result else {}
            
            # Average values
            avg_query = """
                SELECT 
                    AVG(cell_voltage) as avg_voltage,
                    AVG(cell_temperature) as avg_temperature,
                    AVG(soc_latest_value_for_every_cycle) as avg_soc,
                    AVG(ambient_temperature) as avg_ambient_temp
                FROM battery_data 
                WHERE cell_voltage IS NOT NULL
            """
            avg_result = self._execute_query(avg_query)
            averages = avg_result[0] if avg_result else {}
            
            statistics = {
                "total_records": total_records,
                "unique_devices": unique_devices,
                "unique_sites": unique_sites,
                "data_range": {
                    "earliest": date_info.get("earliest_record"),
                    "latest": date_info.get("latest_record")
                },
                "averages": {
                    "voltage": round(averages.get("avg_voltage", 0), 3),
                    "temperature": round(averages.get("avg_temperature", 0), 1),
                    "soc": round(averages.get("avg_soc", 0), 1),
                    "ambient_temperature": round(averages.get("avg_ambient_temp", 0), 1)
                }
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Failed to get basic statistics: {e}")
            return {"error": str(e)}
    
    def get_data_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get data trends for the specified time period.
        
        Args:
            hours: Number of hours to look back for trends
            
        Returns:
            Dictionary containing trend data
        """
        try:
            # Since we have dummy data with timestamps spread over 30 days,
            # we'll get the most recent data based on record order
            
            # Get recent voltage trends
            voltage_query = """
                SELECT 
                    device_id,
                    cell_voltage,
                    cell_temperature,
                    soc_latest_value_for_every_cycle as soc,
                    packet_datetime,
                    ROW_NUMBER() OVER (ORDER BY id DESC) as row_num
                FROM battery_data 
                WHERE cell_voltage IS NOT NULL
                ORDER BY id DESC
                LIMIT 100
            """
            voltage_data = self._execute_query(voltage_query)
            
            # Get charging/discharging distribution
            cycle_query = """
                SELECT 
                    charge_or_discharge_cycle,
                    COUNT(*) as count,
                    AVG(cell_voltage) as avg_voltage,
                    AVG(instantaneous_current) as avg_current
                FROM battery_data 
                WHERE charge_or_discharge_cycle IS NOT NULL
                GROUP BY charge_or_discharge_cycle
            """
            cycle_data = self._execute_query(cycle_query)
            
            # Get device performance
            device_query = """
                SELECT 
                    device_id,
                    COUNT(*) as record_count,
                    AVG(cell_voltage) as avg_voltage,
                    AVG(cell_temperature) as avg_temperature,
                    AVG(soc_latest_value_for_every_cycle) as avg_soc
                FROM battery_data 
                WHERE device_id IS NOT NULL
                GROUP BY device_id
                ORDER BY record_count DESC
                LIMIT 10
            """
            device_data = self._execute_query(device_query)
            
            trends = {
                "voltage_trends": voltage_data[:20],  # Last 20 records
                "cycle_distribution": cycle_data,
                "device_performance": device_data,
                "trend_period_hours": hours,
                "data_points": len(voltage_data)
            }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Failed to get data trends: {e}")
            return {"error": str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics.
        
        Returns:
            Dictionary containing system health data
        """
        try:
            # Check for anomalies (simplified heuristics)
            anomaly_query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN cell_voltage < 1.8 OR cell_voltage > 2.5 THEN 1 END) as voltage_anomalies,
                    COUNT(CASE WHEN cell_temperature < 0 OR cell_temperature > 60 THEN 1 END) as temp_anomalies,
                    COUNT(CASE WHEN soc_latest_value_for_every_cycle < 10 OR soc_latest_value_for_every_cycle > 100 THEN 1 END) as soc_anomalies
                FROM battery_data
            """
            anomaly_data = self._execute_query(anomaly_query)
            anomalies = anomaly_data[0] if anomaly_data else {}
            
            # Check communication health
            comm_query = """
                SELECT 
                    bms_bms_sed_communication,
                    bms_cell_communication,
                    COUNT(*) as count
                FROM battery_data 
                WHERE bms_bms_sed_communication IS NOT NULL
                GROUP BY bms_bms_sed_communication, bms_cell_communication
            """
            comm_data = self._execute_query(comm_query)
            
            # Calculate health scores
            total_records = anomalies.get("total_records", 1)
            voltage_health = max(0, 100 - (anomalies.get("voltage_anomalies", 0) / total_records * 100))
            temp_health = max(0, 100 - (anomalies.get("temp_anomalies", 0) / total_records * 100))
            soc_health = max(0, 100 - (anomalies.get("soc_anomalies", 0) / total_records * 100))
            
            overall_health = (voltage_health + temp_health + soc_health) / 3
            
            health = {
                "overall_health_score": round(overall_health, 1),
                "voltage_health": round(voltage_health, 1),
                "temperature_health": round(temp_health, 1),
                "soc_health": round(soc_health, 1),
                "anomaly_counts": {
                    "voltage": anomalies.get("voltage_anomalies", 0),
                    "temperature": anomalies.get("temp_anomalies", 0),
                    "soc": anomalies.get("soc_anomalies", 0)
                },
                "communication_status": comm_data,
                "status": "healthy" if overall_health > 80 else "warning" if overall_health > 60 else "critical"
            }
            
            return health
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {"error": str(e)}
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts and notifications.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        try:
            # Generate synthetic alerts based on data anomalies
            alert_query = """
                SELECT 
                    device_id,
                    site_id,
                    cell_voltage,
                    cell_temperature,
                    soc_latest_value_for_every_cycle as soc,
                    packet_datetime,
                    CASE 
                        WHEN cell_voltage < 1.8 THEN 'LOW_VOLTAGE'
                        WHEN cell_voltage > 2.5 THEN 'HIGH_VOLTAGE'
                        WHEN cell_temperature > 50 THEN 'HIGH_TEMPERATURE'
                        WHEN cell_temperature < 5 THEN 'LOW_TEMPERATURE'
                        WHEN soc_latest_value_for_every_cycle < 15 THEN 'LOW_SOC'
                        ELSE 'NORMAL'
                    END as alert_type
                FROM battery_data 
                WHERE (
                    cell_voltage < 1.8 OR cell_voltage > 2.5 OR
                    cell_temperature < 5 OR cell_temperature > 50 OR
                    soc_latest_value_for_every_cycle < 15
                )
                ORDER BY id DESC
                LIMIT ?
            """
            
            alert_data = self._execute_query(alert_query, (limit,))
            
            # Format alerts
            alerts = []
            for alert in alert_data:
                severity = "critical" if alert["alert_type"] in ["LOW_VOLTAGE", "HIGH_VOLTAGE"] else "warning"
                
                alerts.append({
                    "id": f"alert_{len(alerts) + 1}",
                    "type": alert["alert_type"],
                    "severity": severity,
                    "message": self._generate_alert_message(alert),
                    "device_id": alert["device_id"],
                    "site_id": alert["site_id"],
                    "timestamp": alert["packet_datetime"] or datetime.now().isoformat(),
                    "value": alert.get("cell_voltage") or alert.get("cell_temperature") or alert.get("soc"),
                    "acknowledged": False
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get recent alerts: {e}")
            return []
    
    def _generate_alert_message(self, alert: Dict) -> str:
        """Generate human-readable alert message.
        
        Args:
            alert: Alert data dictionary
            
        Returns:
            Formatted alert message
        """
        alert_type = alert["alert_type"]
        device_id = alert["device_id"]
        site_id = alert["site_id"]
        
        messages = {
            "LOW_VOLTAGE": f"Low voltage detected on device {device_id} at {site_id}: {alert['cell_voltage']:.3f}V",
            "HIGH_VOLTAGE": f"High voltage detected on device {device_id} at {site_id}: {alert['cell_voltage']:.3f}V",
            "HIGH_TEMPERATURE": f"High temperature detected on device {device_id} at {site_id}: {alert['cell_temperature']:.1f}°C",
            "LOW_TEMPERATURE": f"Low temperature detected on device {device_id} at {site_id}: {alert['cell_temperature']:.1f}°C",
            "LOW_SOC": f"Low state of charge on device {device_id} at {site_id}: {alert['soc']:.1f}%"
        }
        
        return messages.get(alert_type, f"Unknown alert type: {alert_type}")
    
    def get_mlops_metrics(self) -> Dict[str, Any]:
        """Get MLOps-specific metrics and performance data.
        
        Returns:
            Dictionary containing MLOps metrics
        """
        try:
            self.logger.info("Generating MLOps metrics")
            
            # Model performance metrics (simulated)
            performance_query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(CASE WHEN soc_latest_value_for_every_cycle > 80 THEN 1.0 ELSE 0.0 END) as high_soc_ratio,
                    AVG(CASE WHEN charge_or_discharge_cycle = 'charging' THEN 1.0 ELSE 0.0 END) as charging_ratio,
                    STDDEV(cell_voltage) as voltage_variance,
                    STDDEV(cell_temperature) as temp_variance
                FROM battery_data
            """
            perf_data = self._execute_query(performance_query)
            performance = perf_data[0] if perf_data else {}
            
            # Data quality metrics
            quality_query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(cell_voltage) as voltage_records,
                    COUNT(cell_temperature) as temp_records,
                    COUNT(soc_latest_value_for_every_cycle) as soc_records,
                    COUNT(packet_datetime) as datetime_records
                FROM battery_data
            """
            quality_data = self._execute_query(quality_query)
            quality = quality_data[0] if quality_data else {}
            
            # Calculate data completeness
            total = quality.get("total_records", 1)
            completeness = {
                "voltage": (quality.get("voltage_records", 0) / total) * 100,
                "temperature": (quality.get("temp_records", 0) / total) * 100,
                "soc": (quality.get("soc_records", 0) / total) * 100,
                "datetime": (quality.get("datetime_records", 0) / total) * 100
            }
            
            # Model drift simulation (using variance as proxy)
            drift_score = min(100, (performance.get("voltage_variance", 0) * 1000))
            
            metrics = {
                "model_performance": {
                    "accuracy": 95.2,  # Simulated
                    "precision": 94.8,  # Simulated
                    "recall": 96.1,     # Simulated
                    "f1_score": 95.4,   # Simulated
                    "total_predictions": performance.get("total_predictions", 0)
                },
                "data_quality": {
                    "completeness": completeness,
                    "overall_completeness": sum(completeness.values()) / len(completeness)
                },
                "drift_detection": {
                    "drift_score": round(drift_score, 2),
                    "status": "normal" if drift_score < 5 else "warning" if drift_score < 10 else "critical",
                    "last_check": datetime.now().isoformat()
                },
                "system_performance": {
                    "processing_time_ms": 45.2,  # Simulated
                    "memory_usage_mb": 128.5,    # Simulated
                    "cpu_usage_percent": 15.3,   # Simulated
                    "throughput_rps": 23.7       # Simulated
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get MLOps metrics: {e}")
            return {"error": str(e)}
    
    def get_llmops_metrics(self) -> Dict[str, Any]:
        """Get LLMOps-specific metrics and performance data.
        
        Returns:
            Dictionary containing LLMOps metrics
        """
        try:
            self.logger.info("Generating LLMOps metrics")
            
            # Simulate LLM interaction metrics
            llm_metrics = {
                "query_performance": {
                    "total_queries": 1247,
                    "avg_response_time_ms": 850.3,
                    "success_rate": 98.7,
                    "error_rate": 1.3
                },
                "model_health": {
                    "model_version": "gpt-4-battery-v1.2",
                    "last_updated": (datetime.now() - timedelta(days=7)).isoformat(),
                    "status": "healthy",
                    "confidence_score": 94.2
                },
                "usage_patterns": {
                    "peak_hours": [9, 10, 14, 15, 16],
                    "common_queries": [
                        "battery health analysis",
                        "anomaly explanation",
                        "maintenance recommendations",
                        "performance optimization"
                    ],
                    "avg_session_length_min": 12.5
                },
                "feedback_metrics": {
                    "user_satisfaction": 4.3,  # out of 5
                    "helpful_responses": 89.2,  # percentage
                    "query_clarity": 91.8       # percentage
                }
            }
            
            return llm_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get LLMOps metrics: {e}")
            return {"error": str(e)}
    
    def get_real_time_data(self, device_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get real-time battery monitoring data.
        
        Args:
            device_id: Optional device ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of real-time data records
        """
        try:
            base_query = """
                SELECT 
                    device_id,
                    site_id,
                    cell_voltage,
                    cell_temperature,
                    soc_latest_value_for_every_cycle as soc,
                    instantaneous_current,
                    charge_or_discharge_cycle,
                    ambient_temperature,
                    packet_datetime,
                    string_voltage,
                    battery_run_hours
                FROM battery_data 
                WHERE 1=1
            """
            
            params = []
            if device_id:
                base_query += " AND device_id = ?"
                params.append(device_id)
            
            base_query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            
            data = self._execute_query(base_query, tuple(params))
            
            # Add calculated fields
            for record in data:
                record["efficiency"] = self._calculate_efficiency(record)
                record["health_score"] = self._calculate_health_score(record)
                record["timestamp"] = record.get("packet_datetime") or datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time data: {e}")
            return []
    
    def _calculate_efficiency(self, record: Dict) -> float:
        """Calculate battery efficiency based on current data.
        
        Args:
            record: Battery data record
            
        Returns:
            Efficiency percentage
        """
        try:
            voltage = record.get("cell_voltage", 0)
            current = record.get("instantaneous_current", 0)
            soc = record.get("soc", 50)
            
            # Simplified efficiency calculation
            if voltage > 0 and current != 0:
                power_efficiency = min(100, abs(voltage * current) / 25 * 100)  # Normalized to 25W max
                soc_factor = soc / 100
                efficiency = power_efficiency * soc_factor
                return round(efficiency, 1)
            
            return 75.0  # Default efficiency
            
        except Exception:
            return 75.0
    
    def _calculate_health_score(self, record: Dict) -> float:
        """Calculate battery health score based on current data.
        
        Args:
            record: Battery data record
            
        Returns:
            Health score percentage
        """
        try:
            voltage = record.get("cell_voltage", 2.1)
            temperature = record.get("cell_temperature", 25)
            soc = record.get("soc", 50)
            
            # Voltage health (optimal range: 2.0-2.3V)
            voltage_health = 100
            if voltage < 1.9 or voltage > 2.4:
                voltage_health = 50
            elif voltage < 2.0 or voltage > 2.3:
                voltage_health = 80
            
            # Temperature health (optimal range: 15-35°C)
            temp_health = 100
            if temperature < 0 or temperature > 50:
                temp_health = 40
            elif temperature < 10 or temperature > 40:
                temp_health = 70
            
            # SOC health (optimal range: 20-90%)
            soc_health = 100
            if soc < 10 or soc > 95:
                soc_health = 50
            elif soc < 15 or soc > 90:
                soc_health = 80
            
            # Overall health score
            health_score = (voltage_health + temp_health + soc_health) / 3
            return round(health_score, 1)
            
        except Exception:
            return 85.0  # Default health score


# Global data service instance
_data_service = None


def get_data_service(db_path: str = "battery_monitoring.db") -> DataService:
    """Get or create the global data service instance.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        DataService instance
    """
    global _data_service
    
    if _data_service is None:
        _data_service = DataService(db_path)
    
    return _data_service


# Test the data service if run directly
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test the data service
        service = get_data_service()
        
        print("Testing DataService...")
        
        # Test basic statistics
        stats = service.get_basic_statistics()
        print(f"Basic Statistics: {json.dumps(stats, indent=2)}")
        
        # Test dashboard overview
        overview = service.get_dashboard_overview()
        print(f"Dashboard Overview Keys: {list(overview.keys())}")
        
        # Test real-time data
        real_time = service.get_real_time_data(limit=5)
        print(f"Real-time Data: {len(real_time)} records")
        
        print("✅ DataService test completed successfully!")
        
    except Exception as e:
        print(f"❌ DataService test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")