"""
Battery Monitoring System with ML/LLM and MLOps

A comprehensive battery monitoring system that provides:
- Anomaly detection for cell voltage, temperature, and specific gravity
- Cell health prediction (dead/alive) with confidence scores
- Future value forecasting for battery parameters
- MLOps with continuous monitoring and deployment
- LLM-powered chatbot for data analysis
- Real-time web application with WebSocket support
"""

__version__ = "1.0.0"
__author__ = "Battery Monitoring Team"
__email__ = "team@battery-monitoring.com"

from .core.config import Config
from .core.logger import setup_logging
from .core.database import DatabaseManager

__all__ = [
    "Config",
    "setup_logging", 
    "DatabaseManager",
] 