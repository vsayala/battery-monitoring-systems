"""
Custom exceptions for battery monitoring system.

Defines specific exception classes for different types of errors that can occur
in the battery monitoring system.
"""

from typing import Any, Dict, Optional


class BatteryMonitoringError(Exception):
    """Base exception for battery monitoring system."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(BatteryMonitoringError):
    """Raised when there's an error in configuration."""
    pass


class DataError(BatteryMonitoringError):
    """Raised when there's an error with data processing."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataNotFoundError(DataError):
    """Raised when required data is not found."""
    pass


class ModelError(BatteryMonitoringError):
    """Raised when there's an error with ML models."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    pass


class LLMError(BatteryMonitoringError):
    """Raised when there's an error with LLM operations."""
    pass


class LLMConnectionError(LLMError):
    """Raised when LLM connection fails."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid."""
    pass


class DatabaseError(BatteryMonitoringError):
    """Raised when there's an error with database operations."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails."""
    pass


class WebAppError(BatteryMonitoringError):
    """Raised when there's an error with web application."""
    pass


class APIError(WebAppError):
    """Raised when there's an error with API operations."""
    pass


class WebSocketError(WebAppError):
    """Raised when there's an error with WebSocket operations."""
    pass


class MLOpsError(BatteryMonitoringError):
    """Raised when there's an error with MLOps operations."""
    pass


class MonitoringError(MLOpsError):
    """Raised when there's an error with monitoring operations."""
    pass


class AlertingError(MLOpsError):
    """Raised when there's an error with alerting operations."""
    pass


class LoggingError(BatteryMonitoringError):
    """Raised when there's an error with logging operations."""
    pass


class SecurityError(BatteryMonitoringError):
    """Raised when there's a security-related error."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass


class PerformanceError(BatteryMonitoringError):
    """Raised when there's a performance-related error."""
    pass


class TimeoutError(PerformanceError):
    """Raised when an operation times out."""
    pass


class ResourceError(PerformanceError):
    """Raised when there's insufficient resources."""
    pass


def handle_exception(func):
    """Decorator to handle exceptions and log them properly."""
    import functools
    import logging
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BatteryMonitoringError as e:
            logging.error(f"Battery monitoring error in {func.__name__}: {e.message}", extra=e.details)
            raise
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise BatteryMonitoringError(f"Unexpected error: {str(e)}")
    
    return wrapper 