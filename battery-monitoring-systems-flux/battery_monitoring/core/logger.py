"""
Logging configuration for battery monitoring system.

Provides centralized logging setup with master log and module-specific logs.
Includes structured logging, log rotation, and performance monitoring.
"""

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from .config import get_config
from .exceptions import LoggingError


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.info(f"Started operation: {operation}")
    
    def end_timer(self, operation: str, success: bool = True) -> float:
        """End timing an operation and log the duration."""
        if operation not in self.start_times:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        status = "SUCCESS" if success else "FAILED"
        
        self.logger.info(
            f"Completed operation: {operation}",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                "status": status,
                "timestamp": time.time()
            }
        )
        
        del self.start_times[operation]
        return duration
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log a performance metric."""
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "timestamp": time.time()
            }
        )


class MasterLogger:
    """Master logger that aggregates logs from all modules."""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_master_logger()
        self.performance_logger = PerformanceLogger(self.logger)
    
    def _setup_master_logger(self) -> logging.Logger:
        """Setup the master logger."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path(self.config.logging.file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logger
            logger = logging.getLogger("battery_monitoring.master")
            logger.setLevel(getattr(logging, self.config.logging.level.upper()))
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # Create formatter
            formatter = logging.Formatter(self.config.logging.format)
            
            # File handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.logging.file,
                maxBytes=self.config.logging.max_size,
                backupCount=self.config.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Console handler for development
            if self.config.debug:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # Prevent propagation to root logger
            logger.propagate = False
            
            return logger
        
        except Exception as e:
            raise LoggingError(f"Failed to setup master logger: {str(e)}")
    
    def log_system_start(self) -> None:
        """Log system startup."""
        self.logger.info(
            "Battery Monitoring System Starting",
            extra={
                "event": "system_start",
                "version": self.config.app_version,
                "timestamp": time.time()
            }
        )
    
    def log_system_stop(self) -> None:
        """Log system shutdown."""
        self.logger.info(
            "Battery Monitoring System Stopping",
            extra={
                "event": "system_stop",
                "timestamp": time.time()
            }
        )
    
    def log_module_start(self, module_name: str) -> None:
        """Log module startup."""
        self.logger.info(
            f"Module {module_name} starting",
            extra={
                "event": "module_start",
                "module": module_name,
                "timestamp": time.time()
            }
        )
    
    def log_module_stop(self, module_name: str) -> None:
        """Log module shutdown."""
        self.logger.info(
            f"Module {module_name} stopping",
            extra={
                "event": "module_stop",
                "module": module_name,
                "timestamp": time.time()
            }
        )
    
    def log_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Log an error with context."""
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={
                "event": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "timestamp": time.time()
            },
            exc_info=True
        )


class ModuleLogger:
    """Logger for specific modules."""
    
    def __init__(self, module_name: str, config):
        self.module_name = module_name
        self.config = config
        self.logger = self._setup_module_logger()
        self.performance_logger = PerformanceLogger(self.logger)
    
    def _setup_module_logger(self) -> logging.Logger:
        """Setup module-specific logger."""
        try:
            # Get module log file path
            log_file_attr = f"{self.module_name.lower().replace(' ', '_')}_log"
            log_file = getattr(self.config.logging, log_file_attr, None)
            
            if not log_file:
                # Fallback to master log
                return logging.getLogger("battery_monitoring.master")
            
            # Create logs directory if it doesn't exist
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logger
            logger = logging.getLogger(f"battery_monitoring.{self.module_name}")
            logger.setLevel(getattr(logging, self.config.logging.level.upper()))
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # Create formatter
            formatter = logging.Formatter(self.config.logging.format)
            
            # File handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.logging.max_size,
                backupCount=self.config.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Console handler for development
            if self.config.debug:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # Prevent propagation to root logger
            logger.propagate = False
            
            return logger
        
        except Exception as e:
            raise LoggingError(f"Failed to setup module logger for {self.module_name}: {str(e)}")
    
    def log_operation_start(self, operation: str, **kwargs) -> None:
        """Log the start of an operation."""
        self.logger.info(
            f"Starting operation: {operation}",
            extra={
                "event": "operation_start",
                "operation": operation,
                "module": self.module_name,
                "timestamp": time.time(),
                **kwargs
            }
        )
    
    def log_operation_end(self, operation: str, success: bool = True, **kwargs) -> None:
        """Log the end of an operation."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"Completed operation: {operation} - {status}",
            extra={
                "event": "operation_end",
                "operation": operation,
                "module": self.module_name,
                "status": status,
                "timestamp": time.time(),
                **kwargs
            }
        )
    
    def log_data_processing(self, data_info: Dict) -> None:
        """Log data processing information."""
        self.logger.info(
            "Data processing completed",
            extra={
                "event": "data_processing",
                "module": self.module_name,
                "timestamp": time.time(),
                **data_info
            }
        )
    
    def log_model_operation(self, operation: str, model_info: Dict) -> None:
        """Log model-related operations."""
        self.logger.info(
            f"Model operation: {operation}",
            extra={
                "event": "model_operation",
                "operation": operation,
                "module": self.module_name,
                "timestamp": time.time(),
                **model_info
            }
        )


# Global logger instances
_master_logger: Optional[MasterLogger] = None
_module_loggers: Dict[str, ModuleLogger] = {}


def setup_logging() -> MasterLogger:
    """Setup the master logging system."""
    global _master_logger
    
    try:
        config = get_config()
        _master_logger = MasterLogger(config)
        
        # Configure structlog for structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Log system startup
        _master_logger.log_system_start()
        
        return _master_logger
    
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.error(f"Failed to setup logging: {str(e)}")
        raise LoggingError(f"Failed to setup logging: {str(e)}")


def get_logger(module_name: str = "master") -> logging.Logger:
    """Get a logger for a specific module."""
    global _master_logger, _module_loggers
    
    if _master_logger is None:
        _master_logger = setup_logging()
    
    if module_name == "master":
        return _master_logger.logger
    
    if module_name not in _module_loggers:
        config = get_config()
        _module_loggers[module_name] = ModuleLogger(module_name, config)
    
    return _module_loggers[module_name].logger


def get_performance_logger(module_name: str = "master") -> PerformanceLogger:
    """Get a performance logger for a specific module."""
    global _master_logger, _module_loggers
    
    if _master_logger is None:
        _master_logger = setup_logging()
    
    if module_name == "master":
        return _master_logger.performance_logger
    
    if module_name not in _module_loggers:
        config = get_config()
        _module_loggers[module_name] = ModuleLogger(module_name, config)
    
    return _module_loggers[module_name].performance_logger


def log_system_metrics() -> None:
    """Log system metrics."""
    import psutil
    
    try:
        logger = get_logger("master")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        logger.info(
            "System metrics",
            extra={
                "event": "system_metrics",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "timestamp": time.time()
            }
        )
    
    except Exception as e:
        logger = get_logger("master")
        logger.error(f"Failed to log system metrics: {str(e)}")


def cleanup_logging() -> None:
    """Cleanup logging resources."""
    global _master_logger, _module_loggers
    
    if _master_logger:
        _master_logger.log_system_stop()
    
    # Close all handlers
    for logger in [_master_logger] + list(_module_loggers.values()):
        if logger:
            for handler in logger.logger.handlers:
                handler.close()
                logger.logger.removeHandler(handler)
    
    _master_logger = None
    _module_loggers.clear() 