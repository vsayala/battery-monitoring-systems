#!/usr/bin/env python3
"""
Comprehensive Logging Utilities for Battery Monitoring System.

Provides decorators, error handling, resource monitoring, and structured logging
utilities for the entire project. Ensures all operations are logged with proper
error handling and performance metrics.
"""

import functools
import logging
import time
import traceback
import psutil
import sys
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from pathlib import Path
import json

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])

# Global logger instances
_loggers: Dict[str, logging.Logger] = {}
_performance_metrics: Dict[str, List[float]] = {}
_resource_monitor_active = False
_resource_thread: Optional[threading.Thread] = None


class LoggedError(Exception):
    """Custom exception that automatically logs error details."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        
        # Automatically log the error
        logger = get_module_logger("error_handler")
        logger.error(
            f"LoggedError [{self.error_code}]: {self.message}",
            extra={
                "error_code": self.error_code,
                "context": self.context,
                "timestamp": self.timestamp,
                "traceback": traceback.format_exc()
            }
        )


def get_module_logger(module_name: str) -> logging.Logger:
    """Get or create a logger for a specific module.
    
    Args:
        module_name: Name of the module requesting the logger
        
    Returns:
        Configured logger instance
    """
    if module_name not in _loggers:
        logger = logging.getLogger(f"battery_monitoring.{module_name}")
        
        # Set up basic configuration if not already configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        _loggers[module_name] = logger
    
    return _loggers[module_name]


def log_execution_time(func: F) -> F:
    """Decorator to log function execution time and basic info.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with execution time logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_module_logger(func.__module__ or "unknown")
        function_name = f"{func.__module__}.{func.__name__}" if func.__module__ else func.__name__
        
        start_time = time.time()
        logger.info(f"Starting execution: {function_name}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Store performance metric
            if function_name not in _performance_metrics:
                _performance_metrics[function_name] = []
            _performance_metrics[function_name].append(execution_time)
            
            logger.info(
                f"Completed execution: {function_name}",
                extra={
                    "execution_time": execution_time,
                    "status": "SUCCESS",
                    "function": function_name
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Failed execution: {function_name}",
                extra={
                    "execution_time": execution_time,
                    "status": "FAILED",
                    "function": function_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    return wrapper


def handle_exceptions(
    reraise: bool = True,
    default_return: Any = None,
    log_level: str = "ERROR"
) -> Callable[[F], F]:
    """Decorator to handle exceptions with comprehensive logging.
    
    Args:
        reraise: Whether to reraise the exception after logging
        default_return: Default value to return if exception occurs and reraise=False
        log_level: Logging level for exceptions
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_module_logger(func.__module__ or "unknown")
            function_name = f"{func.__module__}.{func.__name__}" if func.__module__ else func.__name__
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                error_info = {
                    "function": function_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "args": str(args)[:200],  # Limit length
                    "kwargs": str(kwargs)[:200],  # Limit length
                    "timestamp": datetime.now().isoformat()
                }
                
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(
                    f"Exception in {function_name}: {e}",
                    extra=error_info
                )
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


def log_resource_usage(func: F) -> F:
    """Decorator to log CPU and memory usage during function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with resource usage logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_module_logger(func.__module__ or "unknown")
        function_name = f"{func.__module__}.{func.__name__}" if func.__module__ else func.__name__
        
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Get final resource usage
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            
            memory_delta = final_memory - initial_memory
            
            logger.info(
                f"Resource usage for {function_name}",
                extra={
                    "function": function_name,
                    "execution_time": execution_time,
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_delta_mb": memory_delta,
                    "initial_cpu_percent": initial_cpu,
                    "final_cpu_percent": final_cpu,
                    "status": "SUCCESS"
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            logger.error(
                f"Resource usage for {function_name} (FAILED)",
                extra={
                    "function": function_name,
                    "execution_time": execution_time,
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_delta_mb": memory_delta,
                    "error": str(e),
                    "status": "FAILED"
                }
            )
            raise
    
    return wrapper


def comprehensive_logging(
    log_execution: bool = True,
    log_resources: bool = False,
    handle_errors: bool = True,
    reraise_errors: bool = True
) -> Callable[[F], F]:
    """Comprehensive logging decorator combining multiple logging features.
    
    Args:
        log_execution: Whether to log execution time
        log_resources: Whether to log resource usage
        handle_errors: Whether to handle and log errors
        reraise_errors: Whether to reraise errors after logging
        
    Returns:
        Comprehensive decorator function
    """
    def decorator(func: F) -> F:
        # Apply decorators in reverse order
        decorated_func = func
        
        if handle_errors:
            decorated_func = handle_exceptions(reraise=reraise_errors)(decorated_func)
        
        if log_resources:
            decorated_func = log_resource_usage(decorated_func)
        
        if log_execution:
            decorated_func = log_execution_time(decorated_func)
        
        return decorated_func
    
    return decorator


def start_resource_monitoring(interval: float = 60.0) -> None:
    """Start background resource monitoring.
    
    Args:
        interval: Monitoring interval in seconds
    """
    global _resource_monitor_active, _resource_thread
    
    if _resource_monitor_active:
        return
    
    _resource_monitor_active = True
    
    def monitor_resources():
        logger = get_module_logger("resource_monitor")
        
        while _resource_monitor_active:
            try:
                # System resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Process resources
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
                process_cpu = process.cpu_percent()
                
                logger.info(
                    "System resource monitoring",
                    extra={
                        "system_cpu_percent": cpu_percent,
                        "system_memory_percent": memory.percent,
                        "system_memory_available_gb": memory.available / 1024 / 1024 / 1024,
                        "disk_usage_percent": disk.percent,
                        "disk_free_gb": disk.free / 1024 / 1024 / 1024,
                        "process_memory_mb": process_memory,
                        "process_cpu_percent": process_cpu,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
    
    _resource_thread = threading.Thread(target=monitor_resources, daemon=True)
    _resource_thread.start()
    
    logger = get_module_logger("resource_monitor")
    logger.info("Started background resource monitoring")


def stop_resource_monitoring() -> None:
    """Stop background resource monitoring."""
    global _resource_monitor_active
    
    _resource_monitor_active = False
    logger = get_module_logger("resource_monitor")
    logger.info("Stopped background resource monitoring")


def get_performance_summary() -> Dict[str, Dict[str, float]]:
    """Get summary of performance metrics.
    
    Returns:
        Dictionary with performance statistics for each function
    """
    summary = {}
    
    for function_name, times in _performance_metrics.items():
        if times:
            summary[function_name] = {
                "call_count": len(times),
                "total_time": sum(times),
                "average_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
    
    return summary


def log_performance_summary() -> None:
    """Log a summary of all performance metrics."""
    logger = get_module_logger("performance_summary")
    summary = get_performance_summary()
    
    if summary:
        logger.info(
            "Performance summary",
            extra={
                "performance_metrics": summary,
                "timestamp": datetime.now().isoformat()
            }
        )
    else:
        logger.info("No performance metrics available")


def log_system_info() -> None:
    """Log comprehensive system information."""
    logger = get_module_logger("system_info")
    
    try:
        # System information
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        # Python information
        python_version = sys.version
        
        system_info = {
            "cpu_count": cpu_count,
            "total_memory_gb": memory.total / 1024 / 1024 / 1024,
            "total_disk_gb": disk.total / 1024 / 1024 / 1024,
            "system_boot_time": boot_time.isoformat(),
            "python_version": python_version,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            "System information",
            extra=system_info
        )
        
    except Exception as e:
        logger.error(f"Failed to log system information: {e}")


def create_operation_context(operation_name: str) -> Dict[str, Any]:
    """Create a context dictionary for logging operations.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        Context dictionary with operation metadata
    """
    return {
        "operation": operation_name,
        "timestamp": datetime.now().isoformat(),
        "thread_id": threading.current_thread().ident,
        "process_id": os.getpid()
    }


class LogContext:
    """Context manager for logging operations with automatic success/failure logging."""
    
    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        log_level: str = "INFO"
    ):
        self.operation_name = operation_name
        self.logger = logger or get_module_logger("operation_context")
        self.log_level = log_level.upper()
        self.start_time = None
        self.context = create_operation_context(operation_name)
    
    def __enter__(self):
        self.start_time = time.time()
        log_method = getattr(self.logger, self.log_level.lower())
        log_method(
            f"Starting operation: {self.operation_name}",
            extra=self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        self.context["execution_time"] = execution_time
        
        if exc_type is None:
            # Success
            self.context["status"] = "SUCCESS"
            log_method = getattr(self.logger, self.log_level.lower())
            log_method(
                f"Completed operation: {self.operation_name}",
                extra=self.context
            )
        else:
            # Failure
            self.context.update({
                "status": "FAILED",
                "error_type": exc_type.__name__,
                "error_message": str(exc_val),
                "traceback": traceback.format_exc()
            })
            self.logger.error(
                f"Failed operation: {self.operation_name}",
                extra=self.context
            )
        
        return False  # Don't suppress exceptions


# Convenience functions for common logging patterns
def log_info(message: str, module: str = "general", **kwargs):
    """Log an info message with optional context."""
    logger = get_module_logger(module)
    logger.info(message, extra=kwargs)


def log_warning(message: str, module: str = "general", **kwargs):
    """Log a warning message with optional context."""
    logger = get_module_logger(module)
    logger.warning(message, extra=kwargs)


def log_error(message: str, module: str = "general", **kwargs):
    """Log an error message with optional context."""
    logger = get_module_logger(module)
    logger.error(message, extra=kwargs)


def log_debug(message: str, module: str = "general", **kwargs):
    """Log a debug message with optional context."""
    logger = get_module_logger(module)
    logger.debug(message, extra=kwargs)


# Initialize system logging
import os
os.makedirs("logs", exist_ok=True)
log_system_info()