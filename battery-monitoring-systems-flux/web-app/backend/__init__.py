"""
Web Application module for battery monitoring system.

This module provides web-based interface and API capabilities
including REST API, WebSocket, and frontend components.
"""

from .api import create_app
from .websocket import WebSocketManager

__all__ = ["create_app", "WebSocketManager"] 