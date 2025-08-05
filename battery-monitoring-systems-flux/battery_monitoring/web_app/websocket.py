"""
WebSocket module for battery monitoring system.

This module provides real-time WebSocket communication for
live battery monitoring data and alerts.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol

from ..core.config import get_config
from ..core.logger import get_logger, get_performance_logger
from ..core.database import get_database_manager
from ..data.loader import DataLoader


class WebSocketManager:
    """
    WebSocket manager for real-time battery monitoring.
    
    Handles WebSocket connections, broadcasts real-time data,
    and manages client subscriptions.
    """
    
    def __init__(self, config=None):
        """Initialize the WebSocket manager."""
        self.config = config or get_config()
        self.logger = get_logger("websocket")
        self.performance_logger = get_performance_logger("websocket")
        
        # WebSocket configuration
        self.host = self.config.host
        self.port = self.config.websocket_port
        self.ping_interval = self.config.web_app.ping_interval
        self.ping_timeout = self.config.web_app.ping_timeout
        self.max_connections = self.config.web_app.max_connections
        
        # Connection management
        self.clients: Set[WebSocketServerProtocol] = set()
        self.client_subscriptions: Dict[WebSocketServerProtocol, Dict[str, Any]] = {}
        
        # Data components
        self.db_manager = get_database_manager()
        self.data_loader = DataLoader()
        
        # Background tasks
        self.broadcast_task = None
        self.monitoring_task = None
        
        self.logger.info(f"WebSocketManager initialized on {self.host}:{self.port}")
    
    async def start_server(self):
        """Start the WebSocket server."""
        try:
            self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            # Start background tasks
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start WebSocket server
            async with websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout
            ):
                self.logger.info("WebSocket server started successfully")
                await asyncio.Future()  # Run forever
                
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        try:
            self.logger.info("Stopping WebSocket server")
            
            # Cancel background tasks
            if self.broadcast_task:
                self.broadcast_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Close all client connections
            for client in list(self.clients):
                await self._close_client(client)
            
            self.logger.info("WebSocket server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")
    
    def stop(self):
        """Synchronous stop method for compatibility."""
        try:
            self.logger.info("Stopping WebSocket server (sync)")
            
            # Cancel background tasks
            if self.broadcast_task:
                self.broadcast_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            self.logger.info("WebSocket server stopped (sync)")
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connections."""
        try:
            # Check connection limit
            if len(self.clients) >= self.max_connections:
                await websocket.close(1013, "Maximum connections reached")
                return
            
            # Add client to set
            self.clients.add(websocket)
            self.client_subscriptions[websocket] = {
                'connected_at': datetime.now(),
                'subscriptions': set(),
                'last_activity': datetime.now()
            }
            
            client_id = id(websocket)
            self.logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
            
            # Send welcome message
            welcome_message = {
                'type': 'welcome',
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'message': 'Connected to Battery Monitoring WebSocket'
            }
            await websocket.send(json.dumps(welcome_message))
            
            # Handle client messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {id(websocket)} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {id(websocket)}: {e}")
        finally:
            await self._close_client(websocket)
    
    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            message_type = data.get('type', 'unknown')
            
            # Update last activity
            self.client_subscriptions[websocket]['last_activity'] = datetime.now()
            
            if message_type == 'subscribe':
                await self._handle_subscribe(websocket, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscribe(websocket, data)
            elif message_type == 'ping':
                await self._handle_ping(websocket)
            elif message_type == 'request_data':
                await self._handle_data_request(websocket, data)
            else:
                await self._send_error(websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            await self._send_error(websocket, str(e))
    
    async def _handle_subscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle subscription requests."""
        try:
            subscription_type = data.get('subscription_type', 'all')
            filters = data.get('filters', {})
            
            # Add subscription
            self.client_subscriptions[websocket]['subscriptions'].add(subscription_type)
            
            # Store filters
            if filters:
                self.client_subscriptions[websocket]['filters'] = filters
            
            # Send confirmation
            response = {
                'type': 'subscription_confirmed',
                'subscription_type': subscription_type,
                'filters': filters,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
            self.logger.info(f"Client {id(websocket)} subscribed to {subscription_type}")
            
        except Exception as e:
            await self._send_error(websocket, f"Error subscribing: {e}")
    
    async def _handle_unsubscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle unsubscription requests."""
        try:
            subscription_type = data.get('subscription_type', 'all')
            
            # Remove subscription
            self.client_subscriptions[websocket]['subscriptions'].discard(subscription_type)
            
            # Send confirmation
            response = {
                'type': 'unsubscription_confirmed',
                'subscription_type': subscription_type,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
            self.logger.info(f"Client {id(websocket)} unsubscribed from {subscription_type}")
            
        except Exception as e:
            await self._send_error(websocket, f"Error unsubscribing: {e}")
    
    async def _handle_ping(self, websocket: WebSocketServerProtocol):
        """Handle ping messages."""
        try:
            response = {
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
        except Exception as e:
            self.logger.error(f"Error handling ping: {e}")
    
    async def _handle_data_request(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle data requests."""
        try:
            request_type = data.get('request_type', 'latest')
            filters = data.get('filters', {})
            
            if request_type == 'latest':
                # Get latest battery data
                df = self.data_loader.load_from_database(
                    device_id=filters.get('device_id'),
                    cell_number=filters.get('cell_number'),
                    limit=100
                )
                
                response = {
                    'type': 'data_response',
                    'request_type': request_type,
                    'data': df.to_dict(orient='records'),
                    'total_records': len(df),
                    'timestamp': datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(response))
            else:
                await self._send_error(websocket, f"Unknown request type: {request_type}")
                
        except Exception as e:
            await self._send_error(websocket, f"Error handling data request: {e}")
    
    async def _close_client(self, websocket: WebSocketServerProtocol):
        """Close client connection and cleanup."""
        try:
            # Remove from clients set
            self.clients.discard(websocket)
            
            # Remove from subscriptions
            if websocket in self.client_subscriptions:
                del self.client_subscriptions[websocket]
            
            # Close connection
            if not websocket.closed:
                await websocket.close()
                
        except Exception as e:
            self.logger.error(f"Error closing client: {e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """Send error message to client."""
        try:
            error_response = {
                'type': 'error',
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            self.logger.error(f"Error sending error message: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any], subscription_type: str = 'all'):
        """Broadcast message to all subscribed clients."""
        try:
            message_json = json.dumps(message)
            disconnected_clients = []
            
            for client in self.clients:
                try:
                    # Check if client is subscribed to this message type
                    if (subscription_type == 'all' or 
                        subscription_type in self.client_subscriptions[client]['subscriptions']):
                        await client.send(message_json)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.append(client)
                except Exception as e:
                    self.logger.error(f"Error broadcasting to client {id(client)}: {e}")
                    disconnected_clients.append(client)
            
            # Clean up disconnected clients
            for client in disconnected_clients:
                await self._close_client(client)
                
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}")
    
    async def _broadcast_loop(self):
        """Background loop for broadcasting real-time data."""
        try:
            while True:
                try:
                    # Get latest data
                    df = self.data_loader.load_from_database(limit=50)
                    
                    if len(df) > 0:
                        # Prepare broadcast message
                        broadcast_data = {
                            'type': 'real_time_data',
                            'data': df.to_dict(orient='records'),
                            'total_records': len(df),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Broadcast to all clients
                        await self.broadcast_message(broadcast_data, 'real_time')
                    
                    # Wait before next broadcast
                    await asyncio.sleep(5)  # Broadcast every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in broadcast loop: {e}")
                    await asyncio.sleep(10)  # Wait longer on error
                    
        except asyncio.CancelledError:
            self.logger.info("Broadcast loop cancelled")
        except Exception as e:
            self.logger.error(f"Broadcast loop error: {e}")
    
    async def _monitoring_loop(self):
        """Background loop for system monitoring and alerts."""
        try:
            while True:
                try:
                    # Check system health
                    db_stats = self.db_manager.get_database_stats()
                    
                    # Check for anomalies or issues
                    if db_stats['total_records'] > 0:
                        # Send system status update
                        status_message = {
                            'type': 'system_status',
                            'database_stats': db_stats,
                            'connected_clients': len(self.clients),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        await self.broadcast_message(status_message, 'system')
                    
                    # Wait before next check
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            'total_clients': len(self.clients),
            'max_connections': self.max_connections,
            'connection_usage': len(self.clients) / self.max_connections,
            'subscriptions': {
                client_id: len(sub['subscriptions']) 
                for client_id, sub in self.client_subscriptions.items()
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the WebSocket manager."""
        return {
            'total_clients': len(self.clients),
            'max_connections': self.max_connections,
            'connection_usage': len(self.clients) / self.max_connections,
            'server_started': self.broadcast_task is not None,
            'last_activity': max(
                (sub['last_activity'] for sub in self.client_subscriptions.values()),
                default=datetime.now()
            )
        } 