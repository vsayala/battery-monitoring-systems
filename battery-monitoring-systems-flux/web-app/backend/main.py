#!/usr/bin/env python3
"""
Battery Monitoring System Web Application Backend
Main entry point for the web application server
"""

import asyncio
import uvicorn
import logging
from pathlib import Path
import sys

# Add the parent directory to the path to import battery_monitoring modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from battery_monitoring.core.logger import setup_logging, get_logger
from battery_monitoring.core.config import get_config
from api import app
from websocket import start_websocket_server

# Setup logging
setup_logging()
logger = get_logger(__name__)

async def main():
    """Main function to start both FastAPI and WebSocket servers"""
    try:
        config = get_config()
        
        logger.info("🚀 Starting Battery Monitoring System Web Application")
        logger.info(f"📊 API Server: http://localhost:{config.web.port}")
        logger.info(f"🔌 WebSocket Server: ws://localhost:{config.web.websocket_port}")
        logger.info(f"🌐 Frontend: http://localhost:{config.web.port}/static")
        
        # Start WebSocket server in a separate task
        websocket_task = asyncio.create_task(
            start_websocket_server(
                host=config.web.host,
                port=config.web.websocket_port
            )
        )
        
        # Start FastAPI server
        uvicorn_config = uvicorn.Config(
            app=app,
            host=config.web.host,
            port=config.web.port,
            log_level="info",
            reload=False
        )
        
        server = uvicorn.Server(uvicorn_config)
        
        # Run both servers concurrently
        await asyncio.gather(
            server.serve(),
            websocket_task
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Battery Monitoring System Web Application")
    except Exception as e:
        logger.error(f"❌ Error starting web application: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 