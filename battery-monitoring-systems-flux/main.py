#!/usr/bin/env python3
"""
Main entry point for Battery Monitoring System with ML/LLM and MLOps.

This script provides a comprehensive battery monitoring system with:
- Anomaly detection for cell voltage, temperature, and specific gravity
- Cell health prediction (dead/alive) with confidence scores
- Future value forecasting for battery parameters
- MLOps with continuous monitoring and deployment
- LLM-powered chatbot for data analysis
- Real-time web application with WebSocket support
"""

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import click
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from battery_monitoring.core.config import get_config, set_config
from battery_monitoring.core.logger import setup_logging, get_logger, cleanup_logging
from battery_monitoring.core.database import get_database_manager, close_database_manager
from battery_monitoring.data.loader import DataLoader
from battery_monitoring.ml.anomaly_detector import AnomalyDetector
from battery_monitoring.ml.cell_predictor import CellPredictor
from battery_monitoring.ml.forecaster import Forecaster
from battery_monitoring.llm.chatbot import BatteryChatbot
# Web application imports - now in web-app/backend/
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "web-app" / "backend"))
from api import app
from websocket import WebSocketManager
from battery_monitoring.mlops.monitor import MLOpsMonitor

# Rich console for beautiful output
console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Battery Monitoring System                 ║
    ║                     with ML/LLM & MLOps                     ║
    ║                                                              ║
    ║  Features:                                                   ║
    ║  • Anomaly Detection (Voltage, Temperature, Specific Gravity)║
    ║  • Cell Health Prediction (Dead/Alive with Confidence)      ║
    ║  • Future Value Forecasting (25 steps ahead)                ║
    ║  • MLOps with Continuous Monitoring & Deployment            ║
    ║  • LLM-powered Chatbot for Data Analysis                    ║
    ║  • Real-time Web Application with WebSocket                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def print_system_info(config):
    """Print system information."""
    table = Table(title="System Configuration")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Application Name", config.app_name)
    table.add_row("Version", config.app_version)
    table.add_row("Debug Mode", str(config.debug))
    table.add_row("Host", config.host)
    table.add_row("Port", str(config.port))
    table.add_row("WebSocket Port", str(config.websocket_port))
    table.add_row("Database Type", config.database.type)
    table.add_row("LLM Provider", config.llm.provider)
    table.add_row("LLM Model", config.llm.model)
    
    console.print(table)


def setup_system(config_path: Optional[str] = None) -> tuple:
    """Setup the battery monitoring system."""
    console.print("\n[bold green]🚀 Setting up Battery Monitoring System...[/bold green]")
    
    try:
        # Load configuration
        if config_path:
            config = get_config().load_from_yaml(config_path)
            set_config(config)
        else:
            config = get_config()
        
        # Setup logging
        master_logger = setup_logging()
        logger = get_logger("main")
        
        # Setup database
        db_manager = get_database_manager()
        
        # Print system info
        print_system_info(config)
        
        console.print("[bold green]✅ System setup completed successfully![/bold green]")
        
        return config, logger, db_manager, master_logger
    
    except Exception as e:
        console.print(f"[bold red]❌ Failed to setup system: {str(e)}[/bold red]")
        sys.exit(1)


def initialize_components(config, logger):
    """Initialize all system components."""
    console.print("\n[bold blue]🔧 Initializing system components...[/bold blue]")
    
    components = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Data loader
        task = progress.add_task("Initializing data loader...", total=None)
        try:
            components['data_loader'] = DataLoader(config)
            progress.update(task, description="✅ Data loader initialized")
        except Exception as e:
            progress.update(task, description=f"❌ Data loader failed: {str(e)}")
            raise
        
        # Anomaly detector
        task = progress.add_task("Initializing anomaly detector...", total=None)
        try:
            components['anomaly_detector'] = AnomalyDetector(config)
            progress.update(task, description="✅ Anomaly detector initialized")
        except Exception as e:
            progress.update(task, description=f"❌ Anomaly detector failed: {str(e)}")
            raise
        
        # Cell predictor
        task = progress.add_task("Initializing cell predictor...", total=None)
        try:
            components['cell_predictor'] = CellPredictor(config)
            progress.update(task, description="✅ Cell predictor initialized")
        except Exception as e:
            progress.update(task, description=f"❌ Cell predictor failed: {str(e)}")
            raise
        
        # Battery forecaster
        task = progress.add_task("Initializing battery forecaster...", total=None)
        try:
            components['forecaster'] = Forecaster(config)
            progress.update(task, description="✅ Battery forecaster initialized")
        except Exception as e:
            progress.update(task, description=f"❌ Battery forecaster failed: {str(e)}")
            raise
        
        # LLM chatbot
        task = progress.add_task("Initializing LLM chatbot...", total=None)
        try:
            components['chatbot'] = BatteryChatbot(config)
            progress.update(task, description="✅ LLM chatbot initialized")
        except Exception as e:
            progress.update(task, description=f"❌ LLM chatbot failed: {str(e)}")
            raise
        
        # MLOps monitor
        task = progress.add_task("Initializing MLOps monitor...", total=None)
        try:
            components['mlops_monitor'] = MLOpsMonitor(config)
            progress.update(task, description="✅ MLOps monitor initialized")
        except Exception as e:
            progress.update(task, description=f"❌ MLOps monitor failed: {str(e)}")
            raise
        
        # WebSocket manager
        task = progress.add_task("Initializing WebSocket manager...", total=None)
        try:
            components['websocket_manager'] = WebSocketManager(config)
            progress.update(task, description="✅ WebSocket manager initialized")
        except Exception as e:
            progress.update(task, description=f"❌ WebSocket manager failed: {str(e)}")
            raise
    
    console.print("[bold green]✅ All components initialized successfully![/bold green]")
    return components


def run_data_pipeline(components, logger):
    """Run the data processing pipeline."""
    console.print("\n[bold blue]📊 Running data processing pipeline...[/bold blue]")
    
    try:
        data_loader = components['data_loader']
        
        # Load data
        logger.info("Loading battery monitoring data")
        df = data_loader.load_excel_data()
        
        # Get data summary
        summary = data_loader.get_data_summary(df)
        logger.info(f"Data summary: {summary}")
        
        # Save to database
        logger.info("Saving data to database")
        inserted_count = data_loader.save_to_database(df)
        logger.info(f"Inserted {inserted_count} records to database")
        
        console.print(f"[bold green]✅ Data pipeline completed! Processed {len(df)} records[/bold green]")
        return df
    
    except Exception as e:
        logger.error(f"Data pipeline failed: {str(e)}")
        console.print(f"[bold red]❌ Data pipeline failed: {str(e)}[/bold red]")
        raise


def run_ml_pipeline(components, df, logger):
    """Run the machine learning pipeline."""
    console.print("\n[bold blue]🤖 Running machine learning pipeline...[/bold blue]")
    
    try:
        # Train models first
        logger.info("Training ML models...")
        
        # Train anomaly detection
        anomaly_detector = components['anomaly_detector']
        anomaly_training_results = anomaly_detector.train(df)
        logger.info(f"Anomaly detection model trained: {anomaly_training_results['anomaly_count']} anomalies detected")
        
        # Train cell prediction
        cell_predictor = components['cell_predictor']
        cell_training_results = cell_predictor.train(df)
        logger.info(f"Cell prediction model trained: accuracy={cell_training_results['accuracy']:.3f}")
        
        # Train forecasting
        forecaster = components['forecaster']
        forecast_training_results = forecaster.train(df)
        logger.info(f"Forecasting models trained: {len(forecast_training_results)} models")
        
        # Save trained models
        anomaly_detector.save_model()
        cell_predictor.save_model()
        forecaster.save_model()
        logger.info("All models saved successfully")
        
        # Now run inference on a sample
        sample_df = df.head(100)  # Use a sample for inference
        
        # Anomaly detection
        logger.info("Running anomaly detection")
        anomaly_results = anomaly_detector.detect_anomalies(sample_df)
        logger.info(f"Detected {anomaly_results['is_anomaly'].sum()} anomalies in sample")
        
        # Cell prediction
        logger.info("Running cell health prediction")
        prediction_results = cell_predictor.predict(sample_df)
        logger.info(f"Generated {len(prediction_results)} predictions")
        
        # Forecasting
        logger.info("Running battery forecasting")
        forecast_results = forecaster.forecast(sample_df)
        logger.info(f"Generated forecasts for {len(forecast_results)} parameters")
        
        console.print("[bold green]✅ Machine learning pipeline completed![/bold green]")
        return anomaly_results, prediction_results, forecast_results
    
    except Exception as e:
        logger.error(f"ML pipeline failed: {str(e)}")
        console.print(f"[bold red]❌ ML pipeline failed: {str(e)}[/bold red]")
        raise


def start_web_application(config, components, logger):
    """Start the web application."""
    console.print("\n[bold blue]🌐 Starting web application...[/bold blue]")
    
    try:
        # Start WebSocket server in background
        import asyncio
        import threading
        
        def run_websocket_server():
            """Run WebSocket server in a separate thread."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Import here to avoid circular imports
                sys.path.append(str(Path(__file__).parent / "web-app" / "backend"))
                from websocket import start_websocket_server
                
                loop.run_until_complete(start_websocket_server(config.host, config.websocket_port))
            except Exception as e:
                logger.error(f"WebSocket server error: {e}")
        
        # Start WebSocket server in background thread
        websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
        websocket_thread.start()
        
        # Start the FastAPI server
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=False,  # Disable reload for now
            log_level="info" if config.debug else "warning"
        )
    
    except Exception as e:
        logger.error(f"Failed to start web application: {str(e)}")
        console.print(f"[bold red]❌ Failed to start web application: {str(e)}[/bold red]")
        raise


def cleanup_system(components, logger):
    """Cleanup system resources."""
    console.print("\n[bold yellow]🧹 Cleaning up system resources...[/bold yellow]")
    
    try:
        # Stop MLOps monitoring
        if 'mlops_monitor' in components:
            components['mlops_monitor'].stop_monitoring_sync()
        
        # Stop WebSocket manager
        if 'websocket_manager' in components:
            components['websocket_manager'].stop()
        
        # Close database connections
        close_database_manager()
        
        # Cleanup logging
        cleanup_logging()
        
        console.print("[bold green]✅ System cleanup completed![/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]❌ Cleanup failed: {str(e)}[/bold red]")


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    console.print(f"\n[bold yellow]⚠️  Received signal {signum}, shutting down gracefully...[/bold yellow]")
    sys.exit(0)


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Battery Monitoring System CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.command()
@click.option('--generate-data', is_flag=True, help='Generate synthetic data for testing')
@click.option('--sample-size', default=100, help='Sample size for data loading')
@click.pass_context
def setup(ctx, generate_data, sample_size):
    """Setup and initialize the battery monitoring system."""
    print_banner()
    
    config, logger, db_manager, master_logger = setup_system(ctx.obj['config_path'])
    
    try:
        # Initialize components
        components = initialize_components(config, logger)
        
        # Load or generate data
        if generate_data:
            console.print("\n[bold blue]🔧 Generating synthetic data...[/bold blue]")
            data_loader = components['data_loader']
            df = data_loader.generate_synthetic_data()
            data_loader.save_to_database(df)
            console.print(f"[bold green]✅ Generated and saved {len(df)} synthetic records[/bold green]")
        else:
            # Load sample data
            console.print(f"\n[bold blue]📊 Loading sample data (size: {sample_size})...[/bold blue]")
            data_loader = components['data_loader']
            df = data_loader.load_sample_data(sample_size)
            console.print(f"[bold green]✅ Loaded {len(df)} sample records[/bold green]")
        
        # Run ML pipeline
        run_ml_pipeline(components, df, logger)
        
        console.print("\n[bold green]🎉 System setup completed successfully![/bold green]")
        console.print("\n[bold blue]Next steps:[/bold blue]")
        console.print("1. Run 'python main.py start' to start the web application")
        console.print("2. Open http://localhost:3000 in your browser")
        console.print("3. Use the API at http://localhost:8000/docs")
    
    except Exception as e:
        console.print(f"[bold red]❌ Setup failed: {str(e)}[/bold red]")
        cleanup_system(components, logger)
        sys.exit(1)


@cli.command()
@click.pass_context
def start(ctx):
    """Start the battery monitoring system."""
    print_banner()
    
    config, logger, db_manager, master_logger = setup_system(ctx.obj['config_path'])
    
    try:
        # Initialize components
        components = initialize_components(config, logger)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        console.print("\n[bold green]🚀 Starting Battery Monitoring System...[/bold green]")
        console.print(f"[bold blue]Web Application:[/bold blue] http://localhost:{config.port}")
        console.print(f"[bold blue]API Documentation:[/bold blue] http://localhost:{config.port}/docs")
        console.print(f"[bold blue]WebSocket:[/bold blue] ws://localhost:{config.websocket_port}")
        console.print("\n[bold yellow]Press Ctrl+C to stop the system[/bold yellow]")
        
        # Start web application
        start_web_application(config, components, logger)
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️  Received interrupt signal, shutting down...[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to start system: {str(e)}[/bold red]")
    finally:
        cleanup_system(components, logger)


@cli.command()
@click.option('--device-id', type=int, help='Device ID to analyze')
@click.option('--cell-number', type=int, help='Cell number to analyze')
@click.pass_context
def analyze(ctx, device_id, cell_number):
    """Run analysis on battery data."""
    print_banner()
    
    config, logger, db_manager, master_logger = setup_system(ctx.obj['config_path'])
    
    try:
        # Initialize components
        components = initialize_components(config, logger)
        
        # Load data
        data_loader = components['data_loader']
        df = data_loader.load_from_database(device_id=device_id, cell_number=cell_number)
        
        if df.empty:
            console.print("[bold red]❌ No data found for the specified criteria[/bold red]")
            return
        
        # Run analysis
        console.print(f"\n[bold blue]📊 Analyzing {len(df)} records...[/bold blue]")
        
        # Get data summary
        summary = data_loader.get_data_summary(df)
        console.print(f"[bold green]✅ Analysis completed![/bold green]")
        
        # Display summary
        table = Table(title="Data Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Records", str(summary['total_rows']))
        table.add_row("Total Columns", str(summary['total_columns']))
        table.add_row("Memory Usage (MB)", f"{summary['memory_usage_mb']:.2f}")
        
        if 'CellVoltage_stats' in summary:
            stats = summary['CellVoltage_stats']
            table.add_row("Voltage Mean", f"{stats['mean']:.3f}")
            table.add_row("Voltage Std", f"{stats['std']:.3f}")
        
        if 'CellTemperature_stats' in summary:
            stats = summary['CellTemperature_stats']
            table.add_row("Temperature Mean", f"{stats['mean']:.2f}")
            table.add_row("Temperature Std", f"{stats['std']:.2f}")
        
        if 'CellSpecificGravity_stats' in summary:
            stats = summary['CellSpecificGravity_stats']
            table.add_row("Specific Gravity Mean", f"{stats['mean']:.4f}")
            table.add_row("Specific Gravity Std", f"{stats['std']:.4f}")
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[bold red]❌ Analysis failed: {str(e)}[/bold red]")
    finally:
        cleanup_system(components, logger)


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status."""
    print_banner()
    
    try:
        config, logger, db_manager, master_logger = setup_system(ctx.obj['config_path'])
        
        # Get database stats
        stats = db_manager.get_database_stats()
        
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")
        
        table.add_row("Database", "✅ Connected", f"{stats['battery_data_count']} records")
        table.add_row("Anomaly Detection", "✅ Ready", f"{stats['anomaly_detection_count']} detections")
        table.add_row("Cell Predictions", "✅ Ready", f"{stats['cell_prediction_count']} predictions")
        table.add_row("Forecasting", "✅ Ready", f"{stats['forecasting_count']} forecasts")
        table.add_row("Models", "✅ Ready", f"{stats['model_metadata_count']} models")
        table.add_row("System Metrics", "✅ Active", f"{stats['system_metrics_count']} metrics")
        
        if stats['latest_battery_data']:
            table.add_row("Latest Data", "✅ Available", stats['latest_battery_data'])
        else:
            table.add_row("Latest Data", "❌ None", "No data available")
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[bold red]❌ Failed to get status: {str(e)}[/bold red]")


@cli.command()
@click.option('--days', default=30, help='Number of days to keep')
@click.pass_context
def cleanup(ctx, days):
    """Clean up old data."""
    print_banner()
    
    config, logger, db_manager, master_logger = setup_system(ctx.obj['config_path'])
    
    try:
        console.print(f"\n[bold blue]🧹 Cleaning up data older than {days} days...[/bold blue]")
        
        deleted_count = db_manager.cleanup_old_data(days)
        
        console.print(f"[bold green]✅ Cleanup completed! Deleted {deleted_count} records[/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]❌ Cleanup failed: {str(e)}[/bold red]")


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]❌ Application error: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main() 