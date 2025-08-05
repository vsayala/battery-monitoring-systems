#!/usr/bin/env python3
"""
CD4ML System Demo - Example Usage

This demo script shows how to use the CD4ML system for battery monitoring.
It demonstrates the complete pipeline with real-time monitoring and feedback loops.

Usage:
    python -m battery_monitoring.mlops.examples.cd4ml_demo
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from battery_monitoring.mlops.cd4ml_system import CD4MLSystem
from rich.console import Console
from rich.panel import Panel

console = Console()


async def run_demo():
    """Run the CD4ML system demo."""
    console.print(Panel.fit(
        "[bold yellow]CD4ML System Demo[/bold yellow]\n"
        "[dim]Battery Monitoring with MLOps/LLMOps[/dim]",
        title="🎯 Demo Mode",
        border_style="yellow"
    ))
    
    # Initialize the CD4ML system
    cd4ml = CD4MLSystem()
    
    try:
        # Run the complete pipeline
        console.print("\n[bold cyan]Starting CD4ML Pipeline Demo...[/bold cyan]")
        results = await cd4ml.run_pipeline()
        
        # Show the system dashboard
        console.print("\n[bold green]Demo Results:[/bold green]")
        cd4ml.show_system_dashboard()
        
        # Show detailed results
        console.print("\n[bold]Pipeline Results:[/bold]")
        for key, value in results.items():
            if key != 'timestamp':
                console.print(f"  • {key}: {value}")
        
        console.print("\n[bold green]✅ Demo completed successfully![/bold green]")
        console.print("[dim]The CD4ML system is now ready for production use.[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(run_demo()) 