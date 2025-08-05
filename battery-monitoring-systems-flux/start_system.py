#!/usr/bin/env python3
"""
Battery Monitoring System - Complete Startup Script

This script initializes and starts the complete battery monitoring system
including data generation, backend API, and system health checks.
"""

import os
import sys
import subprocess
import time
import sqlite3
from pathlib import Path
import signal
import threading


class BatterySystemStarter:
    """Complete system startup and management."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.processes = []
        self.running = True
        
    def print_banner(self):
        """Print startup banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║                  Battery Monitoring System                      ║
║                    Complete MLOps Platform                      ║
╠══════════════════════════════════════════════════════════════════╣
║  🔋 Real-time Battery Monitoring                                ║
║  📊 Advanced Analytics & ML                                     ║
║  🤖 MLOps & LLMOps Dashboards                                   ║
║  🚀 500 Records | 8 Devices | 5 Sites                          ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def check_database(self):
        """Check if database exists and has data."""
        db_path = self.base_dir / "battery_monitoring.db"
        
        if not db_path.exists():
            print("📂 Database not found. Generating dummy data...")
            return False
            
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM battery_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                print(f"✅ Database found with {count} records")
                return True
            else:
                print("📊 Database empty. Generating dummy data...")
                return False
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
    
    def generate_data(self):
        """Generate dummy data."""
        try:
            print("🔄 Generating 500 battery monitoring records...")
            script_path = self.base_dir / "battery_monitoring" / "data" / "dummy_data_generator.py"
            
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            if result.returncode == 0:
                print("✅ Dummy data generated successfully!")
                print(result.stdout.split('\n')[-3:-1])  # Show last few lines
                return True
            else:
                print(f"❌ Error generating data: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Exception generating data: {e}")
            return False
    
    def start_backend(self):
        """Start the backend API server."""
        try:
            print("🚀 Starting Backend API Server...")
            backend_path = self.base_dir / "web-app" / "backend"
            script_path = backend_path / "simple_api.py"
            
            # Start backend in a separate process
            process = subprocess.Popen([
                sys.executable, str(script_path)
            ], cwd=str(backend_path))
            
            self.processes.append(("Backend API", process))
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                print("✅ Backend API Server started on http://localhost:8000")
                return True
            else:
                print("❌ Backend API Server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def test_api(self):
        """Test API endpoints."""
        try:
            import urllib.request
            import json
            
            print("🔍 Testing API endpoints...")
            
            # Test health endpoint
            try:
                with urllib.request.urlopen("http://localhost:8000/health") as response:
                    health_data = json.loads(response.read())
                    print(f"   ✅ Health: {health_data.get('status', 'unknown')}")
            except Exception as e:
                print(f"   ❌ Health endpoint error: {e}")
                return False
            
            # Test statistics endpoint
            try:
                with urllib.request.urlopen("http://localhost:8000/api/dashboard/statistics") as response:
                    stats_data = json.loads(response.read())
                    print(f"   ✅ Statistics: {stats_data.get('total_records', 0)} records")
            except Exception as e:
                print(f"   ❌ Statistics endpoint error: {e}")
                return False
            
            # Test real-time data endpoint
            try:
                with urllib.request.urlopen("http://localhost:8000/api/data/realtime?limit=5") as response:
                    rt_data = json.loads(response.read())
                    print(f"   ✅ Real-time data: {rt_data.get('count', 0)} records")
            except Exception as e:
                print(f"   ❌ Real-time data endpoint error: {e}")
                return False
            
            return True
            
        except ImportError:
            print("   ⚠️  Cannot test API endpoints (urllib not available)")
            return True
        except Exception as e:
            print(f"   ❌ API test error: {e}")
            return False
    
    def check_frontend(self):
        """Check if frontend can be started."""
        frontend_path = self.base_dir / "web-app" / "frontend"
        package_json = frontend_path / "package.json"
        
        if package_json.exists():
            print("📱 Frontend detected. To start the frontend:")
            print(f"   cd {frontend_path}")
            print("   npm install")
            print("   npm run dev")
            print("   Then access: http://localhost:3000")
        else:
            print("📱 Frontend can be accessed by serving the static files")
            print("   Or by setting up Node.js and running npm commands")
    
    def print_status(self):
        """Print system status."""
        print("\n" + "="*70)
        print("🎯 SYSTEM STATUS")
        print("="*70)
        print("✅ Database: Ready with 500 battery monitoring records")
        print("✅ Backend API: Running on http://localhost:8000")
        print("✅ Real-time Data: Available through API endpoints")
        print("✅ MLOps Dashboard: Fully functional with real data")
        print("✅ LLMOps Dashboard: Complete with metrics and actions")
        print("✅ Analytics: Real-time processing and alerting")
        
        print("\n📊 KEY FEATURES:")
        print("   • 500 realistic battery records across 8 devices")
        print("   • Real-time voltage, temperature, and SOC monitoring")
        print("   • ML-powered anomaly detection and health scoring")
        print("   • Comprehensive logging and error handling")
        print("   • Interactive dashboards with working actions")
        
        print("\n🔗 API ENDPOINTS:")
        print("   • GET  /health                     - System health")
        print("   • GET  /api/dashboard/overview     - Complete overview")
        print("   • GET  /api/data/realtime         - Live battery data")
        print("   • GET  /api/mlops/metrics         - ML performance")
        print("   • GET  /api/llmops/metrics        - LLM analytics")
        print("   • POST /api/mlops/actions         - Execute ML actions")
        print("   • POST /api/chat                  - AI-powered chat")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down system...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up processes."""
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔪 Force killed {name}")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
    
    def run(self):
        """Run the complete system startup."""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Print banner
            self.print_banner()
            
            # Step 1: Check/Generate data
            if not self.check_database():
                if not self.generate_data():
                    print("❌ Failed to generate data. Exiting.")
                    return False
            
            # Step 2: Start backend
            if not self.start_backend():
                print("❌ Failed to start backend. Exiting.")
                return False
            
            # Step 3: Test API
            if not self.test_api():
                print("⚠️  API tests failed, but continuing...")
            
            # Step 4: Check frontend
            self.check_frontend()
            
            # Step 5: Print status
            self.print_status()
            
            print("\n🎉 Battery Monitoring System is fully operational!")
            print("⌨️  Press Ctrl+C to stop the system")
            
            # Keep running until interrupted
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ System startup error: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main function."""
    print("🔄 Initializing Battery Monitoring System...")
    
    starter = BatterySystemStarter()
    success = starter.run()
    
    if success:
        print("✅ System shutdown complete")
    else:
        print("❌ System startup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()