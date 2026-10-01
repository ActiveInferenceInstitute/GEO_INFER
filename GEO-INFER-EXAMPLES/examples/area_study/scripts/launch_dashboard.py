#!/usr/bin/env python3
"""
Area Study Dashboard Launcher

Launches an interactive dashboard for exploring area study results.
Provides visualizations and community engagement features.
"""

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

#: Distribution names checked before launch, keyed by import name.
REQUIRED_DEPENDENCIES: dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "plotly": "plotly",
    "requests": "requests",
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("area_study_dashboard")


class AreaStudyDashboard:
    def __init__(self):
        self.logger = setup_logging()
        self.data = None
        self.load_data()

    def load_data(self):
        """Load area study data from output directory."""
        output_dir = Path(__file__).parent.parent / "output"

        # Look for the most recent results file
        json_files = list(output_dir.glob("*area_study_results*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file) as f:
                self.data = json.load(f)
            self.logger.info(f"Loaded data from {latest_file.name}")
        else:
            self.logger.warning("No area study results found, using sample data")
            self.create_sample_data()

    def create_sample_data(self):
        """Create sample data for demonstration."""
        self.data = {
            "study_area": {
                "name": "Downtown Neighborhood",
                "population": 8500,
                "area_hectares": 150,
            },
            "integrated_data": {
                "technical_metrics": {
                    "connectivity_score": 0.72,
                    "infrastructure_quality": 0.68,
                    "iot_sensor_density": 3.2,
                },
                "social_metrics": {
                    "community_cohesion": 0.71,
                    "social_vulnerability": 0.35,
                    "organizational_density": 4.1,
                },
                "environmental_metrics": {
                    "air_quality_index": 68,
                    "green_space_coverage": 0.18,
                    "noise_level": 62,
                },
            },
            "spatial_analysis": {
                "hotspots": {
                    "technical_deficit_zones": ["Zone_3", "Zone_7"],
                    "social_vulnerability_zones": ["Zone_2", "Zone_5"],
                    "environmental_concern_zones": ["Zone_1", "Zone_4"],
                }
            },
        }

    def create_dashboard(self):
        """Create the main dashboard layout."""
        # This method is no longer used since we separated the Streamlit app
        # into a separate file (dashboard_app.py)
        return True


def open_browser(url, delay=2):
    """Open browser after a delay to ensure Streamlit is ready."""
    import time
    import webbrowser
    import threading

    def _open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"🌐 Browser opened at: {url}")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"📱 Please manually open: {url}")

    # Start browser opening in a separate thread
    browser_thread = threading.Thread(target=_open_browser)
    browser_thread.daemon = True
    browser_thread.start()


def check_server_connection(port=8501, timeout=30):
    """Check if Streamlit server is responding."""
    import time
    import requests

    url = f"http://localhost:{port}/_stcore/health"
    start_time = time.time()

    print(f"🔍 Checking server connection on port {port}...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is responding on port {port}")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    print(f"❌ Server not responding after {timeout} seconds")
    return False


def run_streamlit_app(port=8501):
    """Run Streamlit app with robust server management."""
    import subprocess
    import sys
    import time
    import atexit

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_script = os.path.join(script_dir, "dashboard_app.py")

    # Streamlit command with configuration
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        dashboard_script,
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--theme.base",
        "light",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]

    print(f"🚀 Starting Streamlit server on port {port}...")
    print("📱 Dashboard will be available at: http://localhost:8501")
    print("📱 Browser will open automatically when server is ready...")
    print("🛑 Press Ctrl+C to stop the dashboard")
    # Start Streamlit server in background
    try:
        process = subprocess.Popen(
            cmd,
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )

        # Register cleanup function
        def cleanup():
            print("\n🛑 Stopping dashboard server...")
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            print("✅ Dashboard server stopped")

        atexit.register(cleanup)

        # Wait a bit for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)

        # Check if server is responding
        if check_server_connection(port, timeout=15):
            # Open browser
            open_browser(f"http://localhost:{port}", delay=1)
            print(f"🎉 Dashboard is now running at: http://localhost:{port}")
            print("🔄 The server will keep running until you stop it (Ctrl+C)")
        else:
            print("❌ Server failed to start properly")
            print(
                "💡 Try running manually: streamlit run dashboard_app.py --server.port 8501"
            )
            return 1

        # Keep the server running
        try:
            while True:
                if process.poll() is not None:
                    print("⚠️  Server process ended unexpectedly")
                    return process.returncode or 1
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
            cleanup()
            return 0

    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return 1


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")

    missing_deps = [
        package
        for module, package in REQUIRED_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]
    for package in missing_deps:
        print(f"❌ {package} not found")

    if missing_deps:
        print(f"\n📦 Missing dependencies: {', '.join(missing_deps)}")
        print("\n💡 Install with:")
        print("   uv pip install streamlit pandas plotly requests")
        return False

    print("✅ All dependencies found!")
    return True


def main():
    """Main function to launch the dashboard."""
    print("🏛️ GEO-INFER Area Study Dashboard")
    print("=" * 50)
    print("Multi-disciplinary area analysis with automatic browser launch")
    print("=" * 50)

    try:
        # Check dependencies first
        if not check_dependencies():
            return 1

        # Dependencies are good, launch the Streamlit app
        print("\n🔄 Launching interactive dashboard...")
        print("💡 Alternative launch methods if this doesn't work:")
        print("   Simple Python: python3 simple_launch.py")
        print("   Shell script:  ./quick_launch.sh")
        print("   Manual:        streamlit run dashboard_app.py --server.port 8501")
        print()

        return run_streamlit_app(port=8501)

    except Exception as e:
        print(f"❌ Dashboard failed: {e}")
        print("\n💡 Try alternative launch methods:")
        print("   python3 simple_launch.py")
        print("   ./quick_launch.sh")
        logging.exception("Detailed error information:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
