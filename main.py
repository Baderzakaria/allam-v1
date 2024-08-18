import subprocess
import signal
import os
import sys
import psutil

# Paths to your Flask and Streamlit scripts
FLASK_APP_PATH = 'app.py'
STREAMLIT_APP_PATH = 'streamlit_app.py'

# Define the ports used by Flask and Streamlit
FLASK_PORT = 5001
STREAMLIT_PORT = 8501

# Function to clean up ports
def free_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.net_connections()
            for conn in connections:
                if conn.laddr.port == port:
                    print(f"Terminating process on port {port}: {proc.info['name']} (PID {proc.info['pid']})")
                    proc.kill()  # Forcefully terminate the process
                    proc.wait(timeout=3)  # Wait with a timeout to prevent hanging
                    print(f"Process on port {port} terminated.")
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess):
            continue
        except Exception as e:
            print(f"Error accessing process information: {e}")

# Function to handle termination signals
def signal_handler(sig, frame):
    print("Shutting down both Flask and Streamlit...")

    # Terminate both processes
    flask_process.terminate()
    streamlit_process.terminate()

    # Wait for processes to terminate
    flask_process.wait(timeout=5)
    streamlit_process.wait(timeout=5)

    # Free the ports used by Flask and Streamlit
    free_port(FLASK_PORT)
    free_port(STREAMLIT_PORT)

    print("Both Flask and Streamlit have been shut down and ports have been freed.")
    sys.exit(0)

# Free the ports before starting the servers, in case they're still in use
free_port(FLASK_PORT)
free_port(STREAMLIT_PORT)

# Start the Flask server
flask_process = subprocess.Popen(['python', FLASK_APP_PATH])

# Start the Streamlit app
streamlit_process = subprocess.Popen(['streamlit', 'run', STREAMLIT_APP_PATH])

# Attach the signal handler to handle CTRL+C or termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Wait for the processes to finish
try:
    flask_process.wait()
    streamlit_process.wait()
except KeyboardInterrupt:
    signal_handler(None, None)
