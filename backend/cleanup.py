import threading
import os
import time
import shutil
from datetime import datetime, timedelta

# Settings
DATA_FOLDER = "data"  # or "data"
EXPIRY_HOURS = 24  # delete folders not modified in the last 12 hours

def is_folder_expired(path, expiry_time):
    last_modified = datetime.fromtimestamp(os.path.getmtime(path))
    return last_modified < expiry_time

def cleanup_sessions():
    now = datetime.now()
    expiry_time = now - timedelta(hours=EXPIRY_HOURS)
    print(f"Cleaning sessions older than {expiry_time}")

    for session_folder in os.listdir(DATA_FOLDER):
        full_path = os.path.join(DATA_FOLDER, session_folder)
        if os.path.isdir(full_path) and is_folder_expired(full_path, expiry_time):
            print(f"Deleting expired session: {full_path}")
            try:
                shutil.rmtree(full_path)
            except Exception as e:
                print(f"Failed to delete {full_path}: {e}")

def start_cleanup_thread():
    def cleanup_loop():
        while True:
            cleanup_sessions()
            time.sleep(3600)  # every hour

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()