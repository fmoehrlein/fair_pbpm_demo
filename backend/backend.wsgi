import sys
import os

# Set the working directory to the root of the project
project_root = '/app'
os.chdir(project_root)

# Add the backend folder to Python path so `backend` module can be imported
sys.path.insert(0, os.path.join(project_root, 'backend'))

from backend import app as application
