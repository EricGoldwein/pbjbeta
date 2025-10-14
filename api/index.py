#!/usr/bin/env python3
"""
Vercel-compatible Flask app for facility 495241 dashboard
This is the main entry point for Vercel deployment
"""

import sys
import os

# Add the parent directory to the path so we can import the main app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main Flask app
from facility_495241_flask_app import app

# This is the entry point for Vercel
if __name__ == '__main__':
    app.run()
