"""
Agricultural Produce Classification Tool
Main entry point for the application
"""

import sys
import os
from src.gui import ProduceClassifierGUI

def main():
    """Initialize and run the application"""
    try:
        app = ProduceClassifierGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
