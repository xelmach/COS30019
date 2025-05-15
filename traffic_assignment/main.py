#!/usr/bin/env python3
#105106819 Suman Sutparai
"""
Traffic-based Route Guidance System (TBRGS)
Main application entry point.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from config_loader import ConfigLoader
from logger import setup_logger

def load_data(config):
    """Load and preprocess the traffic data."""
    try:
        print("Attempting to load data...")
        input_file = config.get_data_config().get('input_file')
        print(f"Looking for input file: {input_file}")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Data file not found: {input_file}")
        
        print("Reading Excel file...")
        try:
            # Read the 'Data' sheet with the correct header row
            print("Reading 'Data' sheet...")
            df = pd.read_excel(input_file, sheet_name='Data', header=1)  # Use row 1 as header (0-based index)
            print("Successfully loaded 'Data' sheet")
            print(f"Data shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            
            # Clean up column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Rename the SCATS site column if it exists
            if 'SCATS Number' in df.columns:
                df = df.rename(columns={'SCATS Number': 'SCATS_SITE'})
                print("Renamed 'SCATS Number' to 'SCATS_SITE'")
            elif 'SCATS_SITE' not in df.columns:
                print("Warning: Could not find SCATS site column")
                print("Available columns:", df.columns.tolist())
            
            # Convert SCATS_SITE to string type if it exists
            if 'SCATS_SITE' in df.columns:
                df['SCATS_SITE'] = df['SCATS_SITE'].astype(str)
            
            print("\nFinal DataFrame columns:", df.columns.tolist())
            return df
            
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def main():
    """Main application entry point."""
    try:
        print("Starting application...")
        # Initialize application
        app = QApplication(sys.argv)
        print("QApplication initialized")
        
        # Setup logging
        global logger
        logger = setup_logger()
        logger.info("Starting TBRGS application")
        print("Logger setup complete")
        
        # Load configuration
        print("Loading configuration...")
        config = ConfigLoader()
        logger.info("Configuration loaded successfully")
        print("Configuration loaded")
        
        # Load data
        print("Loading data...")
        df = load_data(config)
        
        # Create and show main window
        print("Creating main window...")
        window = MainWindow(config, df)
        print("Showing main window...")
        window.show()
        
        # Start application event loop
        print("Starting application event loop...")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 