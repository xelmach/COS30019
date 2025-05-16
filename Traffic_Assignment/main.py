#105106819 Suman Sutparai

import sys
import os
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from logger import setup_logger
from config_loader import ConfigLoader
from processor import DataProcessor

def main():
    # Setup logging
    logger = setup_logger()
    logger.info("Starting TBRGS application")
    
    try:
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("Traffic-based Route Guidance System")
        
        # Load configuration
        config = ConfigLoader()
        
        # Initialize data processor and load data
        processor = DataProcessor(config.get('ml', {}))
        traffic_data, edges, coordinates = processor.load_data()
        
        # Debug: Print DataFrame columns
        print("DataFrame columns:", traffic_data.columns.tolist())
        
        # Create and show main window
        window = MainWindow(config, traffic_data, edges, coordinates)
        window.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 