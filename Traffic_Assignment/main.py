#105106819 Suman Sutparai
# Main entry for TBRGS
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
        print("Loading configuration...")
        config = ConfigLoader()
        
        # Initialize data processor and load data
        print("Initializing data processor...")
        processor = DataProcessor(config.get('ml', {}))
        
        print("Loading data...")
        try:
            traffic_data, edges, coordinates = processor.load_data()
            print(f"Successfully loaded data with {len(traffic_data)} rows")
            print(f"Number of edges: {len(edges)}")
            print(f"Number of coordinates: {len(coordinates)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Debug: Print DataFrame columns
        print("DataFrame columns:", traffic_data.columns.tolist())
        
        # Create and show main window
        print("Creating main window...")
        window = MainWindow(config, traffic_data, edges, coordinates)
        window.show()
        
        # Start event loop
        print("Starting event loop...")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Application error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 