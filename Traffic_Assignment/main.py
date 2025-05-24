#105106819 Suman Sutparai
# Main entry for TBRGS
import sys
import os
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from main_window import MainWindow
from logger import setup_logger
from config_loader import ConfigLoader
from processor import DataProcessor
import pandas as pd
from train_models import train_all_sites
import glob

def all_models_exist():
    data_dir = "data"
    model_dir = "models"
    files = glob.glob(os.path.join(data_dir, "scats_*_normalized.csv"))
    for file in files:
        site_id = file.split('_')[1]
        model_path = os.path.join(model_dir, f"lstm_site_{site_id}.h5")
        if not os.path.exists(model_path):
            return False
    return True

def main():
    # Setup logging
    logger = setup_logger()
    logger.info("Starting TBRGS application")
    
    try:
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("Traffic-based Route Guidance System")
        
        # Show loading splash if needed
        # if not all_models_exist():
        #     splash = QWidget()
        #     splash.setWindowTitle("TBRGS - Loading")
        #     layout = QVBoxLayout(splash)
        #     label = QLabel("<h2>Opening...<br>Training models, please wait...</h2>")
        #     label.setAlignment(Qt.AlignCenter)
        #     layout.addWidget(label)
        #     splash.setLayout(layout)
        #     splash.resize(400, 200)
        #     splash.show()
        #     app.processEvents()
        #     train_all_sites()
        #     splash.close()
        
        # Load configuration
        print("Loading configuration...")
        config = ConfigLoader()
        
        # Initialize data processor and load data
        print("Initializing data processor...")
        processor = DataProcessor(config.get('ml', {}))
        
        print("Loading valid edges from edges.csv...")
        edges_csv_path = os.path.join(os.path.dirname(__file__), 'edges.csv')
        if os.path.exists(edges_csv_path):
            edges_df = pd.read_csv(edges_csv_path)
            valid_edges = list(edges_df.itertuples(index=False, name=None))
        else:
            print("edges.csv not found! Using fallback (fully connected graph, not recommended)")
            valid_edges = None
        
        print("Loading data...")
        try:
            traffic_data, edges, coordinates = processor.load_data()
            # Rebuild edges using valid_edges if available
            if valid_edges is not None:
                edges, coordinates = processor.build_graph(traffic_data, coordinates, valid_edges)
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
        
        # Create main window
        print("Creating main window...")
        window = MainWindow(config, traffic_data, edges, coordinates)
        
        # Ensure proper initialization order
        def initialize_ui():
            try:
                window.show()
                # Force a repaint to ensure all widgets are properly initialized
                window.repaint()
            except Exception as e:
                print(f"Error during UI initialization: {str(e)}")
                logger.error(f"Error during UI initialization: {str(e)}")
        
        # Use QTimer to ensure proper initialization
        QTimer.singleShot(100, initialize_ui)
        
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