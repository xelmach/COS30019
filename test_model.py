import sys
import os
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from config_loader import ConfigLoader
from processor import DataProcessor

def test_model():
    """Test if the model runs correctly and can form graphs."""
    try:
        # Initialize QApplication
        app = QApplication(sys.argv)
        
        # Load configuration
        config = ConfigLoader()
        
        # Initialize data processor
        processor = DataProcessor(config.get('ml', {}))
        
        # Load data
        print("Loading data...")
        df, edges, coordinates = processor.load_data()
        
        if df is None or len(df) == 0:
            print("Error: No data loaded")
            return False
            
        print(f"Successfully loaded data with {len(df)} rows")
        print(f"Number of edges: {len(edges)}")
        print(f"Number of coordinates: {len(coordinates)}")
        
        # Create main window
        print("Creating main window...")
        window = MainWindow(config, df, edges, coordinates)
        
        # Test a simple route using sites that are known to be connected
        print("\nTesting route finding...")
        test_origin = "4040"  # Site with most data (186 rows)
        test_dest = "4032"    # Another site with good data (124 rows)
        
        # Find route
        routes = window.route_finder.find_top_k_assignment_routes(test_origin, test_dest, k=2)
        
        if routes:
            print("\nRoutes found:")
            for i, (route, time) in enumerate(routes):
                print(f"Route {i+1}: {' -> '.join(route)}")
                print(f"Travel time: {time:.1f} minutes")
        else:
            print("No routes found")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    test_model() 