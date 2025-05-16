"""
Main window for the TBRGS GUI.
"""

#105106819 Suman Sutparai
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QTabWidget, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem, QSplitter, QProgressDialog, QApplication, QProgressBar
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap
import folium
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os
import io
from logger import setup_logger
from route_finder import RouteFinder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from train_models import train_models

class TrainingProgress(QObject):
    epoch_signal = pyqtSignal(int, int)  # current, total
    training_done = pyqtSignal()

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, progress):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress = progress
    def on_epoch_begin(self, epoch, logs=None):
        self.progress.epoch_signal.emit(epoch + 1, self.total_epochs)

class MainWindow(QMainWindow):
    """Main window for the TBRGS application."""
    
    def __init__(self, config, df, edges=None, coordinates=None):
        """Initialize the main window.
        
        Args:
            config (ConfigLoader): Configuration loader instance.
            df (pandas.DataFrame): Traffic data.
            edges (list): List of (origin, dest, weight) tuples for the graph.
            coordinates (dict): Dictionary of site coordinates.
        """
        super().__init__()
        self.config = config
        
        # Debug: Print raw DataFrame info
        print("\nRaw DataFrame Info:")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"site_id dtype: {df['site_id'].dtype}")
        print(f"Sample of site_ids: {df['site_id'].head()}")
        
        # Create a copy and ensure site_id is string
        self.df = df.copy()
        self.df['site_id'] = self.df['site_id'].astype(str)
        
        # Debug: Print processed DataFrame info
        print("\nProcessed DataFrame Info:")
        print(f"DataFrame shape: {self.df.shape}")
        print(f"site_id dtype: {self.df['site_id'].dtype}")
        print(f"Sample of site_ids: {self.df['site_id'].head()}")
        print(f"Unique site_ids: {sorted(self.df['site_id'].unique())}")
        
        # Debug: Check for site 2000 specifically
        site_2000_data = self.df[self.df['site_id'] == '2000']
        print(f"\nSite 2000 data:")
        print(f"Number of rows: {len(site_2000_data)}")
        if len(site_2000_data) > 0:
            print(f"Sample row: {site_2000_data.iloc[0].to_dict()}")
        
        self.logger = setup_logger()
        self.setWindowTitle("Traffic-based Route Guidance System (TBRGS)")
        
        # Set window size from config
        window_size = self.config.get_gui_config().get('window_size', '1200x800')
        width, height = map(int, window_size.split('x'))
        self.setGeometry(100, 100, width, height)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.tab_widget.addTab(self._create_route_tab(), "Route Planning")
        self.tab_widget.addTab(self._create_prediction_tab(), "Traffic Prediction")
        self.tab_widget.addTab(self._create_settings_tab(), "Settings")
        
        # Initialize route finder
        self.route_finder = RouteFinder(self.config)
        if edges and coordinates:
            print("\nInitializing route finder with graph data...")
            print(f"Number of edges: {len(edges)}")
            print(f"Number of coordinates: {len(coordinates)}")
            self.route_finder.build_graph(edges, coordinates)
        
        # Initialize site dropdowns
        self._initialize_sites()
        
        # Initialize progress signal
        self.progress = TrainingProgress()
        
    def _initialize_sites(self):
        """Initialize site dropdowns."""
        try:
            print("\nInitializing sites...")
            print(f"DataFrame shape before processing: {self.df.shape}")
            
            # Get unique sites with their descriptions
            sites = self.df[['site_id', 'Location']].drop_duplicates()
            print(f"Unique sites found: {len(sites)}")
            
            # Sort sites by ID
            sites = sites.sort_values('site_id')
            
            # Log available site IDs for debugging
            print("\nAvailable site IDs:")
            for _, row in sites.iterrows():
                print(f"Site {row['site_id']}: {row['Location']}")
            
            # Clear existing items
            self.origin_combo.clear()
            self.dest_combo.clear()
            self.site_combo.clear()
            
            # Add items to dropdowns
            for _, row in sites.iterrows():
                display_text = f"{row['site_id']} - {row['Location']}"
                self.origin_combo.addItem(display_text)
                self.dest_combo.addItem(display_text)
                self.site_combo.addItem(display_text)
            
            print(f"\nAdded {self.origin_combo.count()} items to dropdowns")
            
            # Set default selections
            if self.origin_combo.count() > 0:
                self.origin_combo.setCurrentIndex(0)
                print(f"Default origin: {self.origin_combo.currentText()}")
            if self.dest_combo.count() > 0:
                self.dest_combo.setCurrentIndex(0)
                print(f"Default destination: {self.dest_combo.currentText()}")
            if self.site_combo.count() > 0:
                self.site_combo.setCurrentIndex(0)
                print(f"Default site: {self.site_combo.currentText()}")
                
        except Exception as e:
            print(f"Error initializing sites: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error initializing sites: {str(e)}")
            raise

    def create_controls(self):
        self.controls_widget = QWidget()
        controls_layout = QHBoxLayout(self.controls_widget)
        self.start_label = QLabel("Start:")
        self.start_input = QLineEdit()
        self.end_label = QLabel("End:")
        self.end_input = QLineEdit()
        self.find_route_btn = QPushButton("Find Route")
        self.find_route_btn.clicked.connect(self.find_route)
        controls_layout.addWidget(self.start_label)
        controls_layout.addWidget(self.start_input)
        controls_layout.addWidget(self.end_label)
        controls_layout.addWidget(self.end_input)
        controls_layout.addWidget(self.find_route_btn)
        
    def create_map(self):
        self.map_widget = QWebEngineView()
        m = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)
        data = io.BytesIO()
        m.save(data, close_file=False)
        self.map_widget.setHtml(data.getvalue().decode())
        
    def create_results_table(self):
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Route", "Cost", "Travel Time (min)"])
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.map_widget)
        self.splitter.addWidget(self.results_table)
        self.splitter.setSizes([500, 200])
        
    def find_route(self):
        start = self.start_input.text().strip()
        end = self.end_input.text().strip()
        if not start or not end:
            QMessageBox.warning(self, "Input Error", "Please enter both start and end locations.")
            return
        self.logger.info(f"Finding route from {start} to {end}")
        self.results_table.setRowCount(0)
        for i in range(3):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(f"Route {i+1}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(10 + i)))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(20 + i*5)))
        self.update_map_with_route()
        
    def update_map_with_route(self):
        m = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)
        folium.Marker([-37.8136, 144.9631], tooltip="Start").add_to(m)
        folium.Marker([-37.8186, 144.9731], tooltip="End").add_to(m)
        folium.PolyLine([[-37.8136, 144.9631], [-37.8186, 144.9731]], color="blue", weight=5).add_to(m)
        data = io.BytesIO()
        m.save(data, close_file=False)
        self.map_widget.setHtml(data.getvalue().decode())
        
    def _create_route_tab(self):
        """Create the route planning tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Origin selection
        origin_layout = QHBoxLayout()
        origin_label = QLabel("Origin Site:")
        self.origin_combo = QComboBox()
        origin_layout.addWidget(origin_label)
        origin_layout.addWidget(self.origin_combo)
        layout.addLayout(origin_layout)
        
        # Destination selection
        dest_layout = QHBoxLayout()
        dest_label = QLabel("Destination Site:")
        self.dest_combo = QComboBox()
        dest_layout.addWidget(dest_label)
        dest_layout.addWidget(self.dest_combo)
        layout.addLayout(dest_layout)
        
        # Find routes button
        find_routes_btn = QPushButton("Find Routes")
        find_routes_btn.clicked.connect(self._find_routes)
        layout.addWidget(find_routes_btn)
        
        # Map view
        self.map_view = QWebEngineView()
        layout.addWidget(self.map_view)
        
        # Initialize map with default view of Melbourne
        self._initialize_map()
        
        return widget
        
    def _initialize_map(self):
        """Initialize the map with a default view of Melbourne."""
        try:
            # Get map center and zoom from config
            map_center = self.config.get_gui_config().get('map_center', [-37.8136, 144.9631])
            map_zoom = self.config.get_gui_config().get('map_zoom', 12)
            
            # Create map
            m = folium.Map(location=map_center, zoom_start=map_zoom)
            
            # Add all sites as markers
            for _, row in self.df.drop_duplicates('site_id').iterrows():
                folium.Marker(
                    [row['NB_LATITUDE'], row['NB_LONGITUDE']],
                    tooltip=f"Site: {row['site_id']}",
                    popup=f"Location: {row['Location']}"
                ).add_to(m)
            
            # Save and display map
            data = io.BytesIO()
            m.save(data, close_file=False)
            self.map_view.setHtml(data.getvalue().decode())
            
        except Exception as e:
            self.logger.error(f"Error initializing map: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error initializing map: {str(e)}")
        
    def _create_prediction_tab(self):
        """Create the traffic prediction tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Site selection
        site_layout = QHBoxLayout()
        site_label = QLabel("Site:")
        self.site_combo = QComboBox()
        site_layout.addWidget(site_label)
        site_layout.addWidget(self.site_combo)
        layout.addLayout(site_layout)
        
        # Prediction horizon
        horizon_layout = QHBoxLayout()
        horizon_label = QLabel("Prediction Horizon (hours):")
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 24)
        self.horizon_spin.setValue(1)
        horizon_layout.addWidget(horizon_label)
        horizon_layout.addWidget(self.horizon_spin)
        layout.addLayout(horizon_layout)
        
        # Predict button
        predict_btn = QPushButton("Predict Traffic")
        predict_btn.clicked.connect(self._predict_traffic)
        layout.addWidget(predict_btn)
        
        # Prediction results
        self.prediction_label = QLabel("Prediction results will appear here")
        layout.addWidget(self.prediction_label)
        
        return widget
        
    def _create_settings_tab(self):
        """Create the settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Add settings controls here
        settings_label = QLabel("Settings will be implemented here")
        layout.addWidget(settings_label)
        
        return widget
        
    def _find_routes(self):
        """Find routes between selected sites."""
        try:
            # Get selected sites
            origin_text = self.origin_combo.currentText()
            dest_text = self.dest_combo.currentText()
            
            print(f"\nFinding routes...")
            print(f"Selected origin text: {origin_text}")
            print(f"Selected destination text: {dest_text}")
            
            # Extract site IDs from the display text
            origin_id = origin_text.split(' - ')[0]
            dest_id = dest_text.split(' - ')[0]
            
            print(f"\nDebug - Searching for sites:")
            print(f"Origin ID: {origin_id} (type: {type(origin_id)})")
            print(f"Destination ID: {dest_id} (type: {type(dest_id)})")
            print(f"DataFrame site_id type: {self.df['site_id'].dtype}")
            print(f"Unique site IDs in DataFrame: {sorted(self.df['site_id'].unique())}")
            
            # Debug: Check exact values in DataFrame
            print("\nChecking DataFrame values:")
            print(f"DataFrame site_id values: {self.df['site_id'].value_counts().head()}")
            
            # Get site data with explicit string conversion
            origin_data = self.df[self.df['site_id'].astype(str) == str(origin_id)]
            dest_data = self.df[self.df['site_id'].astype(str) == str(dest_id)]
            
            print(f"\nDebug - Found data:")
            print(f"Origin data rows: {len(origin_data)}")
            print(f"Destination data rows: {len(dest_data)}")
            
            if len(origin_data) == 0:
                error_msg = f"No data found for origin site {origin_id}"
                print(error_msg)
                print(f"Available site IDs: {sorted(self.df['site_id'].unique())}")
                QMessageBox.warning(self, "Error", error_msg)
                return
                
            if len(dest_data) == 0:
                error_msg = f"No data found for destination site {dest_id}"
                print(error_msg)
                print(f"Available site IDs: {sorted(self.df['site_id'].unique())}")
                QMessageBox.warning(self, "Error", error_msg)
                return
            
            # Get coordinates
            origin_coords = (origin_data.iloc[0]['NB_LATITUDE'], origin_data.iloc[0]['NB_LONGITUDE'])
            dest_coords = (dest_data.iloc[0]['NB_LATITUDE'], dest_data.iloc[0]['NB_LONGITUDE'])
            
            print(f"\nDebug - Coordinates:")
            print(f"Origin: {origin_coords}")
            print(f"Destination: {dest_coords}")
            
            # Show progress dialog
            progress = QProgressDialog("Calculating routes...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Route Calculation")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            
            # Update progress
            progress.setLabelText("Initializing route finder...")
            progress.setValue(10)
            QApplication.processEvents()
            
            # Find routes
            progress.setLabelText("Finding shortest paths...")
            progress.setValue(30)
            QApplication.processEvents()
            
            routes = self.route_finder.find_top_k_routes(origin_id, dest_id, k=3)
            
            if not routes:
                progress.close()
                error_msg = "No routes found between the selected sites"
                print(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
            
            # Update progress
            progress.setLabelText("Processing routes...")
            progress.setValue(60)
            QApplication.processEvents()
            
            # Clear existing routes
            self.map_view.setHtml("")  # Clear the map
            
            # Create new map
            m = folium.Map(location=origin_coords, zoom_start=12)
            
            # Add markers for origin and destination
            folium.Marker(
                origin_coords,
                tooltip=f"Origin: {origin_id}",
                popup=f"Location: {origin_data.iloc[0]['Location']}"
            ).add_to(m)
            
            folium.Marker(
                dest_coords,
                tooltip=f"Destination: {dest_id}",
                popup=f"Location: {dest_data.iloc[0]['Location']}"
            ).add_to(m)
            
            # Update progress
            progress.setLabelText("Drawing routes on map...")
            progress.setValue(80)
            QApplication.processEvents()
            
            # Add routes to map
            colors = ['blue', 'red', 'green']
            for i, (route, cost) in enumerate(routes):
                route_coords = self.route_finder.get_route_coords(route)
                if route_coords:
                    folium.PolyLine(
                        route_coords,
                        color=colors[i % len(colors)],
                        weight=5,
                        opacity=0.8,
                        tooltip=f"Route {i+1} - Cost: {cost:.2f} minutes"
                    ).add_to(m)
            
            # Save and display map
            data = io.BytesIO()
            m.save(data, close_file=False)
            self.map_view.setHtml(data.getvalue().decode())
            
            # Complete progress
            progress.setLabelText("Route calculation complete!")
            progress.setValue(100)
            QApplication.processEvents()
            progress.close()
            
            # Show route summary
            summary = "Route Summary:\n\n"
            for i, (route, cost) in enumerate(routes):
                summary += f"Route {i+1}:\n"
                summary += f"Path: {' -> '.join(route)}\n"
                summary += f"Travel Time: {cost:.2f} minutes\n\n"
            
            QMessageBox.information(self, "Routes Found", summary)
            
        except Exception as e:
            error_msg = f"Error finding routes: {str(e)}"
            print(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
            raise

    def _predict_traffic(self):
        """Handle predict traffic button click."""
        site = self.site_combo.currentText()
        
        if not site:
            QMessageBox.warning(self, "Error", "Please select a site")
            return
            
        try:
            # Extract site_id from the display text
            site_id = site.split(' - ')[0]
            
            # Show progress dialog
            epochs = 50  # or get from config
            progress_dialog = QProgressDialog("Training model...", "Cancel", 0, epochs, self)
            progress_dialog.setWindowTitle("Training Progress")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()
            
            def update_progress(current, total):
                progress_dialog.setValue(current)
                progress_dialog.setLabelText(f"Training model... Epoch {current}/{total}")
                QApplication.processEvents()
                
            progress = TrainingProgress()
            progress.epoch_signal.connect(update_progress)
            
            def on_training_done():
                progress_dialog.close()
                self.predict_traffic_after_training(site_id)
                
            progress.training_done.connect(on_training_done)
            
            import threading
            def train_and_predict():
                train_models(site_id=int(site_id), progress_callback=ProgressCallback(epochs, progress), epochs_override=epochs)
                progress.training_done.emit()
                
            threading.Thread(target=train_and_predict).start()
            
        except Exception as e:
            self.logger.error(f"Error predicting traffic: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error predicting traffic: {str(e)}")
            
    def predict_traffic_after_training(self, site_id):
        try:
            # Convert site_id to string for consistent comparison
            site_id = str(site_id)
            
            # Get data for the selected site
            site_data = self.df[self.df['site_id'].astype(str) == site_id].copy()
            
            print(f"\nDebug - Prediction for site {site_id}:")
            print(f"DataFrame shape: {self.df.shape}")
            print(f"Site data rows found: {len(site_data)}")
            
            if len(site_data) == 0:
                QMessageBox.warning(self, "Error", f"No data found for site {site_id}")
                return
            
            # Get the time columns (15-minute intervals)
            time_columns = [col for col in site_data.columns if (':' in str(col)) or (str(col).startswith('V') and str(col)[1:].isdigit() and len(str(col)) == 3)]
            
            if not time_columns:
                QMessageBox.warning(self, "Error", "No time-based columns found in the data")
                return
            
            print(f"Found {len(time_columns)} time columns")
            
            # Convert time columns to numeric values and normalize
            for col in time_columns:
                site_data[col] = pd.to_numeric(site_data[col], errors='coerce')
            
            # Prepare data for prediction
            window_size = 12  # Using the same window size as the GRU model
            data = site_data[time_columns].values.flatten()
            
            # Normalize the data
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data.reshape(-1, 1))
            
            # Create sequences for prediction
            X = []
            for i in range(len(normalized_data) - window_size):
                X.append(normalized_data[i:i + window_size])
            X = np.array(X)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Load and use the GRU model
            model_path = f"models/gru_site_{site_id}.h5"  # Changed from gru_model_scats to gru_site
            if not os.path.exists(model_path):
                QMessageBox.warning(
                    self,
                    "Model Not Found",
                    f"Prediction model for site {site_id} not found. Please train the model first."
                )
                return
            
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(predictions)
            
            # Plot the predictions
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(predictions)), predictions, 'b-', label='Predicted')
            plt.title(f'Traffic Prediction for Site {site_id}')
            plt.xlabel('Time Step')
            plt.ylabel('Traffic Volume')
            plt.legend()
            plt.tight_layout()
            
            # Save plot to a temporary file
            temp_file = 'temp_prediction.png'
            plt.savefig(temp_file)
            plt.close()
            
            # Display the plot in the QLabel
            self.prediction_label.setPixmap(QPixmap(temp_file))
            
            # Show prediction summary
            avg_prediction = np.mean(predictions)
            max_prediction = np.max(predictions)
            min_prediction = np.min(predictions)
            
            QMessageBox.information(
                self,
                "Prediction Complete",
                f"Traffic prediction for site {site_id}:\n\n" +
                f"Average predicted volume: {avg_prediction:.0f}\n" +
                f"Maximum predicted volume: {max_prediction:.0f}\n" +
                f"Minimum predicted volume: {min_prediction:.0f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting traffic: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error predicting traffic: {str(e)}") 