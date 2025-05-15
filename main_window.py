"""
Main window for the TBRGS GUI.
"""

#105106819 Suman Sutparai
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QTabWidget, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem, QSplitter
)
from PyQt5.QtCore import Qt
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
from PyQt5.QtGui import QPixmap
import tensorflow as tf

class MainWindow(QMainWindow):
    """Main window for the TBRGS application."""
    
    def __init__(self, config, df):
        """Initialize the main window.
        
        Args:
            config (ConfigLoader): Configuration loader instance.
            df (pandas.DataFrame): Traffic data.
        """
        super().__init__()
        self.config = config
        self.df = df
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
        
        # Initialize SCATS site dropdowns
        self._initialize_scats_sites()
        
    def _initialize_scats_sites(self):
        """Initialize SCATS site dropdowns with data from the DataFrame."""
        try:
            # Try different possible column names for SCATS sites
            possible_columns = ['SCATS_SITE', 'Site', 'Site ID', 'SCATS Site', 'SCATS_ID', 'SiteID']
            scats_column = None
            
            for col in possible_columns:
                if col in self.df.columns:
                    scats_column = col
                    break
            
            if scats_column is None:
                available_columns = ', '.join(self.df.columns)
                error_msg = f"Could not find SCATS site column. Available columns: {available_columns}"
                self.logger.error(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
            
            # Get unique SCATS sites from the data
            scats_sites = sorted(self.df[scats_column].unique())
            
            if len(scats_sites) == 0:
                error_msg = "No SCATS sites found in the data"
                self.logger.error(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
                return
            
            # Add sites to dropdowns
            self.origin_combo.addItems(map(str, scats_sites))
            self.dest_combo.addItems(map(str, scats_sites))
            self.site_combo.addItems(map(str, scats_sites))
            
            self.logger.info(f"Initialized {len(scats_sites)} SCATS sites using column '{scats_column}'")
        except Exception as e:
            error_msg = f"Error initializing SCATS sites: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
        
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
        origin_label = QLabel("Origin SCATS Site:")
        self.origin_combo = QComboBox()
        origin_layout.addWidget(origin_label)
        origin_layout.addWidget(self.origin_combo)
        layout.addLayout(origin_layout)
        
        # Destination selection
        dest_layout = QHBoxLayout()
        dest_label = QLabel("Destination SCATS Site:")
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
        
        return widget
        
    def _create_prediction_tab(self):
        """Create the traffic prediction tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # SCATS site selection
        site_layout = QHBoxLayout()
        site_label = QLabel("SCATS Site:")
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
        """Handle find routes button click."""
        origin = self.origin_combo.currentText()
        destination = self.dest_combo.currentText()
        
        if not origin or not destination:
            QMessageBox.warning(self, "Error", "Please select both origin and destination")
            return
        
        try:
            # Get coordinates for origin and destination
            origin_data = self.df[self.df['SCATS_SITE'] == origin].iloc[0]
            dest_data = self.df[self.df['SCATS_SITE'] == destination].iloc[0]
            
            # Create edges from the data
            edges = []
            coordinates = {}
            
            # Add origin and destination coordinates
            coordinates[origin] = (origin_data['NB_LATITUDE'], origin_data['NB_LONGITUDE'])
            coordinates[destination] = (dest_data['NB_LATITUDE'], dest_data['NB_LONGITUDE'])
            
            # Create a simple edge between origin and destination for now
            # In a real implementation, this would use actual road network data
            edges.append((origin, destination, 1.0))  # Using 1.0 as default cost
            
            # Initialize route finder
            route_finder = RouteFinder(self.config)
            route_finder.build_graph(edges, coordinates)
            
            # Find top 3 routes
            routes = route_finder.find_top_k_routes(origin, destination, k=3)
            
            if not routes:
                QMessageBox.warning(self, "No Routes", "No routes found between the selected locations")
                return
            
            # Update map with routes
            m = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)
            
            # Add markers for origin and destination
            folium.Marker(
                [origin_data['NB_LATITUDE'], origin_data['NB_LONGITUDE']],
                tooltip=f"Origin: {origin}"
            ).add_to(m)
            
            folium.Marker(
                [dest_data['NB_LATITUDE'], dest_data['NB_LONGITUDE']],
                tooltip=f"Destination: {destination}"
            ).add_to(m)
            
            # Add routes to map
            colors = ['blue', 'green', 'red']
            for i, (path, cost) in enumerate(routes):
                # Create polyline for the route
                route_points = [coordinates[node] for node in path]
                folium.PolyLine(
                    route_points,
                    color=colors[i % len(colors)],
                    weight=5,
                    opacity=0.7,
                    tooltip=f"Route {i+1}"
                ).add_to(m)
            
            # Save and display map
            data = io.BytesIO()
            m.save(data, close_file=False)
            self.map_view.setHtml(data.getvalue().decode())
            
            # Show route information
            QMessageBox.information(
                self,
                "Routes Found",
                f"Found {len(routes)} routes between {origin} and {destination}\n" +
                "\n".join([f"Route {i+1}: {len(path)-1} steps, cost: {cost:.2f}" 
                          for i, (path, cost) in enumerate(routes)])
            )
            
        except Exception as e:
            self.logger.error(f"Error finding routes: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error finding routes: {str(e)}")
        
    def _predict_traffic(self):
        """Handle predict traffic button click."""
        site = self.site_combo.currentText()
        horizon = self.horizon_spin.value()
        
        if not site:
            QMessageBox.warning(self, "Error", "Please select a SCATS site")
            return
        
        try:
            # Get data for the selected site
            site_data = self.df[self.df['SCATS_SITE'] == site].copy()
            
            if len(site_data) == 0:
                QMessageBox.warning(self, "Error", f"No data found for SCATS site {site}")
                return
            
            # Get the time columns (15-minute intervals)
            time_columns = [col for col in site_data.columns if ':' in str(col)]
            if not time_columns:
                QMessageBox.warning(self, "Error", "No time-based columns found in the data")
                return
            
            # Convert time columns to numeric values
            for col in time_columns:
                site_data[col] = pd.to_numeric(site_data[col], errors='coerce')
            
            # Prepare data for prediction
            sequence_length = self.config.get('data', {}).get('sequence_length', 24)
            prediction_horizon = horizon * 4  # Convert hours to 15-minute intervals
            
            # Get the most recent sequence
            recent_data = site_data[time_columns].iloc[-sequence_length:].values.flatten()
            
            if len(recent_data) == 0:
                QMessageBox.warning(self, "Error", "No recent data available for prediction")
                return
            
            # Normalize the data
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(recent_data.reshape(-1, 1))
            
            # Reshape for LSTM input (samples, time steps, features)
            X = normalized_data.reshape(1, sequence_length, 1)
            
            # Load and use the LSTM model
            model_path = f"models/lstm_site_{site}.h5"
            if not os.path.exists(model_path):
                QMessageBox.warning(
                    self,
                    "Model Not Found",
                    f"Prediction model for site {site} not found. Please train the model first."
                )
                return
            
            model = tf.keras.models.load_model(model_path)
            
            # Make predictions
            predictions = []
            current_input = X.copy()
            
            for _ in range(prediction_horizon):
                # Make prediction
                pred = model.predict(current_input, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update input sequence
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1] = pred[0, 0]
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)
            
            # Create time points for x-axis
            last_time = pd.to_datetime(site_data['Date'].iloc[-1])
            time_points = [last_time + pd.Timedelta(minutes=15*(i+1)) for i in range(prediction_horizon)]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, predictions, 'b-', label='Predicted Traffic')
            plt.title(f'Traffic Prediction for SCATS Site {site}')
            plt.xlabel('Time')
            plt.ylabel('Traffic Volume')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save plot to a temporary file
            temp_file = 'temp_prediction.png'
            plt.savefig(temp_file)
            plt.close()
            
            # Display the plot
            self.prediction_label.setPixmap(QPixmap(temp_file))
            
            # Show prediction summary
            avg_prediction = np.mean(predictions)
            max_prediction = np.max(predictions)
            min_prediction = np.min(predictions)
            
            QMessageBox.information(
                self,
                "Prediction Complete",
                f"Traffic prediction for site {site}:\n\n" +
                f"Average predicted volume: {avg_prediction:.0f}\n" +
                f"Maximum predicted volume: {max_prediction:.0f}\n" +
                f"Minimum predicted volume: {min_prediction:.0f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting traffic: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error predicting traffic: {str(e)}") 