#105106819 Suman Sutparai
# Main window for TBRGS
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QTabWidget, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem, QSplitter, QProgressDialog, QApplication, QProgressBar, QFrame, QTextEdit, QSizePolicy, QListWidget, QListWidgetItem, QAbstractItemView, QScrollArea, QDateEdit, QTimeEdit
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer, QThread, QDate, QTime
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
import openrouteservice
from route_models.dfs_search import dfs
from route_models.bfs_search import build_graph as bfs_build_graph, bfs
from route_models.gbfs_search import build_graph as gbfs_build_graph, gbfs
from route_models.astar_search import build_graph as astar_build_graph, astar
from route_models.cus1_search import build_graph as cus1_build_graph, cus1_search
from route_models.cus2_search import graph as cus2_graph, cus2_search
from collections import defaultdict
import math
import geopy.distance
import threading
import matplotlib.dates as mdates

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
    
    prediction_result_signal = pyqtSignal(object, str)  # (QPixmap, summary_text)
    
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
        self.df = df.copy()
        self.df['site_id'] = self.df['site_id'].astype(str)
        
        # Initialize logger
        self.logger = setup_logger()
        self.setWindowTitle("Traffic-based Route Guidance System (TBRGS)")
        
        # Set window size
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.layout.setSpacing(18)
        
        # Create title label
        self.title_label = QLabel("Traffic-based Route Guidance System (TBRGS)")
        self.title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #fff; margin-bottom: 12px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabWidget::pane { border-radius: 12px; padding: 8px; }")
        self.layout.addWidget(self.tab_widget)
        
        # Initialize route finder
        self.route_finder = RouteFinder()
        # Build the real graph from the provided edges (undirected)
        self.real_graph = self.build_real_graph(edges) if edges else {}
        
        # Initialize progress signal
        self.progress = TrainingProgress()
        
        # Create tabs
        self.tab_widget.addTab(self._create_route_tab(), "Route Planning")
        self.tab_widget.addTab(self._create_settings_tab(), "Settings")

        # Set dark theme as default, after theme_combo is created
        if hasattr(self, 'theme_combo') and self.theme_combo is not None:
            self.theme_combo.setCurrentText("Dark")
            self.apply_theme("Dark")
        
        # Initialize site dropdowns after all UI elements are created
        QTimer.singleShot(100, self._delayed_initialize_sites)
        
        # Show window
        self.show()

        self.prediction_result_signal.connect(self._update_prediction_result)

        self.ors_cache = {}  # Cache for ORS polyline results

    def _delayed_initialize_sites(self):
        """Initialize site dropdowns with a delay to ensure proper widget creation."""
        try:
            # Get unique sites with their descriptions
            sites = self.df[['site_id', 'Location']].drop_duplicates()
            sites = sites.sort_values('site_id')
            
            # Store the combo boxes in a list to prevent them from being garbage collected
            self.combo_boxes = []
            
            # Initialize route tab sites
            if hasattr(self, 'origin_combo'):
                self.origin_combo.clear()
                self.combo_boxes.append(self.origin_combo)
            for _, row in sites.iterrows():
                display_text = f"{row['site_id']} - {row['Location']}"
                self.origin_combo.addItem(display_text)
            if self.origin_combo.count() > 0:
                self.origin_combo.setCurrentIndex(0)
            
            if hasattr(self, 'dest_combo'):
                self.dest_combo.clear()
                self.combo_boxes.append(self.dest_combo)
                for _, row in sites.iterrows():
                    display_text = f"{row['site_id']} - {row['Location']}"
                    self.dest_combo.addItem(display_text)
            if self.dest_combo.count() > 0:
                self.dest_combo.setCurrentIndex(0)
            
            # Populate waypoints list
            if hasattr(self, 'waypoints_list'):
                self.waypoints_list.clear()
                for _, row in self.df[['site_id', 'Location']].drop_duplicates().sort_values('site_id').iterrows():
                    item = QListWidgetItem(f"{row['site_id']} - {row['Location']}")
                    item.setData(Qt.UserRole, str(row['site_id']))
                    self.waypoints_list.addItem(item)
                
        except Exception as e:
            self.logger.error(f"Error initializing sites: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error initializing sites: {str(e)}")

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
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar (full black theme)
        sidebar = QFrame()
        sidebar.setStyleSheet("""
            QFrame { background: #181c20; border-radius: 16px; }
            QLabel#title { font-size: 24px; font-weight: bold; color: #fff; margin-bottom: 18px; }
            QPushButton { background: #1677ff; color: #fff; font-size: 17px; font-weight: bold; border-radius: 8px; padding: 12px 0; }
            QPushButton:hover { background: #409eff; }
            QLineEdit, QComboBox, QDateEdit, QTimeEdit { font-size: 14px; border-radius: 8px; padding: 6px 8px; border: 1.5px solid #23272b; background: #23272b; color: #fff; min-height: 28px; max-height: 32px; }
            QLabel#legend { color: #bbb; font-size: 16px; margin-top: 18px; }
        """)
        sidebar.setFixedWidth(390)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(28, 28, 28, 28)
        sidebar_layout.setSpacing(18)

        # Title
        title = QLabel("Route Request")
        title.setObjectName("title")
        sidebar_layout.addWidget(title)

        # Origin site dropdown
        self.origin_combo = QComboBox()
        self.origin_combo.setEditable(False)
        self.origin_combo.setInsertPolicy(QComboBox.NoInsert)
        self.origin_combo.setMinimumHeight(36)
        self.origin_combo.setStyleSheet("QComboBox { background: #23272b; color: #fff; }")
        site_options = [f"{row['site_id']} - {row['Location']}" for _, row in self.df[['site_id', 'Location']].drop_duplicates().sort_values('site_id').iterrows()]
        self.origin_combo.addItems(site_options)
        sidebar_layout.addWidget(self.origin_combo)

        # Destination site dropdown
        self.dest_combo = QComboBox()
        self.dest_combo.setEditable(False)
        self.dest_combo.setInsertPolicy(QComboBox.NoInsert)
        self.dest_combo.setMinimumHeight(36)
        self.dest_combo.setStyleSheet("QComboBox { background: #23272b; color: #fff; }")
        self.dest_combo.addItems(site_options)
        sidebar_layout.addWidget(self.dest_combo)

        # Model dropdown: LSTM, GRU, CNN
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "GRU", "CNN"])
        sidebar_layout.addWidget(self.model_combo)

        # Algorithm dropdown: A*, BFS, DFS, GBFS, CUS1, CUS2
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["A*", "BFS", "DFS", "GBFS", "CUS1", "CUS2"])
        sidebar_layout.addWidget(self.algo_combo)

        # Date and Time inputs
        self.date_input = QDateEdit()
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        sidebar_layout.addWidget(self.date_input)

        self.time_input = QTimeEdit()
        self.time_input.setDisplayFormat("HH:mm")
        self.time_input.setTime(QTime.currentTime())
        sidebar_layout.addWidget(self.time_input)

        # Search Button
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._find_routes)
        sidebar_layout.addWidget(search_btn)

        # Results area (dark card style)
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_content = QWidget()
        self.results_layout = QVBoxLayout(self.results_content)
        self.results_content.setLayout(self.results_layout)
        self.results_area.setWidget(self.results_content)
        self.results_area.setStyleSheet("QScrollArea { background: #181c20; border: none; }")
        sidebar_layout.addWidget(self.results_area, 1)

        main_layout.addWidget(sidebar)

        # Map area (make sure it's initialized and shown)
        if not hasattr(self, 'map_view'):
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            self.map_view = QWebEngineView()
        main_layout.addWidget(self.map_view, 1)

        # Always initialize the map!
        self._initialize_map()

        return main_widget
        
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
                    popup=f"Location: {row['Location']}",
                    icon=folium.Icon(color='gray')
                ).add_to(m)
            
            # Save and display map
            data = io.BytesIO()
            m.save(data, close_file=False)
            self.map_view.setHtml(data.getvalue().decode())
            
        except Exception as e:
            self.logger.error(f"Error initializing map: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error initializing map: {str(e)}")
        
    def _create_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # Theme selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)

        return widget

    def apply_theme(self, theme_name):
        """Apply the selected theme to the application."""
        if theme_name == "Dark":
            dark_stylesheet = """
                QWidget {
                    background-color: #232629;
                    color: #f0f0f0;
                    font-family: 'Arial', 'Helvetica', 'sans-serif';
                    font-size: 15px;
                }
                QFrame[card="true"] {
                    background: #232b33;
                    border-radius: 18px;
                    border: 1.5px solid #888;
                    margin: 0 0 18px 0;
                }
                QLabel[heading="true"] {
                    font-size: 22px;
                    font-weight: bold;
                    color: #888;
                    margin-bottom: 10px;
                }
                QLineEdit, QComboBox, QSpinBox, QTableWidget, QTabWidget::pane, QProgressBar {
                    background-color: #31363b;
                    color: #f0f0f0;
                    border-radius: 8px;
                    border: 1.5px solid #888;
                    padding: 6px 12px;
                    font-size: 16px;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 2px solid #888;
                    background-color: #232b33;
                }
                QComboBox::down-arrow {
                    image: url(/Users/yaxzyra/Documents/University/Introduction_to_AI/Traffic_Assignment/resources/arrow-down.svg);
                    width: 16px;
                    height: 16px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 32px;
                    border-left-width: 1px;
                    border-left-color: #888;
                    border-left-style: solid;
                    border-top-right-radius: 8px;
                    border-bottom-right-radius: 8px;
                }
                QSpinBox::up-arrow, QSpinBox::down-arrow {
                    width: 18px;
                    height: 18px;
                }
                QSpinBox::up-button, QSpinBox::down-button {
                    min-width: 20px;
                    min-height: 20px;
                    width: 20px;
                    height: 20px;
                    margin: 1px;
                    padding: 0px;
                }
                QSpinBox::up-arrow {
                    image: url(/Users/yaxzyra/Documents/University/Introduction_to_AI/Traffic_Assignment/resources/arrow-up.svg);
                }
                QSpinBox::down-arrow {
                    image: url(/Users/yaxzyra/Documents/University/Introduction_to_AI/Traffic_Assignment/resources/arrow-down.svg);
                }
                QPushButton {
                    background-color: #888;
                    color: #fff;
                    font-size: 17px;
                    font-weight: 600;
                    border-radius: 10px;
                    padding: 10px 0;
                    margin-top: 8px;
                    border: none;
                    transition: background 0.2s;
                }
                QPushButton:hover {
                    background-color: #aaa;
                }
                QPushButton:pressed {
                    background-color: #555;
                }
                QTabBar::tab {
                    background: #31363b;
                    color: #f0f0f0;
                    border-radius: 8px;
                    padding: 10px 22px;
                    font-size: 16px;
                    margin: 2px;
                    min-width: 120px;
                    margin-bottom: 20px;
                }
                QTabBar::tab:selected {
                    background: #888;
                    color: #232629;
                }
                QTabWidget::pane {
                    border-radius: 12px;
                    border: 1.5px solid #888;
                    padding: 8px;
                }
                QProgressBar {
                    border: 1.5px solid #888;
                    border-radius: 8px;
                    text-align: center;
                    height: 18px;
                    font-size: 15px;
                    background: #232b33;
                    color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #888;
                    border-radius: 8px;
                }
                QTextEdit {
                    background: #232b33;
                    color: #f0f0f0;
                    border-radius: 10px;
                    font-size: 16px;
                    padding: 18px;
                    margin: 10px;
                    border: 1.5px solid #888;
                }
            """
            self.setStyleSheet(dark_stylesheet)
            # Update route summary style for dark theme
            if hasattr(self, 'route_summary'):
                self.route_summary.setStyleSheet("QTextEdit { background: #232629; color: #f0f0f0; border-radius: 8px; font-size: 15px; padding: 12px; margin: 8px; border: none; }")
            # Update prediction summary style for dark theme
            if hasattr(self, 'prediction_summary'):
                self.prediction_summary.setStyleSheet("QTextEdit { background: #232629; color: #f0f0f0; border-radius: 8px; font-size: 15px; padding: 12px; margin: 8px; border: none; }")
        else:
            light_stylesheet = """
                QWidget {
                    background-color: #f5f6fa;
                    color: #232629;
                    font-family: 'Arial', 'Helvetica', 'sans-serif';
                    font-size: 15px;
                }
                QFrame[card="true"] {
                    background: #fff;
                    border-radius: 18px;
                    border: 1.5px solid #888;
                    margin: 0 0 18px 0;
                }
                QLabel[heading="true"] {
                    font-size: 22px;
                    font-weight: bold;
                    color: #888;
                    margin-bottom: 10px;
                }
                QLineEdit, QComboBox, QSpinBox, QTableWidget, QTabWidget::pane, QProgressBar {
                    background-color: #fff;
                    color: #232629;
                    border-radius: 8px;
                    border: 1.5px solid #888;
                    padding: 6px 12px;
                    font-size: 16px;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 2px solid #888;
                    background-color: #eaf6fb;
                }
                QComboBox::down-arrow {
                    image: url(/Users/yaxzyra/Documents/University/Introduction_to_AI/Traffic_Assignment/resources/arrow-down.svg);
                    width: 16px;
                    height: 16px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 32px;
                    border-left-width: 1px;
                    border-left-color: #888;
                    border-left-style: solid;
                    border-top-right-radius: 8px;
                    border-bottom-right-radius: 8px;
                }
                QSpinBox::up-arrow, QSpinBox::down-arrow {
                    width: 18px;
                    height: 18px;
                }
                QSpinBox::up-button, QSpinBox::down-button {
                    min-width: 20px;
                    min-height: 20px;
                    width: 20px;
                    height: 20px;
                    margin: 1px;
                    padding: 0px;
                }
                QSpinBox::up-arrow {
                    image: url(/Users/yaxzyra/Documents/University/Introduction_to_AI/Traffic_Assignment/resources/arrow-up.svg);
                }
                QSpinBox::down-arrow {
                    image: url(/Users/yaxzyra/Documents/University/Introduction_to_AI/Traffic_Assignment/resources/arrow-down.svg);
                }
                QPushButton {
                    background-color: #888;
                    color: #fff;
                    font-size: 17px;
                    font-weight: 600;
                    border-radius: 10px;
                    padding: 10px 0;
                    margin-top: 8px;
                    border: none;
                    transition: background 0.2s;
                }
                QPushButton:hover {
                    background-color: #aaa;
                }
                QPushButton:pressed {
                    background-color: #555;
                }
                QTabBar::tab {
                    background: #fff;
                    color: #232629;
                    border-radius: 8px;
                    padding: 10px 22px;
                    font-size: 16px;
                    margin: 2px;
                    min-width: 120px;
                    margin-bottom: 20px;
                }
                QTabBar::tab:selected {
                    background: #888;
                    color: #232629;
                }
                QTabWidget::pane {
                    border-radius: 12px;
                    border: 1.5px solid #888;
                    padding: 8px;
                }
                QProgressBar {
                    border: 1.5px solid #888;
                    border-radius: 8px;
                    text-align: center;
                    height: 18px;
                    font-size: 15px;
                    background: #eaf6fb;
                    color: #232629;
                }
                QProgressBar::chunk {
                    background-color: #888;
                    border-radius: 8px;
                }
                QTextEdit {
                    background: #fff;
                    color: #232629;
                    border-radius: 10px;
                    font-size: 16px;
                    padding: 18px;
                    margin: 10px;
                    border: 1.5px solid #888;
                }
            """
            self.setStyleSheet(light_stylesheet)
            # Update route summary style for light theme
            if hasattr(self, 'route_summary'):
                self.route_summary.setStyleSheet("QTextEdit { background: #f5f6fa; color: #232629; border-radius: 8px; font-size: 15px; padding: 12px; margin: 8px; border: none; }")
            # Update prediction summary style for light theme
            if hasattr(self, 'prediction_summary'):
                self.prediction_summary.setStyleSheet("QTextEdit { background: #f5f6fa; color: #232629; border-radius: 8px; font-size: 15px; padding: 12px; margin: 8px; border: none; }")

    def get_road_polyline(self, origin_coords, dest_coords):
        """Get a road-following polyline between two coordinates using OpenRouteService, with timeout, robust fallback, and caching."""
        key = (round(origin_coords[0], 6), round(origin_coords[1], 6), round(dest_coords[0], 6), round(dest_coords[1], 6))
        if key in self.ors_cache:
            return self.ors_cache[key]
        try:
            import openrouteservice
            import requests
            ORS_API_KEY = '5b3ce3597851110001cf62482f7615690eae478583e68e57c8f1143f'
            client = openrouteservice.Client(key=ORS_API_KEY, timeout=5)
            coords = ((origin_coords[1], origin_coords[0]), (dest_coords[1], dest_coords[0]))  # (lon, lat)
            try:
                route = client.directions(coords, profile='driving-car', format='geojson')
                road_coords = route['features'][0]['geometry']['coordinates']
                road_coords_latlon = [(lat, lon) for lon, lat in road_coords]
                self.ors_cache[key] = road_coords_latlon
                return road_coords_latlon
            except (openrouteservice.exceptions.ApiError, requests.exceptions.Timeout, Exception) as e:
                print(f"ORS API error or timeout for segment {origin_coords} -> {dest_coords}: {e}")
                self.ors_cache[key] = [(origin_coords[0], origin_coords[1]), (dest_coords[0], dest_coords[1])]
                return self.ors_cache[key]
        except Exception as e:
            print(f"Error setting up ORS client or unknown error: {e}")
            self.ors_cache[key] = [(origin_coords[0], origin_coords[1]), (dest_coords[0], dest_coords[1])]
            return self.ors_cache[key]

    def _find_routes(self):
        from PyQt5.QtWidgets import QProgressDialog, QApplication
        progress = QProgressDialog("Calculating route...", None, 0, 0, self)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        try:
            origin_text = self.origin_combo.currentText()
            dest_text = self.dest_combo.currentText()
            origin_id = origin_text.split(' - ')[0]
            dest_id = dest_text.split(' - ')[0]
            print(f"Finding route from {origin_id} to {dest_id}")  # Debug log
            date_str = self.date_input.date().toString("yyyy-MM-dd")
            time_str = self.time_input.time().toString("HH:mm")
            model_name = self.model_combo.currentText()
            algo = self.algo_combo.currentText()
            graph = self.real_graph
            if not graph:
                print("Error: Graph is empty")
                QMessageBox.warning(self, "Error", "Graph is not properly initialized.")
                return
            print(f"Graph nodes: {list(graph.keys())}")
            print(f"Origin node connections: {graph.get(origin_id, [])}")
            print(f"Destination node connections: {graph.get(dest_id, [])}")

            # --- Routing: Use selected algorithm to generate up to 5 unique routes ---
            from route_models.astar_search import astar
            from route_models.bfs_search import bfs
            from route_models.dfs_search import dfs
            from route_models.gbfs_search import gbfs
            from route_models.cus1_search import cus1_search
            from route_models.cus2_search import cus2_search

            def get_k_unique_routes(algo, graph, origin, dest, k=5):
                routes = []
                blocked_edges = set()
                max_attempts = 30  # Try more times to find alternatives
                attempts = 0
                while len(routes) < k and attempts < max_attempts:
                    mod_graph = {node: [(nbr, w) for nbr, w in nbrs if (node, nbr) not in blocked_edges] for node, nbrs in graph.items()}
                    path = []
                    if algo == "A*":
                        path, _ = astar(mod_graph, origin, dest)
                    elif algo == "BFS":
                        path = bfs(mod_graph, origin, [dest])
                    elif algo == "DFS":
                        path = dfs(mod_graph, origin, dest)
                    elif algo == "GBFS":
                        path, _ = gbfs(mod_graph, origin, dest)
                    elif algo == "CUS1":
                        path = cus1_search(mod_graph, origin, dest)
                    elif algo == "CUS2":
                        _, _, _, path = cus2_search(mod_graph, origin, dest)
                    print(f"Attempt {len(routes) + 1}: Found path: {path}")
                    if not path or len(path) < 2:
                        break
                    # Only check for exact duplicate paths
                    if all(path != r[0] for r in routes):
                        routes.append((path, 0))
                        # Block all edges in this path (except start/end)
                        for i in range(1, len(path)):
                            blocked_edges.add((path[i-1], path[i]))
                    attempts += 1
                return routes[:k]

            routes = get_k_unique_routes(algo, graph, origin_id, dest_id, k=5)
            if not routes or len(routes) < 1:
                print("No routes found")
                QMessageBox.warning(self, "Error", "No route found.")
                return
            print(f"Found {len(routes)} routes")
            def flow_to_speed(flow):
                if flow <= 351:
                    return 60.0
                a = -1.4648375
                b = 93.75
                c = -flow
                discriminant = b**2 - 4*a*c
                if discriminant < 0:
                    return 10.0
                sqrt_disc = discriminant ** 0.5
                speed1 = (-b + sqrt_disc) / (2*a)
                speed2 = (-b - sqrt_disc) / (2*a)
                speed = min(speed1, speed2)
                return max(speed, 5.0)
            def predict_flow(site_id, model_name, date_str, time_str):
                site_data = self.df[self.df['site_id'].astype(str) == str(site_id)]
                time_columns = [col for col in self.df.columns if (':' in str(col)) or (str(col).startswith('V') and str(col)[1:].isdigit() and len(str(col)) == 3)]
                if len(site_data) == 0 or len(time_columns) == 0:
                    return 200.0
                return site_data[time_columns].mean(axis=1).mean()
            intersection_delay = 30  # seconds
            route_infos = []
            for path, _ in routes:
                total_time = 0.0
                total_distance = 0.0
                for j in range(len(path) - 1):
                    a = self.df[self.df['site_id'].astype(str) == str(path[j])].iloc[0]
                    b = self.df[self.df['site_id'].astype(str) == str(path[j+1])].iloc[0]
                    lat1, lon1 = a['NB_LATITUDE'], a['NB_LONGITUDE']
                    lat2, lon2 = b['NB_LATITUDE'], b['NB_LONGITUDE']
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371.0
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    aa = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(aa), sqrt(1-aa))
                    distance = R * c
                    total_distance += distance
                    flow = predict_flow(path[j], model_name, date_str, time_str)
                    speed = flow_to_speed(flow)
                    if speed > 0:
                        time_hr = distance / speed
                    else:
                        time_hr = distance / 10.0
                    time_hr += intersection_delay / 3600.0
                    total_time += time_hr
                route_infos.append((path, total_time * 60, total_distance))  # time in minutes
            import folium
            map_center = [self.df[self.df['site_id'].astype(str) == str(origin_id)].iloc[0]['NB_LATITUDE'],
                         self.df[self.df['site_id'].astype(str) == str(origin_id)].iloc[0]['NB_LONGITUDE']]
            m = folium.Map(location=map_center, zoom_start=13)
            route_styles = [
                {"color": "#FF4136", "width": 2},    # 1st: red
                {"color": "#e67e22", "width": 4},    # 2nd: orange
                {"color": "#27ae60", "width": 6},    # 3rd: green
                {"color": "#8e44ad", "width": 8},    # 4th: purple
                {"color": "#0074D9", "width": 10},   # 5th: blue
            ]
            for idx in range(len(route_infos)-1, -1, -1):  # Draw slowest on top, fastest on bottom
                path, travel_time, distance = route_infos[idx]
                style = route_styles[idx % len(route_styles)]
                full_coords = []
                for j in range(len(path) - 1):
                    a = self.df[self.df['site_id'].astype(str) == str(path[j])].iloc[0]
                    b = self.df[self.df['site_id'].astype(str) == str(path[j+1])].iloc[0]
                    lat1, lon1 = a['NB_LATITUDE'], a['NB_LONGITUDE']
                    lat2, lon2 = b['NB_LATITUDE'], b['NB_LONGITUDE']
                    # Use ORS for real road-following polyline
                    seg_coords = self.get_road_polyline((lat1, lon1), (lat2, lon2))
                    if not full_coords:
                        full_coords.extend(seg_coords)
                    else:
                        # Avoid duplicating the first point of the segment
                        full_coords.extend(seg_coords[1:])
                folium.PolyLine(full_coords, color=style["color"], weight=style["width"], opacity=0.85).add_to(m)
                if full_coords:
                    folium.Marker(
                        full_coords[0],
                        tooltip="Origin",
                        icon=folium.Icon(color='green', icon='play', prefix='fa')
                    ).add_to(m)
                    folium.Marker(
                        full_coords[-1],
                        tooltip="Destination",
                        icon=folium.Icon(color='red', icon='stop', prefix='fa')
                    ).add_to(m)
            data = io.BytesIO()
            m.save(data, close_file=False)
            self.map_view.setHtml(data.getvalue().decode())
            # --- Sidebar Results ---
            for i in reversed(range(self.results_layout.count())):
                widget = self.results_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            header = QLabel(f"<b>Routes from {origin_id} to {dest_id}</b>")
            header.setStyleSheet("font-size: 22px; margin-bottom: 18px; color: #fff;")
            self.results_layout.addWidget(header)
            route_icons = [
                '<span style="display:inline-block;width:22px;height:22px;border-radius:50%;background:#FF4136;margin-left:12px;vertical-align:middle;"></span>',
                '<span style="display:inline-block;width:22px;height:22px;border-radius:50%;background:#e67e22;margin-left:12px;vertical-align:middle;"></span>',
                '<span style="display:inline-block;width:22px;height:22px;border-radius:50%;background:#27ae60;margin-left:12px;vertical-align:middle;"></span>',
                '<span style="display:inline-block;width:22px;height:22px;border-radius:50%;background:#8e44ad;margin-left:12px;vertical-align:middle;"></span>',
                '<span style="display:inline-block;width:22px;height:22px;border-radius:50%;background:#0074D9;margin-left:12px;vertical-align:middle;"></span>'
            ]
            for idx, (path, travel_time, distance) in enumerate(route_infos):
                icon = route_icons[idx % len(route_icons)]
                card_html = f"""
                <div style='background:#23272b; border-radius:18px; margin-bottom:22px; padding:28px 32px; color:#fff; font-size:22px;'>
                  <b style='font-size:26px; color:#fff;'>
                    Route {idx+1} <span style='vertical-align:middle;' title='Route color'>{icon}</span>
                  </b><br>
                  <span style='font-size:21px; color:#bbb;'><b>Time:</b> {travel_time:.2f} min</span><br>
                  <span style='font-size:21px; color:#bbb;'><b>Distance:</b> {distance:.2f} km</span><br>
                  <span style='font-size:18px; color:#888;'><b>Path:</b> {' → '.join(str(s) for s in path)}</span>
                </div>
                """
                card = QLabel(card_html)
                card.setTextFormat(Qt.RichText)
                card.setWordWrap(True)
                card.setStyleSheet("font-size: 22px; margin-bottom: 18px; background: transparent;")
                self.results_layout.addWidget(card)
            self.results_layout.addStretch(1)
            progress.close()
        except Exception as e:
            progress.close()
            error_msg = f"Error finding routes: {str(e)}"
            print(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
            return

    def get_road_route(self, origin_coords, dest_coords):
        """Get a road-following route between two coordinates using OpenRouteService."""
        try:
            ORS_API_KEY = '5b3ce3597851110001cf62482f7615690eae478583e68e57c8f1143f'
            client = openrouteservice.Client(key=ORS_API_KEY)
            coords = ((origin_coords[1], origin_coords[0]), (dest_coords[1], dest_coords[0]))  # (lon, lat)
            route = client.directions(coords, profile='driving-car', format='geojson')
            road_coords = route['features'][0]['geometry']['coordinates']
            road_coords_latlon = [(lat, lon) for lon, lat in road_coords]
            return road_coords_latlon
        except Exception as e:
            print(f"Error fetching road route: {e}")
            return None

    def _predict_traffic(self):
        site = self.site_combo.currentText()
        if not site:
            QMessageBox.warning(self, "Error", "Please select a site")
            return
        try:
            site_id = site.split(' - ')[0]
            horizon_hours = self.horizon_spin.value()
            if horizon_hours > 3:
                horizon_hours = 3
                self.horizon_spin.setValue(3)
            steps = horizon_hours * 4  # 15-min intervals

            # Update UI: show 'Model is running...'
            self.prediction_label.setText("<b>Model is running...</b>")
            self.prediction_label.setStyleSheet(
                "QLabel { background: #232629; color: #fff; border: none; border-radius: 16px; font-size: 16px; padding: 0; }"
            )
            self.training_progress_bar.setVisible(False)
            self.predict_btn.setVisible(False)
            self.cancel_btn.setVisible(False)
            self._cancel_requested = False

            # Run prediction in a separate thread
            threading.Thread(target=self.predict_traffic_after_training, args=(site_id, steps)).start()
        except Exception as e:
            self.logger.error(f"Error predicting traffic: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error predicting traffic: {str(e)}")
            self.training_progress_bar.setVisible(False)
            self.cancel_btn.setVisible(False)
            self.predict_btn.setVisible(True)

    def _cancel_training(self):
        self._cancel_requested = True

    def build_real_graph(self, edges):
        """Build a graph representation from the edges list.
        
        Args:
            edges (list): List of (origin, dest, weight) tuples.
            
        Returns:
            dict: Graph representation where each node maps to a list of (neighbor, weight) tuples.
        """
        graph = defaultdict(list)
        for origin, dest, weight in edges:
            # Convert to strings and add both directions for undirected graph
            origin_str = str(origin)
            dest_str = str(dest)
            graph[origin_str].append((dest_str, weight))
            graph[dest_str].append((origin_str, weight))  # Add reverse edge
        return dict(graph)

    def offset_coords(self, coords, offset_meters):
        # Approximate conversion: 1 deg latitude ~ 111,320 meters
        offset_deg = offset_meters / 111320.0
        offset_coords = []
        for i in range(len(coords)):
            if i == 0 and len(coords) > 1:
                dx = coords[i+1][1] - coords[i][1]
                dy = coords[i+1][0] - coords[i][0]
            elif i > 0:
                dx = coords[i][1] - coords[i-1][1]
                dy = coords[i][0] - coords[i-1][0]
            else:
                dx, dy = 0, 0
            length = math.hypot(dx, dy)
            if length == 0:
                ox, oy = 0, 0
            else:
                ox = -dy / length * offset_deg
                oy = dx / length * offset_deg
            offset_coords.append((coords[i][0] + ox, coords[i][1] + oy))
        return offset_coords

    def find_route_with_waypoints(self, waypoints):
        """Find a route that visits all waypoints in order using the current graph."""
        full_route = []
        total_cost = 0
        for i in range(len(waypoints) - 1):
            segment, cost = self.route_finder.find_top_k_routes(waypoints[i], waypoints[i+1], k=1)[0]
            if i > 0:
                segment = segment[1:]  # Avoid duplicate nodes
            full_route.extend(segment)
            total_cost += cost
        return full_route, total_cost

    # Example usage (for demonstration):
    def demo_waypoint_route(self):
        # Hardcoded waypoints for demonstration (replace with UI selection as needed)
        waypoints = ['4057', '4032', '4051']
        route, cost = self.find_route_with_waypoints(waypoints)
        print(f"Waypoint route: {' -> '.join(route)} (Total cost: {cost:.2f} min)")
        # You can add code here to visualize this route on the map, etc.

    def predict_traffic_after_training(self, site_id, steps):
        """
        Use the selected pre-trained model to predict and update the GUI.
        Handles all file checks, data prep, model training, and plotting.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            from PyQt5.QtGui import QPixmap
            import numpy as np
            import pandas as pd
            import os
            import matplotlib.dates as mdates

            model_name = self.model_combo.currentText()
            window_size = 96  # Match the model's training window size

            # 1. Ensure raw data exists
            raw_path = f"data/scats_{site_id}.csv"
            if not os.path.exists(raw_path):
                # Try to generate it from the Excel file
                try:
                    from generate_normalized_data import export_site_csvs
                    export_site_csvs()  # This will create all missing raw files
                except Exception as e:
                    self.prediction_result_signal.emit(None, f"Error: Could not generate raw data for site {site_id}: {str(e)}")
                    return
                if not os.path.exists(raw_path):
                    self.prediction_result_signal.emit(None, f"Error: Raw data for site {site_id} not found after export.")
                return
            
            # 2. Ensure normalized data exists
            normalized_path = f"data/scats_{site_id}_normalized.csv"
            if not os.path.exists(normalized_path):
                try:
                    from generate_normalized_data import normalize_scats_file
                    normalize_scats_file(raw_path, normalized_path)
                except Exception as e:
                    self.prediction_result_signal.emit(None, f"Error: Could not normalize data for site {site_id}: {str(e)}")
                    return

            # 3. Prepare input for prediction
            input_path = normalized_path
            df = pd.read_csv(input_path)
            data = df['normalized_volume'].values
            last_window = data[-window_size:].reshape(1, window_size, 1)

            # 4. Ensure model exists, train if needed
            if model_name == "LSTM":
                from lstm_model import train_lstm_model
                model_path = f"models/lstm_site_{site_id}.h5"
            elif model_name == "GRU":
                from gru_model import train_gru_model
                model_path = f"models/gru_site_{site_id}.h5"
            elif model_name == "CNN":
                from cnn_model import train_cnn_model
                model_path = f"models/cnn_site_{site_id}.h5"
            else:
                self.prediction_result_signal.emit(None, f"Unknown model: {model_name}")
                return
            
            if not os.path.exists(model_path):
                # Show a message while training
                self.prediction_result_signal.emit(None, f"Training {model_name} model for site {site_id}... Please wait.")
                if model_name == "LSTM":
                    train_lstm_model(input_path, output_path=None, window_size=window_size, epochs=20)
                elif model_name == "GRU":
                    train_gru_model(input_path, output_path=None, window_size=window_size, epochs=20)
                elif model_name == "CNN":
                    train_cnn_model(input_path, output_path=None, window_size=window_size, epochs=20)

            # 5. Load model safely (avoid 'mse' error)
            from tensorflow.keras.models import load_model
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='mse')

            # 6. Predict sequence
            predictions = []
            current_input = last_window.copy()
            for _ in range(steps):
                pred = model.predict(current_input)
                predictions.append(pred[0, 0])
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = pred[0, 0]
            predictions = np.array(predictions)

            # 7. Inverse transform and get time
            from sklearn.preprocessing import MinMaxScaler
            df_original = pd.read_csv(raw_path)
            scaler = MinMaxScaler()
            scaler.fit(df_original[['Volume']])
            predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            context_steps = window_size  # Show last window_size context points

            # Use the actual time column if available, else generate a time range
            if 'Time' in df_original.columns:
                time_col = pd.to_datetime(df_original['Time'])
            else:
                # fallback: generate a time range starting at midnight
                time_col = pd.date_range("00:00", periods=len(df_original), freq="15T")

            context_times = time_col.iloc[-context_steps:]
            last_time = context_times.iloc[-1]
            future_times = pd.date_range(last_time + pd.Timedelta(minutes=15), periods=steps, freq='15T')

            # Ensure all time data is datetime
            context_times = pd.to_datetime(context_times)
            future_times = pd.to_datetime(future_times)

            plt.figure(figsize=(12, 5))
            plt.plot(context_times, context_data_inv[-context_steps:], label='True Flow', color='blue')
            plt.plot(future_times, predictions_inv, label=f'{model_name} Prediction', color='red', linestyle='--')
            plt.xlabel('Time of Day')
            plt.ylabel('Vehicles per Hour')
            plt.title(f'{model_name} Prediction for Site {site_id}')
            plt.legend()
            ax = plt.gca()
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))  # Minor ticks every 15 min
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))     # Major ticks/labels every hour
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', which='minor', labelbottom=False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.18)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            qimg = QPixmap()
            qimg.loadFromData(buf.getvalue())
            summary_text = f"<b>{model_name} Prediction complete for {steps} steps ({steps//4} hours)</b>"
            self.prediction_result_signal.emit(qimg, summary_text)
        except Exception as e:
            self.prediction_result_signal.emit(None, f"Error in prediction: {str(e)}")

    def _update_prediction_result(self, qimg, summary_text):
        if qimg is not None:
            self.prediction_label.setPixmap(qimg)
        else:
            self.prediction_label.setText(summary_text)
        self.prediction_summary.setText(summary_text)
        self.predict_btn.setVisible(True)