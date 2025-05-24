#105106819 Suman Sutparai
# Main window for TBRGS
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QTabWidget, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem, QSplitter, QProgressDialog, QApplication, QProgressBar, QFrame, QTextEdit, QSizePolicy, QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer, QThread
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
        self.title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #888; margin-bottom: 12px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabWidget::pane { border-radius: 12px; padding: 8px; }")
        self.layout.addWidget(self.tab_widget)
        
        # Initialize route finder
        self.route_finder = RouteFinder()
        if edges and coordinates:
            self.route_finder.build_graph(edges, coordinates)
            self.real_graph = self.build_real_graph(edges)
        
        # Initialize progress signal
        self.progress = TrainingProgress()
        
        # Create tabs
        self.tab_widget.addTab(self._create_route_tab(), "Route")
        self.tab_widget.addTab(self._create_prediction_tab(), "Prediction")
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
            
            # Initialize prediction tab sites
            if hasattr(self, 'site_combo'):
                self.site_combo.clear()
                self.combo_boxes.append(self.site_combo)
                for _, row in sites.iterrows():
                    display_text = f"{row['site_id']} - {row['Location']}"
                    self.site_combo.addItem(display_text)
                if self.site_combo.count() > 0:
                    self.site_combo.setCurrentIndex(0)
            
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
        widget = QWidget()
        widget.setStyleSheet("background: #232629;")
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # Card-like, dark, rounded controls area
        controls_frame = QFrame()
        controls_frame.setStyleSheet("QFrame { background: #232629; border-radius: 18px; padding: 32px 32px 24px 32px; }")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(18)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Origin selection (card style)
        origin_card = QFrame()
        origin_card.setStyleSheet("QFrame { background: #2d3238; border-radius: 16px; padding: 8px 24px; }")
        origin_layout = QHBoxLayout(origin_card)
        origin_layout.setContentsMargins(0, 0, 0, 0)
        origin_label = QLabel("Origin Site:")
        self.origin_combo = QComboBox()
        self.origin_combo.setMinimumHeight(36)
        origin_layout.addWidget(origin_label)
        origin_layout.addWidget(self.origin_combo)
        controls_layout.addWidget(origin_card)

        # Destination selection (card style)
        dest_card = QFrame()
        dest_card.setStyleSheet("QFrame { background: #2d3238; border-radius: 16px; padding: 8px 24px; }")
        dest_layout = QHBoxLayout(dest_card)
        dest_layout.setContentsMargins(0, 0, 0, 0)
        dest_label = QLabel("Destination Site:")
        self.dest_combo = QComboBox()
        self.dest_combo.setMinimumHeight(36)
        dest_layout.addWidget(dest_label)
        dest_layout.addWidget(self.dest_combo)
        controls_layout.addWidget(dest_card)

        # Find routes button (full width, bright, rounded)
        find_routes_btn = QPushButton("Find Routes")
        find_routes_btn.setMinimumHeight(54)
        find_routes_btn.setMaximumHeight(62)
        from PyQt5.QtWidgets import QSizePolicy
        find_routes_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        find_routes_btn.setStyleSheet("QPushButton { background-color: #888; color: #fff; font-size: 17px; font-weight: 600; border-radius: 10px; padding: 10px 0; margin-top: 18px; border: none; transition: background 0.2s; } QPushButton:hover { background-color: #aaa; } QPushButton:pressed { background-color: #555; }")
        find_routes_btn.clicked.connect(self._find_routes)
        controls_layout.addWidget(find_routes_btn)

        layout.addWidget(controls_frame)

        # Map view and route summary side by side
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter { background: #232629; }")
        self.map_view = QWebEngineView()
        splitter.addWidget(self.map_view)

        # Route summary card
        summary_card = QFrame()
        summary_card.setStyleSheet("QFrame { background: #232629; border-radius: 16px; border: 1.5px solid #888; padding: 0; }")
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        self.route_summary = QTextEdit()
        self.route_summary.setReadOnly(True)
        self.route_summary.setStyleSheet("QTextEdit { background: #232629; color: #f0f0f0; border-radius: 8px; font-size: 15px; padding: 12px; margin: 8px; border: none; }")
        self.route_summary.setMinimumWidth(320)
        self.route_summary.setMaximumWidth(480)
        self.route_summary.setHtml('<span style="color:#888;font-size:15px;">Route summary will appear here after you search for routes.</span>')
        summary_layout.addWidget(self.route_summary)
        splitter.addWidget(summary_card)
        splitter.setSizes([900, 350])
        layout.addWidget(splitter)

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
        
    def _create_prediction_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left: Controls panel
        controls_panel = QFrame()
        controls_panel.setStyleSheet("QFrame { background: transparent; }")
        controls_panel.setFixedWidth(380)
        controls_panel_layout = QVBoxLayout(controls_panel)
        controls_panel_layout.setContentsMargins(24, 4, 24, 24)
        controls_panel_layout.setSpacing(0)
        controls_panel_layout.addStretch(1)

        # Model selection dropdown
        model_card = QFrame()
        model_card.setStyleSheet("QFrame { background: #2d3238; border-radius: 16px; padding: 4px 24px 16px 24px; margin-top: 2px; }")
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(12)
        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-weight: 600; font-size: 17px; color: #f0f0f0; margin: 0 0 1px 0; padding-top: 1px; padding-bottom: 1px;")
        model_label.setAlignment(Qt.AlignCenter)
        model_label.setWordWrap(True)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(36)
        self.model_combo.setStyleSheet(
            "QComboBox { background: #232629; color: #f0f0f0; border-radius: 8px; border: 1.5px solid #888; padding: 6px 12px; font-size: 16px; }"
            "QComboBox::drop-down { border: none; background: transparent; }"
            "QComboBox QAbstractItemView { background: #232629; color: #f0f0f0; border-radius: 8px; }"
        )
        self.model_combo.addItems(["LSTM", "GRU", "CNN"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        controls_panel_layout.addWidget(model_card, alignment=Qt.AlignTop)

        # Site selection
        site_card = QFrame()
        site_card.setStyleSheet("QFrame { background: #2d3238; border-radius: 16px; padding: 4px 24px 16px 24px; margin-top: 2px; }")
        site_layout = QVBoxLayout(site_card)
        site_layout.setContentsMargins(0, 0, 0, 0)
        site_layout.setSpacing(12)
        site_label = QLabel("Site:")
        site_label.setStyleSheet("font-weight: 600; font-size: 17px; color: #f0f0f0; margin: 0 0 1px 0; padding-top: 1px; padding-bottom: 1px;")
        site_label.setAlignment(Qt.AlignCenter)
        site_label.setWordWrap(True)
        self.site_combo = QComboBox()
        self.site_combo.setMinimumHeight(36)
        self.site_combo.setStyleSheet(
            "QComboBox { background: #232629; color: #f0f0f0; border-radius: 8px; border: 1.5px solid #888; padding: 6px 12px; font-size: 16px; }"
            "QComboBox::drop-down { border: none; background: transparent; }"
            "QComboBox QAbstractItemView { background: #232629; color: #f0f0f0; border-radius: 8px; }"
        )
        site_layout.addWidget(site_label)
        site_layout.addWidget(self.site_combo)
        controls_panel_layout.addWidget(site_card, alignment=Qt.AlignTop)

        # Prediction horizonye
        horizon_card = QFrame()
        horizon_card.setStyleSheet("QFrame { background: #2d3238; border-radius: 16px; padding: 12px 12px 24px 12px; }")
        horizon_layout = QVBoxLayout(horizon_card)
        horizon_layout.setContentsMargins(0, 0, 0, 0)
        horizon_layout.setSpacing(8)
        horizon_label = QLabel("Prediction Horizon (hours):")
        horizon_label.setStyleSheet("font-weight: 600; font-size: 16px; color: #f0f0f0; margin: 0 0 8px 0; padding-top: 1px; padding-bottom: 1px;")
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 3)  # Only allow 1, 2, or 3 hours
        self.horizon_spin.setValue(1)
        self.horizon_spin.setMinimumHeight(38)
        self.horizon_spin.setStyleSheet(
            "QSpinBox { background: #232629; color: #f0f0f0; border-radius: 8px; border: 1.5px solid #888; padding: 6px 12px; font-size: 16px; }"
        )
        horizon_layout.addWidget(horizon_label)
        horizon_layout.addWidget(self.horizon_spin)
        controls_panel_layout.addWidget(horizon_card, alignment=Qt.AlignTop)

        # Predict button (in controls card)
        self.predict_btn = QPushButton("Predict Traffic")
        self.predict_btn.setMinimumHeight(54)
        self.predict_btn.setMinimumWidth(0)
        self.predict_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.predict_btn.clicked.connect(self._predict_traffic)
        controls_panel_layout.addWidget(self.predict_btn)

        # Cancel button (in controls card, hidden by default)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(54)
        self.cancel_btn.setMinimumWidth(0)
        self.cancel_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background: #c00; color: #fff; border-radius: 10px; font-size: 17px; font-weight: 600; padding: 10px 0; border: none; }"
            "QPushButton:hover { background: #e33; }"
        )
        self.cancel_btn.clicked.connect(self._cancel_training)
        self.cancel_btn.setVisible(False)
        controls_panel_layout.addWidget(self.cancel_btn)

        controls_panel_layout.addStretch(1)
        controls_panel_layout.addStretch(2)

        # Right: Graph and summary
        right_panel = QFrame()
        right_panel.setStyleSheet("QFrame { background: transparent; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 32, 32, 32)

        # Graph card
        graph_card = QFrame()
        graph_card.setStyleSheet(
            "QFrame { background: #232629; border-radius: 16px; border: 1.5px solid #888; padding: 0; margin-bottom: 0; }"
        )
        graph_layout = QVBoxLayout(graph_card)
        graph_layout.setContentsMargins(8, 8, 8, 8)
        graph_layout.setSpacing(0)
        self.prediction_label = QLabel()
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setMinimumHeight(320)
        self.prediction_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.prediction_label.setScaledContents(True)
        self.prediction_label.setStyleSheet(
            "QLabel { background: #232629; color: #b0b0b0; border: none; border-radius: 16px; font-size: 18px; padding: 0; }"
        )
        self.prediction_label.setText('<div style="color:#888;font-size:18px;line-height:1.6;">\n<span style="font-size:48px;">📈</span><br>Click "Predict Traffic" to generate a prediction.</div>')
        graph_layout.addWidget(self.prediction_label)
        right_layout.addWidget(graph_card)

        # Progress bar and Cancel button (between graph and summary)
        progress_cancel_layout = QHBoxLayout()
        progress_cancel_layout.setContentsMargins(0, 0, 0, 0)
        progress_cancel_layout.setSpacing(8)
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setMinimum(0)
        self.training_progress_bar.setMaximum(100)
        self.training_progress_bar.setValue(0)
        self.training_progress_bar.setTextVisible(False)
        self.training_progress_bar.setVisible(False)
        self.training_progress_bar.setMinimumHeight(2)
        self.training_progress_bar.setMaximumHeight(2)
        self.training_progress_bar.setStyleSheet(
            "QProgressBar { border: none; border-radius: 1px; height: 2px; background: #232629; margin: 0 8px 0 8px; }"
            "QProgressBar::chunk { background-color: #fff; border-radius: 1px; }"
        )
        progress_cancel_layout.addWidget(self.training_progress_bar)
        right_layout.addLayout(progress_cancel_layout)

        # Summary card
        summary_card = QFrame()
        summary_card.setStyleSheet(
            "QFrame { background: #232629; border-radius: 16px; border: 1.5px solid #888; padding: 0; }"
        )
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        self.prediction_summary = QTextEdit()
        self.prediction_summary.setReadOnly(True)
        self.prediction_summary.setStyleSheet(
            "QTextEdit { background: #232629; color: #f0f0f0; border-radius: 14px; font-size: 16px; padding: 24px; margin: 0; border: none; }"
        )
        self.prediction_summary.setMinimumHeight(200)
        self.prediction_summary.setHtml('<div style="color:#888;font-size:16px;">Prediction summary will appear here after you run a prediction.</div>')
        summary_layout.addWidget(self.prediction_summary)
        right_layout.addWidget(summary_card)

        layout.addWidget(controls_panel)
        layout.addWidget(right_panel)
        layout.setStretch(0, 3)
        layout.setStretch(1, 7)

        return widget
        
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

    def _find_routes(self):
        try:
            origin_text = self.origin_combo.currentText()
            dest_text = self.dest_combo.currentText()
            origin_id = origin_text.split(' - ')[0]
            dest_id = dest_text.split(' - ')[0]
            origin_data = self.df[self.df['site_id'].astype(str) == str(origin_id)]
            dest_data = self.df[self.df['site_id'].astype(str) == str(dest_id)]
            if len(origin_data) == 0 or len(dest_data) == 0:
                QMessageBox.warning(self, "Error", "No data found for origin or destination site.")
                self.route_summary.setText("")
                return
            origin_coords = (origin_data.iloc[0]['NB_LATITUDE'], origin_data.iloc[0]['NB_LONGITUDE'])
            dest_coords = (dest_data.iloc[0]['NB_LATITUDE'], dest_data.iloc[0]['NB_LONGITUDE'])
            progress = QProgressDialog("Calculating routes...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Route Calculation")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            progress.setLabelText("Analyzing traffic flow...")
            progress.setValue(10)
            QApplication.processEvents()
            time_columns = [col for col in self.df.columns if (':' in str(col)) or (str(col).startswith('V') and str(col)[1:].isdigit() and len(str(col)) == 3)]
            edge_flows = {}
            for u, edges in self.real_graph.items():
                for v, _ in edges:
                    u_data = self.df[self.df['site_id'].astype(str) == str(u)][time_columns].mean().mean()
                    v_data = self.df[self.df['site_id'].astype(str) == str(v)][time_columns].mean().mean()
                    edge_flows[(str(u), str(v))] = (u_data + v_data) / 2
            max_flow = max(edge_flows.values()) if edge_flows else 1
            edge_times = {edge: (flow / max_flow) * 30 for edge, flow in edge_flows.items()}
            # Call the new find_top_k_assignment_routes method from route_finder.py
            routes = self.route_finder.find_top_k_assignment_routes(origin_id, dest_id, k=3)
            fastest_route = routes[0] if len(routes) > 0 else None
            alternative_route = routes[1] if len(routes) > 1 else None
            if not fastest_route:
                QMessageBox.warning(self, "Error", "No route found.")
                self.route_summary.setText("")
                return
            # Visualization: red first (thinner), then blue (thicker) on top
            colors = ['#FF4136', '#0074D9']  # red, blue
            weights = [4, 8]
            m = folium.Map(location=origin_coords, zoom_start=12)
            draw_order = [0, 1] if alternative_route else [0]
            for idx in draw_order:
                if idx == 0:
                    draw_route, travel_time = fastest_route
                    color = colors[1]  # blue
                    weight = weights[1]
                else:
                    draw_route, travel_time = alternative_route
                    color = colors[0]  # red
                    weight = weights[0]
                road_route_coords = []
                for j in range(len(draw_route) - 1):
                    node_a = draw_route[j]
                    node_b = draw_route[j+1]
                    node_a_data = self.df[self.df['site_id'].astype(str) == str(node_a)]
                    node_b_data = self.df[self.df['site_id'].astype(str) == str(node_b)]
                    if len(node_a_data) == 0 or len(node_b_data) == 0:
                        continue
                    coords_a = (node_a_data.iloc[0]['NB_LATITUDE'], node_a_data.iloc[0]['NB_LONGITUDE'])
                    coords_b = (node_b_data.iloc[0]['NB_LATITUDE'], node_b_data.iloc[0]['NB_LONGITUDE'])
                    segment = self.get_road_route(coords_a, coords_b)
                    if segment:
                        if road_route_coords and road_route_coords[-1] == segment[0]:
                            road_route_coords.extend(segment[1:])
                        else:
                            road_route_coords.extend(segment)
                if road_route_coords:
                    folium.PolyLine(
                        road_route_coords,
                        color=color,
                        weight=weight,
                        opacity=1.0,
                        popup=f"Route {idx+1} - Travel Time: {travel_time:.1f} minutes"
                    ).add_to(m)
                for site_id in draw_route:
                    site_data = self.df[self.df['site_id'].astype(str) == str(site_id)]
                    if len(site_data) == 0:
                        continue
                    coords = (site_data.iloc[0]['NB_LATITUDE'], site_data.iloc[0]['NB_LONGITUDE'])
                    area = site_data.iloc[0]['Location']
                    folium.CircleMarker(
                        location=coords,
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.8,
                        tooltip=f"{site_id}: {area}"
                    ).add_to(m)
            folium.Marker(
                origin_coords,
                popup=f"Origin: {origin_id}",
                tooltip=f"Origin: {origin_id}",
                icon=folium.Icon(color='green', icon='home')
            ).add_to(m)
            folium.Marker(
                dest_coords,
                popup=f"Destination: {dest_id}",
                tooltip=f"Destination: {dest_id}",
                icon=folium.Icon(color='red', icon='flag')
            ).add_to(m)
            all_site_ids = set(self.df['site_id'].astype(str).unique())
            route_site_ids = set()
            for route, _ in [fastest_route, alternative_route] if alternative_route else [fastest_route]:
                route_site_ids.update(route)
            origin_site_id = str(origin_id)
            dest_site_id = str(dest_id)
            for site_id in all_site_ids:
                if (site_id in route_site_ids) or (site_id == origin_site_id) or (site_id == dest_site_id):
                    continue
                site_data = self.df[self.df['site_id'].astype(str) == str(site_id)]
                if len(site_data) == 0:
                    continue
                coords = (site_data.iloc[0]['NB_LATITUDE'], site_data.iloc[0]['NB_LONGITUDE'])
                area = site_data.iloc[0]['Location']
                folium.CircleMarker(
                    location=coords,
                    radius=4,
                    color='#888',
                    fill=True,
                    fill_color='#888',
                    fill_opacity=0.7,
                    tooltip=f"{site_id}: {area}"
                ).add_to(m)
            data = io.BytesIO()
            m.save(data, close_file=False)
            self.map_view.setHtml(data.getvalue().decode())
            progress.setLabelText("Route calculation complete!")
            progress.setValue(100)
            QApplication.processEvents()
            progress.close()
            summary = "<b>Route Summary (K-shortest, blue on top):</b><br><br>"
            if alternative_route:
                summary += f"<b>Fastest Route 🔵</b><br> (Travel Time: {fastest_route[1]:.1f} minutes):<br>"
                summary += " → ".join(fastest_route[0]) + "<br><br>"
                summary += f"<b>Alternative Route 🔴</b><br> (Travel Time: {alternative_route[1]:.1f} minutes):<br>"
                summary += " → ".join(alternative_route[0]) + "<br><br>"
            else:
                summary += f"<b>Fastest Route 🔵</b><br> (Travel Time: {fastest_route[1]:.1f} minutes):<br>"
                summary += " → ".join(fastest_route[0]) + "<br><br>"
            self.route_summary.setText(summary)
        except Exception as e:
            error_msg = f"Error finding routes: {str(e)}"
            print(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
            self.route_summary.setText("")
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
            graph[str(origin)].append((str(dest), weight))
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