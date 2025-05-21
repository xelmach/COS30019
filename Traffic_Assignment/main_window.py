#105106819 Suman Sutparai
# Main window for TBRGS
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QTabWidget, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem, QSplitter, QProgressDialog, QApplication, QProgressBar, QFrame, QTextEdit, QSizePolicy
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
        
        # Get the screen size and set window to cover entire screen
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        # Main layout with more padding
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.layout.setSpacing(18)
        
        # App title at the top
        self.title_label = QLabel("Traffic-based Route Guidance System (TBRGS)")
        self.title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #888; margin-bottom: 12px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabWidget::pane { border-radius: 12px; padding: 8px; }")
        self.layout.addWidget(self.tab_widget)
        
        # Create all tabs first
        print("Creating tabs...")
        self.tab_widget.addTab(self._create_route_tab(), "Route Planning")
        self.tab_widget.addTab(self._create_prediction_tab(), "Traffic Prediction")
        self.tab_widget.addTab(self._create_settings_tab(), "Settings")
        
        # Initialize route finder
        print("Initializing route finder...")
        self.route_finder = RouteFinder(self.config)
        if edges and coordinates:
            print("\nInitializing route finder with graph data...")
            print(f"Number of edges: {len(edges)}")
            print(f"Number of coordinates: {len(coordinates)}")
            self.route_finder.build_graph(edges, coordinates)
        
        # Initialize site dropdowns
        print("Initializing site dropdowns...")
        self._initialize_sites()
        
        # Initialize progress signal
        self.progress = TrainingProgress()
        
        # Set dark theme as default
        if hasattr(self, 'theme_combo'):
            self.theme_combo.setCurrentText("Dark")
            self.apply_theme("Dark")
        
        # Show the window after everything is initialized
        print("Showing main window...")
        self.show()
        
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
                
            # Set styles
            self.origin_combo.setStyleSheet("QComboBox { background: #232629; color: #f0f0f0; border-radius: 8px; border: 1.5px solid #232629; padding: 6px 12px; font-size: 16px; } QComboBox::drop-down { border: none; background: transparent; } QComboBox QAbstractItemView { background: #232629; color: #f0f0f0; border-radius: 8px; }")
            self.dest_combo.setStyleSheet("QComboBox { background: #232629; color: #f0f0f0; border-radius: 8px; border: 1.5px solid #232629; padding: 6px 12px; font-size: 16px; } QComboBox::drop-down { border: none; background: transparent; } QComboBox QAbstractItemView { background: #232629; color: #f0f0f0; border-radius: 8px; }")
            
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

        # Left side: Controls panel (transparent)
        controls_panel = QFrame()
        controls_panel.setStyleSheet("QFrame { background: transparent; }")
        controls_panel.setFixedWidth(380)
        controls_panel_layout = QVBoxLayout(controls_panel)
        controls_panel_layout.setContentsMargins(24, 4, 24, 24)  # Add space between border and card (left, top, right, bottom)
        controls_panel_layout.setSpacing(0)
        controls_panel_layout.addStretch(1)

        # Card-like, dark, rounded controls area for prediction tab
        controls_card = QFrame()
        controls_card.setStyleSheet("QFrame { background: #2d3238; border-radius: 18px; padding: 32px 32px 24px 32px; }")
        controls_card_layout = QVBoxLayout(controls_card)
        controls_card_layout.setSpacing(18)
        controls_card_layout.setContentsMargins(0, 0, 0, 0)

        # Site selection (card style)
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
        controls_card_layout.addWidget(site_card)

        # Prediction horizon (card style)
        horizon_card = QFrame()
        horizon_card.setStyleSheet("QFrame { background: #2d3238; border-radius: 16px; padding: 12px 12px 24px 12px; }")
        horizon_layout = QVBoxLayout(horizon_card)
        horizon_layout.setContentsMargins(0, 0, 0, 0)
        horizon_layout.setSpacing(8)
        horizon_label = QLabel("Prediction Horizon (hours):")
        horizon_label.setStyleSheet("font-weight: 600; font-size: 16px; color: #f0f0f0; margin: 0 0 8px 0; padding-top: 1px; padding-bottom: 1px;")
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 24)
        self.horizon_spin.setValue(1)
        self.horizon_spin.setMinimumHeight(36)
        self.horizon_spin.setStyleSheet(
            "QSpinBox { background: #232629; color: #f0f0f0; border-radius: 8px; border: 1.5px solid #888; padding: 6px 12px; font-size: 16px; }"
        )
        horizon_layout.addWidget(horizon_label)
        horizon_layout.addWidget(self.horizon_spin)
        controls_card_layout.addWidget(horizon_card)

        # Predict button (compact, full width)
        predict_btn = QPushButton("Predict Traffic")
        predict_btn.setMinimumHeight(54)
        predict_btn.setMinimumWidth(0)
        predict_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        predict_btn.clicked.connect(self._predict_traffic)
        controls_card_layout.addWidget(predict_btn)
        controls_card_layout.addStretch(1)

        # Add the card to the panel, centered vertically
        controls_panel_layout.addWidget(controls_card, alignment=Qt.AlignTop)
        controls_panel_layout.addStretch(2)

        # Right side: Graph and Summary
        right_panel = QFrame()
        right_panel.setStyleSheet("QFrame { background: transparent; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(32)
        right_layout.setContentsMargins(0, 32, 32, 32)

        # Graph card
        graph_card = QFrame()
        graph_card.setStyleSheet(
            "QFrame { background: #232629; border-radius: 16px; border: 1.5px solid #888; padding: 0; margin-bottom: 24px; }"
        )
        graph_layout = QVBoxLayout(graph_card)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)
        self.prediction_label = QLabel()
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setMinimumHeight(400)
        self.prediction_label.setStyleSheet(
            "QLabel { "
            "background: #232629; "
            "color: #b0b0b0; "
            "border: none; "
            "border-radius: 14px; "
            "font-size: 18px; "
            "padding: 24px 32px 32px 32px; "
            "}"
        )
        self.prediction_label.setText('<div style="color:#888;font-size:18px;line-height:1.6;">\n<span style="font-size:48px;">📈</span><br>Click "Predict Traffic" to generate a prediction.</div>')
        graph_layout.addWidget(self.prediction_label)
        right_layout.addWidget(graph_card)

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

        # Progress bar for training (styled to blend in)
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setMinimum(0)
        self.training_progress_bar.setMaximum(100)
        self.training_progress_bar.setValue(0)
        self.training_progress_bar.setTextVisible(True)
        self.training_progress_bar.setVisible(False)
        self.training_progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #3daee9; border-radius: 8px; text-align: center; height: 16px; font-size: 14px; margin: 24px 48px 24px 48px; background: #232629; color: #f0f0f0; }"
            "QProgressBar::chunk { background-color: #3daee9; border-radius: 8px; }"
        )
        right_layout.addWidget(self.training_progress_bar)

        # Add panels to main layout
        layout.addWidget(controls_panel)
        layout.addWidget(right_panel)
        layout.setStretch(0, 3)  # controls_panel: 30%
        layout.setStretch(1, 7)  # right_panel: 70%

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
                self.route_summary.setText("")
                return
                
            if len(dest_data) == 0:
                error_msg = f"No data found for destination site {dest_id}"
                print(error_msg)
                print(f"Available site IDs: {sorted(self.df['site_id'].unique())}")
                QMessageBox.warning(self, "Error", error_msg)
                self.route_summary.setText("")
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
                self.route_summary.setText("")
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
                popup=f"Location: {origin_data.iloc[0]['Location']}",
                icon=folium.Icon(color='gray')
            ).add_to(m)
            
            folium.Marker(
                dest_coords,
                tooltip=f"Destination: {dest_id}",
                popup=f"Location: {dest_data.iloc[0]['Location']}",
                icon=folium.Icon(color='gray')
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
            
            # Show route summary in the QTextEdit
            summary = "<b>Route Summary:</b><br><br>"
            for i, (route, cost) in enumerate(routes):
                summary += f"<b>Route {i+1}:</b><br>"
                summary += f"Path: {' → '.join(route)}<br>"
                summary += f"Travel Time: <b>{cost:.2f} minutes</b><br><br>"
            self.route_summary.setHtml(summary)
            
        except Exception as e:
            error_msg = f"Error finding routes: {str(e)}"
            print(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
            self.route_summary.setText("")
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
            epochs = 50  # or get from config

            # Clear the prediction graph area and show training message
            self.prediction_label.setText(f"<b>Training model...</b><br>Epoch 0/{epochs}")
            self.prediction_label.setStyleSheet(
                "QLabel { "
                "background: #f5f6fa; "
                "color: #232629; "
                "border: 2px dashed #b0b0b0; "
                "border-radius: 12px; "
                "font-size: 16px; "
                "padding: 16px; "
                "}" 
            )
            self.training_progress_bar.setVisible(True)
            self.training_progress_bar.setValue(0)

            def update_progress(current, total):
                self.prediction_label.setText(f"<b>Training model...</b><br>Epoch {current}/{total}")
                percent = int((current / total) * 100)
                self.training_progress_bar.setValue(percent)
                QApplication.processEvents()

            progress = TrainingProgress()
            progress.epoch_signal.connect(update_progress)

            def on_training_done():
                self.training_progress_bar.setVisible(False)
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
            
            # Get data for the selected site - only get the most recent data
            site_data = self.df[self.df['site_id'].astype(str) == site_id].tail(96).copy()
            
            print(f"\nDebug - Prediction for site {site_id}:")
            print(f"Site data rows found: {len(site_data)}")
            
            if len(site_data) == 0:
                QMessageBox.warning(self, "Error", f"No data found for site {site_id}")
                return
            
            # Get the time columns (15-minute intervals)
            time_columns = [col for col in site_data.columns if (':' in str(col)) or (str(col).startswith('V') and str(col)[1:].isdigit() and len(str(col)) == 3)]
            
            if not time_columns:
                QMessageBox.warning(self, "Error", "No time-based columns found in the data")
                return
            
            # Convert time columns to numeric values and normalize - do it all at once
            data = site_data[time_columns].values.flatten()
            data = pd.to_numeric(data, errors='coerce')
            
            # Remove any NaN values
            data = data[~np.isnan(data)]
            
            if len(data) < 96:
                QMessageBox.warning(self, "Error", "Not enough data points for prediction")
                return
            
            # Normalize the data
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data.reshape(-1, 1))
            
            # Prepare for multi-step prediction
            horizon_steps = self.horizon_spin.value() * 4  # 4 steps per hour (15-min intervals)
            predictions = []
            current_input = normalized_data[-96:].reshape(1, 96, 1)
            
            # Load and use the GRU model
            model_path = f"models/gru_site_{site_id}.h5"
            if not os.path.exists(model_path):
                QMessageBox.warning(
                    self,
                    "Model Not Found",
                    f"Prediction model for site {site_id} not found. Please train the model first."
                )
                return
            
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Recursively predict for the horizon
            for _ in range(horizon_steps):
                pred = model.predict(current_input, verbose=0)
                predictions.append(pred[0, 0])
                # Update input: remove first, append new prediction
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = pred[0, 0]
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)
            
            # Plot the prediction
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(predictions)), predictions, 'b-', linewidth=2, label='Predicted Traffic')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Traffic Prediction for Site {site_id}', fontsize=14, pad=15)
            plt.xlabel('Time Steps (15-minute intervals)', fontsize=12)
            plt.ylabel('Traffic Volume', fontsize=12)
            plt.legend(fontsize=10)
            plt.tight_layout()
            temp_file = 'temp_prediction.png'
            plt.savefig(temp_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Display the plot in the QLabel with proper scaling
            pixmap = QPixmap(temp_file)
            scaled_pixmap = pixmap.scaled(self.prediction_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.prediction_label.setPixmap(scaled_pixmap)
            
            # Show prediction summary
            avg_prediction = np.mean(predictions)
            max_prediction = np.max(predictions)
            min_prediction = np.min(predictions)
            summary_html = (
                f"<b>Traffic prediction for site {site_id}:</b><br><br>"
                f"<b>Average predicted volume:</b> {avg_prediction:.0f}<br>"
                f"<b>Maximum predicted volume:</b> {max_prediction:.0f}<br>"
                f"<b>Minimum predicted volume:</b> {min_prediction:.0f}"
            )
            self.prediction_summary.setHtml(summary_html)
            
        except Exception as e:
            self.logger.error(f"Error predicting traffic: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error predicting traffic: {str(e)}") 