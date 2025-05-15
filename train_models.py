#105106819 Suman Sutparai
"""
Script to train ML models for TBRGS.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processor import DataProcessor
from ml.lstm_model import LSTMModel
from ml.gru_model import GRUModel
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

def train_models():
    """Train all ML models."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting model training")
    
    # Load configuration
    config = ConfigLoader()
    ml_config = config.get('ml')
    
    # Initialize data processor
    processor = DataProcessor(ml_config)
    
    # Load data
    logger.info("Loading data")
    traffic_data, scats_data, coordinates = processor.load_data()
    
    # Add time features
    traffic_data = processor.add_time_features(traffic_data)
    
    # Get unique site IDs
    site_ids = traffic_data['site_id'].unique()
    
    # Train models for each site
    for site_id in site_ids:
        logger.info(f"Training models for site {site_id}")
        
        # Prepare sequences
        X, y = processor.prepare_sequences(traffic_data, site_id)
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(
            X, y,
            train_ratio=ml_config.get('train_test_split'),
            val_ratio=ml_config.get('validation_split')
        )
        
        # Train LSTM model
        logger.info("Training LSTM model")
        lstm_model = LSTMModel(ml_config)
        lstm_model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate LSTM model
        lstm_metrics = lstm_model.evaluate(X_test, y_test)
        logger.info(f"LSTM model metrics: {lstm_metrics}")
        
        # Save LSTM model
        lstm_model.save(f"models/lstm_site_{site_id}.h5")
        
        # Train GRU model
        logger.info("Training GRU model")
        gru_model = GRUModel(ml_config)
        gru_model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate GRU model
        gru_metrics = gru_model.evaluate(X_test, y_test)
        logger.info(f"GRU model metrics: {gru_metrics}")
        
        # Save GRU model
        gru_model.save(f"models/gru_site_{site_id}.h5")
        
    logger.info("Model training completed")

if __name__ == "__main__":
    train_models() 