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
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor import DataProcessor
from lstm_model import LSTMModel
from gru_model import GRUModel
from config_loader import ConfigLoader
from logger import setup_logger

def train_models(site_id=None, progress_callback=None, epochs_override=None):
    """Train ML models. If site_id is provided, only train for that site."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting model training")
    
    # Load configuration
    config = ConfigLoader()
    ml_config = config.get('ml', {})
    
    # Set default values for configuration
    ml_config.setdefault('train_test_split', 0.8)
    ml_config.setdefault('validation_split', 0.1)
    ml_config.setdefault('epochs', 50)
    
    # Initialize data processor
    processor = DataProcessor(ml_config)
    
    try:
        # Load data
        logger.info("Loading data")
        traffic_data, scats_data, coordinates = processor.load_data()
        
        if traffic_data is None or len(traffic_data) == 0:
            logger.error("No traffic data loaded")
            return
            
        # Add time features
        traffic_data = processor.add_time_features(traffic_data)
        
        # Get unique site IDs
        site_ids = traffic_data['site_id'].unique()
        logger.info(f"Found {len(site_ids)} unique sites")
        
        # If a specific site_id is provided, only train for that site
        if site_id is not None:
            site_id = str(site_id)  # Convert to string for consistency
            if site_id not in site_ids:
                logger.error(f"Site ID {site_id} not found in data")
                return
            site_ids = [site_id]
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train models for each site
        for sid in site_ids:
            logger.info(f"Training models for site {sid}")
            print(f"Training models for site {sid}")
            
            # Prepare sequences
            X, y = processor.prepare_sequences(traffic_data, sid)
            if X.size == 0 or y.size == 0:
                logger.warning(f"No data for site {sid}, skipping.")
                print(f"No data for site {sid}, skipping.")
                continue
            
            # Split data
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(
                X, y,
                train_ratio=ml_config['train_test_split'],
                val_ratio=ml_config['validation_split']
            )
            
            if len(X_train) == 0 or len(y_train) == 0:
                logger.warning(f"Not enough data for training site {sid}, skipping.")
                continue
            
            # Train LSTM model
            logger.info("Training LSTM model")
            epochs = epochs_override if epochs_override is not None else ml_config['epochs']
            lstm_model = LSTMModel(ml_config)
            lstm_model.train(X_train, y_train, X_val, y_val, progress_callback=progress_callback, epochs=epochs)
            
            # Evaluate LSTM model
            lstm_metrics = lstm_model.evaluate(X_test, y_test)
            logger.info(f"LSTM model metrics: {lstm_metrics}")
            
            # Plot LSTM predictions
            print("\nPlotting LSTM predictions...")
            lstm_rmse = lstm_model.plot_predictions(X_test, y_test, title=f"LSTM Traffic Flow Prediction - Site {sid}")
            lstm_model.plot_training_history()
            
            # Save LSTM model
            model_path = f"models/lstm_site_{sid}.h5"
            lstm_model.save(model_path)
            logger.info(f"Saved LSTM model to {model_path}")
            print(f"Saved model: {model_path}")
            
            # Train GRU model
            logger.info("Training GRU model")
            gru_model = GRUModel(ml_config)
            gru_model.train(X_train, y_train, X_val, y_val, progress_callback=progress_callback, epochs=epochs)
            
            # Evaluate GRU model
            gru_metrics = gru_model.evaluate(X_test, y_test)
            logger.info(f"GRU model metrics: {gru_metrics}")
            
            # Plot GRU predictions
            print("\nPlotting GRU predictions...")
            gru_rmse = gru_model.plot_predictions(X_test, y_test, title=f"GRU Traffic Flow Prediction - Site {sid}")
            gru_model.plot_training_history()
            
            # Save GRU model
            model_path = f"models/gru_site_{sid}.h5"
            gru_model.save(model_path)
            logger.info(f"Saved GRU model to {model_path}")
            print(f"Saved model: {model_path}")
            
            # Print comparison of models
            print(f"\nModel Comparison for Site {sid}:")
            print(f"LSTM RMSE: {lstm_rmse:.4f}")
            print(f"GRU RMSE: {gru_rmse:.4f}")
            
        logger.info("Model training completed")
        print("Models directory contents:", os.listdir("models"))
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def train_gru_model(site_id, progress_callback=None, epochs_override=None):
    """Train GRU model for traffic prediction.
    
    Args:
        site_id: Site ID to train model for
        progress_callback: Callback for training progress
        epochs_override: Override number of epochs
    """
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Load and prepare data
        df = pd.read_csv("scats_0970_normalized.csv", index_col=0)
        data = df["Volume_norm"].values
        
        # Construct time window data
        window_size = 12
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build GRU model
        model = Sequential([
            GRU(64, input_shape=(window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        epochs = epochs_override if epochs_override else 20
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            callbacks=[progress_callback] if progress_callback else None
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Test RMSE: {rmse:.4f}")
        
        # Save model
        model_path = f"models/gru_model_scats{site_id}.h5"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save predictions
        np.save(f"models/y_pred_{site_id}.npy", y_pred)
        
        # Plot results
        plt.figure(figsize=(10, 4))
        plt.plot(y_test, label="True")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        plt.title(f"GRU Traffic Flow Prediction - Site {site_id}")
        plt.tight_layout()
        plt.savefig(f"models/prediction_plot_{site_id}.png")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_models() 