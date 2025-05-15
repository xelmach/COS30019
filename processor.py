"""
Data processor for TBRGS.
"""

#105106819 Suman Sutparai
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self, config):
        """Initialize the data processor.
        
        Args:
            config (dict): Configuration settings.
        """
        self.config = config
        self.scaler = MinMaxScaler()
        self.sequence_length = config.get('sequence_length', 24)
        self.prediction_horizon = config.get('prediction_horizon', 4)
        
    def load_data(self):
        """Load and preprocess the traffic data.
        
        Returns:
            tuple: (traffic_data, scats_data, coordinates)
        """
        traffic_data_path = self.config.get('traffic_data_path', 'data/traffic_data.csv')
        scats_data_path = self.config.get('scats_data_path', 'data/scats_data.csv')
        coordinates_path = self.config.get('coordinates_path', 'data/coordinates.csv')
        traffic_data = pd.read_csv(traffic_data_path)
        scats_data = pd.read_csv(scats_data_path)
        coordinates = pd.read_csv(coordinates_path)
        return traffic_data, scats_data, coordinates
        
    def prepare_sequences(self, data, site_id):
        """Prepare sequences for model training.
        
        Args:
            data (pd.DataFrame): Traffic data.
            site_id (int): SCATS site ID.
            
        Returns:
            tuple: (X, y) training sequences and targets.
        """
        site_data = data[data['site_id'] == site_id].sort_values('timestamp')
        values = site_data['traffic_volume'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        X, y = [], []
        for i in range(len(scaled_values) - self.sequence_length):
            X.append(scaled_values[i:i+self.sequence_length])
            y.append(scaled_values[i+self.sequence_length])
        X = np.array(X)
        y = np.array(y)
        return X, y
        
    def inverse_transform(self, scaled):
        """Inverse transform scaled data.
        
        Args:
            scaled (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Original scale data.
        """
        return self.scaler.inverse_transform(scaled)
        
    def add_time_features(self, data):
        """Add time-based features to the data.
        
        Args:
            data (pd.DataFrame): Traffic data.
            
        Returns:
            pd.DataFrame: Data with additional time features.
        """
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['dayofweek'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        return data
        
    def split_data(self, X, y, train_ratio=0.8, val_ratio=0.1):
        """Split data into training, validation, and test sets.
        
        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Targets.
            train_ratio (float): Training set ratio.
            val_ratio (float): Validation set ratio.
            
        Returns:
            tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test) 