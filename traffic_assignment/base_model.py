"""
Base class for ML models in TBRGS.
"""

#105106819 Suman Sutparai
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

class BaseModel(ABC):
    """Base class for all ML models in TBRGS."""
    
    def __init__(self, config):
        """Initialize the base model.
        
        Args:
            config (dict): Model configuration.
        """
        self.config = config
        self.model = None
        self.history = None
        
    @abstractmethod
    def build_model(self, input_shape):
        """Build the model architecture.
        
        Args:
            input_shape (tuple): Shape of input data.
        """
        pass
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation labels.
        """
        if self.model is None:
            self.build_model(X_train.shape[1:])
            
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join('models', f'{self.__class__.__name__}.h5'),
                save_best_only=True
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
    def predict(self, X):
        """Make predictions.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Model predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")
        return self.model.predict(X)
        
    def save(self, filepath):
        """Save the model.
        
        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        
    def load(self, filepath):
        """Load the model.
        
        Args:
            filepath (str): Path to load the model from.
        """
        self.model = tf.keras.models.load_model(filepath)
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
            
        Returns:
            dict: Evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")
        return self.model.evaluate(X_test, y_test, verbose=0) 