#105106819 Suman Sutparai
# LSTM model for TBRGS
import tensorflow as tf
from base_model import BaseModel
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

class CancelTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, should_stop_func):
        super().__init__()
        self.should_stop_func = should_stop_func
    def on_epoch_end(self, epoch, logs=None):
        if self.should_stop_func():
            self.model.stop_training = True

class LSTMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.history = None
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=(input_shape[0], 1)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model

    def train(self, X_train, y_train, X_val, y_val, progress_callback=None, cancel_callback=None, epochs=10):
        if self.model is None:
            self.build_model(X_train.shape[1:])
        callbacks = []
        if progress_callback:
            callbacks.append(progress_callback)
        if cancel_callback:
            callbacks.append(cancel_callback)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks.append(early_stopping)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.config.get('batch_size', 32),
            verbose=0,
            callbacks=callbacks
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            
        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been built.")
            
        metrics = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'loss': metrics[0],
            'mae': metrics[1]
        }

    def plot_predictions(self, X_test, y_test, title="LSTM Traffic Flow Prediction"):
        """Plot model predictions against actual values and save to file.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            title (str): Plot title.
            
        Returns:
            float: RMSE value
        """
        if self.model is None:
            raise ValueError("Model has not been built.")
            
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot actual and predicted values
        plt.plot(y_test, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red')
        
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Traffic Volume')
        plt.title(title)
        plt.legend()
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join("plots", f"{title.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Calculate and print RMSE
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Plot saved to: {plot_path}")
        
        return rmse

    def plot_training_history(self):
        """Plot training history (loss and metrics) and save to file.
        """
        if self.history is None:
            raise ValueError("Model has not been trained.")
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot metrics
        if 'mae' in self.history.history:
            ax2.plot(self.history.history['mae'], label='Training MAE')
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join("plots", "lstm_training_history.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Training history plot saved to: {plot_path}")

    def predict_sequence(self, X, steps):
        """Predict a sequence of future values.
        
        Args:
            X (np.ndarray): Input sequence.
            steps (int): Number of steps to predict.
            
        Returns:
            np.ndarray: Predicted sequence.
        """
        predictions = []
        current_input = X.copy()
        
        for _ in range(steps):
            # Make prediction
            pred = self.predict(current_input)
            predictions.append(pred[0, 0])
            
            # Update input sequence
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1] = pred[0, 0]
            
        return np.array(predictions)

def train_lstm_model(input_path, output_path, window_size=12, epochs=50, cancel_func=None):
    """Train an LSTM model on the given data.
    
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the predictions.
        window_size (int): Size of the input window.
        epochs (int): Number of training epochs.
        
    Returns:
        tuple: (y_test, y_pred, rmse)
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    
    # Load and preprocess data
    df = pd.read_csv(input_path)
    data = df['normalized_volume'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_normalized) - window_size):
        X.append(data_normalized[i:(i + window_size)])
        y.append(data_normalized[i + window_size])
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    config = {
        'batch_size': 32,
        'learning_rate': 0.001
    }
    model = LSTMModel(config)
    model.build_model(input_shape=(window_size, 1))
    cancel_callback = CancelTrainingCallback(cancel_func) if cancel_func else None
    model.train(X_train, y_train, X_test, y_test, epochs=epochs, cancel_callback=cancel_callback)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Save the trained model
    model.model.save(output_path)
    
    # Plot results
    # model.plot_predictions(X_test, y_test)
    
    return y_test, y_pred, rmse 