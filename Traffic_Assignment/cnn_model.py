import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

class CNNModel:
    def __init__(self, window_size=12):
        self.window_size = window_size
        self.model = None

    def prepare_data(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        X = np.array(X).reshape((-1, self.window_size, 1))
        y = np.array(y)
        return X, y

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        self.build_model(input_shape=(X_train.shape[1], 1))
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

    def predict(self, X):
        return self.model.predict(X)


def train_cnn_model(input_path="data/scats_0970_normalized.csv", output_path="output/y_pred_cnn.npy", window_size=12, epochs=20):
    df = pd.read_csv(input_path)
    # Use 'normalized_volume' if available, else 'Volume_norm', else 'Volume'
    if 'normalized_volume' in df.columns:
        data = df['normalized_volume'].values
    elif 'Volume_norm' in df.columns:
        data = df['Volume_norm'].values
    else:
        data = df['Volume'].values
    X, y = CNNModel(window_size).prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = CNNModel(window_size)
    model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=32)
    y_pred = model.predict(X_test)
    if output_path is not None:
        np.save(output_path, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"CNN Test RMSE: {rmse:.4f}")

    # Plot with hour labels on x-axis
    time_indices = df.index[-len(y_test):]
    if 'Date' in df.columns and 'Time' in df.columns:
        # If time columns exist, use them
        x_labels = [f"{df.loc[idx, 'Time']}" for idx in time_indices]
    else:
        # Otherwise, create hour:minute labels
        x_labels = []
        start_hour = 9  # Example: start at 9:00
        for i in range(len(y_test)):
            hour = start_hour + (i // 2)
            minute = '00' if i % 2 == 0 else '30'
            x_labels.append(f"{hour}:{minute}")
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Number of Cars')
    plt.title('CNN Prediction vs Actual')
    plt.xticks(ticks=np.arange(0, len(x_labels), max(1, len(x_labels)//10)), labels=[x_labels[i] for i in np.arange(0, len(x_labels), max(1, len(x_labels)//10))], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save the trained model
    site_id = input_path.split('_')[1].split('.')[0]  # Extract site_id from filename
    os.makedirs("models", exist_ok=True)
    model.model.save(f"models/cnn_site_{site_id}.h5")

    return y_test, y_pred, rmse

# Example usage:
# y_test, y_pred, rmse = train_cnn_model() 