import os

# Import model training functions
from lstm_model import train_lstm_model
from train_models import train_gru_model
from cnn_model import train_cnn_model

# Path to normalized data for a sample site (adjust if needed)
input_path = "data/scats_0970_normalized.csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

window_size = 12

print("\n--- Testing LSTM Model ---")
lstm_output = os.path.join(output_dir, "y_pred_lstm.npy")
y_test, y_pred, lstm_rmse = train_lstm_model(input_path=input_path, output_path=lstm_output, window_size=window_size, epochs=5)
print(f"LSTM Test RMSE: {lstm_rmse:.4f}")

print("\n--- Testing GRU Model ---")
# train_gru_model expects site_id and always loads scats_0970_normalized.csv
train_gru_model(site_id="0970", epochs_override=5)  # RMSE will be printed inside the function

print("\n--- Testing CNN Model ---")
cnn_output = os.path.join(output_dir, "y_pred_cnn.npy")
y_test, y_pred, cnn_rmse = train_cnn_model(input_path=input_path, output_path=cnn_output, window_size=window_size)
print(f"CNN Test RMSE: {cnn_rmse:.4f}") 