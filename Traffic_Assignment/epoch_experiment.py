import time
import os
import numpy as np
import pandas as pd
from lstm_model import train_lstm_model
from train_models import train_gru_model
from cnn_model import train_cnn_model

input_path = "data/scats_0970_normalized.csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
window_size = 12
site_id = "0970"
epoch_values = [5, 10, 20, 50, 100]

results = []

for epochs in epoch_values:
    print(f"\n=== EPOCHS: {epochs} ===")
    # LSTM
    print("\n--- LSTM ---")
    lstm_output = os.path.join(output_dir, f"y_pred_lstm_{epochs}.npy")
    start = time.time()
    y_test, y_pred, lstm_rmse = train_lstm_model(input_path=input_path, output_path=lstm_output, window_size=window_size, epochs=epochs)
    lstm_time = time.time() - start
    print(f"LSTM Test RMSE: {lstm_rmse:.4f} | Time: {lstm_time:.2f}s")
    results.append({"model": "LSTM", "epochs": epochs, "rmse": float(lstm_rmse), "time": lstm_time})

    # GRU
    print("\n--- GRU ---")
    start = time.time()
    # train_gru_model returns True/False, prints RMSE inside
    train_gru_model(site_id=site_id, epochs_override=epochs)
    gru_time = time.time() - start
    # Try to load last RMSE from output (if available)
    # You may want to modify train_gru_model to return RMSE for more accuracy
    results.append({"model": "GRU", "epochs": epochs, "rmse": None, "time": gru_time})

    # CNN
    print("\n--- CNN ---")
    cnn_output = os.path.join(output_dir, f"y_pred_cnn_{epochs}.npy")
    start = time.time()
    y_test, y_pred, cnn_rmse = train_cnn_model(input_path=input_path, output_path=cnn_output, window_size=window_size)
    cnn_time = time.time() - start
    print(f"CNN Test RMSE: {cnn_rmse:.4f} | Time: {cnn_time:.2f}s")
    results.append({"model": "CNN", "epochs": epochs, "rmse": float(cnn_rmse), "time": cnn_time})

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "epoch_experiment_results.csv"), index=False)
print("\nExperiment complete. Results saved to output/epoch_experiment_results.csv") 