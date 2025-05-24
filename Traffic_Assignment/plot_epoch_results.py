import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('output/epoch_experiment_results.csv')

# Plot Epochs vs. RMSE
plt.figure(figsize=(8, 5))
for model in results['model'].unique():
    model_data = results[results['model'] == model]
    if model_data['rmse'].notnull().any():
        plt.plot(model_data['epochs'], model_data['rmse'], marker='o', label=model)
plt.xlabel('Epochs')
plt.ylabel('Test RMSE')
plt.title('Epochs vs. Test RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/epochs_vs_rmse.png')
plt.show()

# Plot Epochs vs. Training Time
plt.figure(figsize=(8, 5))
for model in results['model'].unique():
    model_data = results[results['model'] == model]
    plt.plot(model_data['epochs'], model_data['time'], marker='o', label=model)
plt.xlabel('Epochs')
plt.ylabel('Training Time (s)')
plt.title('Epochs vs. Training Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/epochs_vs_time.png')
plt.show()

print('Plots saved as output/epochs_vs_rmse.png and output/epochs_vs_time.png') 