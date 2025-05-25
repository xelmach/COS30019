import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load the raw data from the 'Data' sheet
print("Loading raw data...")
df = pd.read_excel("Scats Data October 2006.xls", sheet_name="Data", header=1)

# Get all unique site IDs
site_ids = df['SCATS Number'].unique()
print(f"Found {len(site_ids)} unique SCATS sites.")

for site_id in site_ids:
    print(f"Processing site {site_id}...")
    site_data = df[df['SCATS Number'] == site_id].copy()
    if site_data.empty:
        print(f"No data found for site {site_id}.")
        continue
    time_series_cols = [col for col in site_data.columns if col.startswith('V')]
    time_series_cols.sort()
    time_series_data = site_data[time_series_cols].values.flatten()
    time_series_data = time_series_data[~np.isnan(time_series_data)]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(time_series_data.reshape(-1, 1))
    normalized_df = pd.DataFrame(normalized_data, columns=['normalized_volume'])
    output_path = f"data/scats_{site_id}_normalized.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    normalized_df.to_csv(output_path, index=False)
    print(f"Saved normalized data to {output_path}.")

print("All sites processed.")

def normalize_scats_file(input_path, output_path):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import os

    # Load the raw data for the site
    df = pd.read_csv(input_path)
    if 'Volume' in df.columns:
        data = df['Volume'].values
    else:
        # Try to find the correct column if named differently
        data = df.iloc[:, 1].values  # fallback: use the second column

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    normalized_df = pd.DataFrame(normalized_data, columns=['normalized_volume'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    normalized_df.to_csv(output_path, index=False)

def export_site_csvs():
    import pandas as pd
    import numpy as np
    import os
    df = pd.read_excel("Scats Data October 2006.xls", sheet_name="Data", header=1)
    site_ids = df['SCATS Number'].unique()
    for site_id in site_ids:
        site_data = df[df['SCATS Number'] == site_id].copy()
        if site_data.empty:
            continue
        # Flatten all V columns into a single time series
        time_series_cols = [col for col in site_data.columns if col.startswith('V')]
        time_series_cols.sort()
        time_series_data = site_data[time_series_cols].values.flatten()
        # Remove NaNs
        time_series_data = time_series_data[~np.isnan(time_series_data)]
        # Save as CSV
        out_df = pd.DataFrame({'Volume': time_series_data})
        out_path = f"data/scats_{site_id}.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False) 