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
        self.available_sites = None  # Store available sites
        
    def load_data(self):
        """Load and preprocess the traffic data.
        
        Returns:
            tuple: (traffic_data, scats_data, coordinates)
        """
        input_file = self.config.get('input_file', 'Scats Data October 2006.xls')
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Data file not found: {input_file}")
            
        # Read only necessary columns to improve performance
        usecols = [
            'SCATS Number', 'Location', 'Date',
            'NB_LATITUDE', 'NB_LONGITUDE',  # Add latitude and longitude columns
            'CD_MELWAY'  # Add Melway reference
        ]
        # Add all V columns (V00-V94)
        usecols.extend([f'V{i:02d}' for i in range(95)])
        
        # Read the 'Data' sheet with the correct header row
        print("Loading data...")
        df = pd.read_excel(input_file, sheet_name='Data', header=1, usecols=usecols)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Print available columns for debugging
        print("Available columns:", df.columns.tolist())
        
        # Try to find the site ID column
        site_id_columns = ['site_id', 'SCATS Number', 'SCATS_SITE', 'TFM_ID']
        site_id_col = None
        for col in site_id_columns:
            if col in df.columns:
                site_id_col = col
                break
                
        if site_id_col is None:
            raise ValueError("Could not find site ID column in data")
            
        # Rename the site ID column to 'site_id'
        if site_id_col != 'site_id':
            df = df.rename(columns={site_id_col: 'site_id'})
            
        # Convert site_id to string type for consistent comparison
        df['site_id'] = df['site_id'].astype(str)
        
        # Store available sites
        self.available_sites = sorted(df['site_id'].unique())
        
        # Print unique site IDs and their counts for debugging
        site_counts = df['site_id'].value_counts()
        print("\nSite ID counts:")
        for site_id, count in site_counts.items():
            print(f"Site {site_id}: {count} rows")
        
        # For compatibility, add a 'timestamp' column if 'Date' exists
        if 'Date' in df.columns:
            df['timestamp'] = df['Date']
            
        # Create coordinates dictionary
        coordinates = {}
        for _, row in df.drop_duplicates('site_id').iterrows():
            site_id = str(row['site_id'])
            if pd.notna(row['NB_LATITUDE']) and pd.notna(row['NB_LONGITUDE']):
                coordinates[site_id] = {
                    'lat': float(row['NB_LATITUDE']),
                    'lon': float(row['NB_LONGITUDE']),
                    'name': str(row['Location']) if pd.notna(row['Location']) else site_id,
                    'melway': str(row['CD_MELWAY']) if pd.notna(row['CD_MELWAY']) else ''
                }
            
        print(f"\nLoaded coordinates for {len(coordinates)} sites")
        
        # Build graph
        edges, coordinates = self.build_graph(df, coordinates)
        
        return df, edges, coordinates
        
    def prepare_sequences(self, data, site_id):
        """Prepare sequences for model training.
        
        Args:
            data (pd.DataFrame): Traffic data.
            site_id (int or str): SCATS site ID.
            
        Returns:
            tuple: (X, y) training sequences and targets.
        """
        print(f"\nPreparing sequences for site {site_id}")
        
        # Convert site_id to string for comparison
        site_id = str(site_id)
        
        # Check if site exists
        if site_id not in self.available_sites:
            print(f"Site {site_id} not found in data")
            print("Available sites:", self.available_sites)
            return np.array([]), np.array([])
        
        # Print data info before filtering
        print(f"Total rows in data: {len(data)}")
        
        # Filter data for the specific site first to reduce memory usage
        site_data = data[data['site_id'] == site_id].copy()
        
        print(f"Rows for site {site_id}: {len(site_data)}")
        if len(site_data) == 0:
            print(f"No data found for site {site_id}")
            print("Available sites:", self.available_sites)
            return np.array([]), np.array([])
            
        # Use all columns that start with 'V' as time series
        time_columns = [col for col in site_data.columns if str(col).startswith('V') and str(col)[1:].isdigit() and len(str(col)) == 3]
        time_columns = sorted(time_columns)  # Sort columns to ensure correct order (V00, V01, ...)
        
        print(f"Time columns found: {len(time_columns)}")
        if not time_columns:
            print("No time columns found")
            return np.array([]), np.array([])
            
        # Extract only the time series data
        values = site_data[time_columns].values.flatten().reshape(-1, 1)
        
        print(f"Values shape after flattening: {values.shape}")
        if values.size == 0:
            print("No values found after flattening")
            return np.array([]), np.array([])
            
        # Scale the data
        scaled_values = self.scaler.fit_transform(values)
        
        # Prepare sequences more efficiently using numpy operations
        n_samples = len(scaled_values) - self.sequence_length
        X = np.array([scaled_values[i:i+self.sequence_length] for i in range(n_samples)])
        y = scaled_values[self.sequence_length:]
        
        print(f"Final X shape: {X.shape}, y shape: {y.shape}")
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
        # Convert timestamp only once
        timestamps = pd.to_datetime(data['timestamp'])
        data['hour'] = timestamps.dt.hour
        data['dayofweek'] = timestamps.dt.dayofweek
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
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            print("No data to split")
            return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
            
        if train_ratio is None:
            train_ratio = 0.8
        if val_ratio is None:
            val_ratio = 0.1
            
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Ensure we have at least one sample in each set
        if train_end == 0:
            train_end = 1
        if val_end <= train_end:
            val_end = train_end + 1
            
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        print(f"Split data into {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def build_graph(self, df, coordinates):
        """Build a graph from SCATS site data.
        
        Args:
            df (pd.DataFrame): Traffic data
            coordinates (dict): Dictionary of site coordinates
            
        Returns:
            tuple: (edges, coordinates) where edges is a list of (origin, dest, weight) tuples
        """
        try:
            edges = []
            speed_limit = 60  # km/h
            intersection_delay = 30  # seconds
            
            # Get unique site IDs
            sites = sorted(df['site_id'].unique())
            
            # Create edges between all pairs of sites
            for i, site1 in enumerate(sites):
                for site2 in sites[i+1:]:
                    # Get coordinates
                    if site1 not in coordinates or site2 not in coordinates:
                        continue
                        
                    coord1 = coordinates[site1]
                    coord2 = coordinates[site2]
                    
                    # Calculate distance using Haversine formula
                    lat1, lon1 = coord1['lat'], coord1['lon']
                    lat2, lon2 = coord2['lat'], coord2['lon']
                    
                    # Convert to radians
                    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    
                    # Haversine formula
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance = 6371 * c  # Earth's radius in km
                    
                    # Get traffic volume for origin site (as per requirements)
                    site1_data = df[df['site_id'] == site1]
                    if len(site1_data) == 0:
                        continue
                        
                    # Calculate average hourly volume
                    volume_cols = [col for col in site1_data.columns if str(col).startswith('V')]
                    avg_flow = site1_data[volume_cols].mean().mean()  # vehicles per hour
                    
                    # Calculate speed based on flow using the quadratic equation
                    # flow = -1.4648375*(speed)^2 + 93.75*(speed)
                    # We need to solve for speed given flow
                    # Using quadratic formula: speed = (-b ± sqrt(b^2 - 4ac)) / (2a)
                    # where a = -1.4648375, b = 93.75, c = -flow
                    
                    a = -1.4648375
                    b = 93.75
                    c = -avg_flow
                    
                    # Calculate discriminant
                    discriminant = b**2 - 4*a*c
                    
                    if discriminant < 0:
                        # No real solutions, use speed limit
                        speed = speed_limit
                    else:
                        # Get both solutions
                        speed1 = (-b + np.sqrt(discriminant)) / (2*a)
                        speed2 = (-b - np.sqrt(discriminant)) / (2*a)
                        
                        # Choose the appropriate speed based on flow
                        if avg_flow <= 351:  # Under capacity (green line)
                            speed = min(speed1, speed2)  # Take the higher speed
                        else:  # Over capacity (red line)
                            speed = max(speed1, speed2)  # Take the lower speed
                            
                        # Cap speed at speed limit
                        speed = min(speed, speed_limit)
                    
                    # Calculate travel time (in minutes)
                    # time = distance/speed + intersection_delay
                    travel_time = (distance / speed) * 60 + (intersection_delay / 60)
                    
                    # Add bidirectional edges
                    edges.append((site1, site2, travel_time))
                    edges.append((site2, site1, travel_time))
            
            print(f"Built graph with {len(edges)} edges")
            return edges, coordinates
            
        except Exception as e:
            print(f"Error building graph: {str(e)}")
            return [], coordinates 