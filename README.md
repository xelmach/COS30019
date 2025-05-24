# Traffic-based Route Guidance System (TBRGS)

A Python application for traffic prediction and route planning using machine learning.

## Features

- Traffic flow prediction using GRU and LSTM models
- Route planning with multiple alternative routes
- Interactive map visualization
- Real-time traffic data processing

## Requirements

- Python 3.8+
- PyQt5
- TensorFlow
- Pandas
- NumPy
- Folium
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python Traffic_Assignment/main.py
```

2. Use the GUI to:
   - Select origin and destination sites
   - Find optimal routes
   - Predict traffic flow
   - View results on the map

## Project Structure

- `Traffic_Assignment/`: Main application directory
  - `main.py`: Application entry point
  - `main_window.py`: GUI implementation
  - `processor.py`: Data processing
  - `route_finder.py`: Route finding algorithms
  - `train_models.py`: ML model training
  - `config_loader.py`: Configuration management
  - `logger.py`: Logging utilities

## Author

Suman Sutparai (105106819) 