# Traffic-based Route Guidance System (TBRGS)

A Python-based application for traffic prediction and route planning using machine learning models.

## Features

- Traffic prediction using LSTM and GRU models
- Route planning with multiple options
- Interactive map visualization
- Real-time traffic data processing
- User-friendly GUI interface

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Use the GUI to:
   - Select origin and destination points
   - View predicted traffic conditions
   - Get route recommendations
   - Adjust settings as needed

## Project Structure

- `data/`: Contains raw and processed data files
- `models/`: Stores trained ML models
- `config/`: Configuration files
- `logs/`: Application logs
- `main.py`: Application entry point
- `main_window.py`: GUI implementation
- `processor.py`: Data processing module
- `lstm_model.py`: LSTM model implementation
- `gru_model.py`: GRU model implementation
- `train_models.py`: Model training script
- `config_loader.py`: Configuration management
- `logger.py`: Logging setup

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies

## Author

Suman Sutparai (105106819) 