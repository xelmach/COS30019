#105106819 Suman Sutparai
# Traffic Assignment System

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/xelmach/COS30019.git
   cd COS30019/Traffic_Assignment
   ```

2. **Create a virtual environment:**
   ```sh
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```sh
   python main.py
   ```

## Notes
- Do **not** upload the `venv` folder to GitHub.
- If you add new dependencies, run `pip freeze > requirements.txt` and commit the updated file.
- For any issues, contact Suman Sutparai.

## Features

- Traffic prediction using LSTM and GRU models
- Route planning with multiple options
- Interactive map visualization
- Real-time traffic data processing
- User-friendly GUI interface

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