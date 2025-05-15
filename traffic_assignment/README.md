# Traffic-based Route Guidance System (TBRGS)

A machine learning-based traffic prediction and route guidance system for the Boroondara area. This project was developed as part of COS30019 - Introduction to AI Assignment 2.

## Features

- **Traffic Prediction**
  - LSTM-based traffic volume prediction
  - 15-minute interval predictions
  - Visual prediction plots
  - Multiple SCATS site support

- **Route Planning**
  - Multiple route options between SCATS sites
  - Interactive map visualization
  - Cost-based route optimization
  - Real-time traffic consideration

## Project Structure

```
tbrgs/
├── data/               # Data storage
├── models/            # Trained ML models
├── src/               # Source code
│   ├── data/         # Data processing modules
│   ├── ml/           # Machine learning models
│   ├── gui/          # GUI implementation
│   ├── routing/      # Route finding algorithms
│   └── utils/        # Utility functions
├── tests/            # Test cases
└── config/           # Configuration files
```

## Requirements

- Python 3.8+
- PyQt5
- TensorFlow
- Pandas
- NumPy
- scikit-learn
- Folium
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tbrgs.git
cd tbrgs
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python src/main.py
```

2. For training ML models:
```bash
python src/ml/train_models.py
```

## Data

The system uses traffic flow data from VicRoads for the Boroondara area. The data includes:
- Traffic volume at intersections (15-minute intervals)
- SCATS site information
- Geographic coordinates

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is part of COS30019 - Introduction to AI Assignment 2.

## Author

Suman Sutparai (105106819) 