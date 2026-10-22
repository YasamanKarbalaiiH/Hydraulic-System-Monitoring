# Hydraulic System Monitoring

A Python-based data analysis and monitoring project for hydraulic systems. The project analyzes sensor measurements, extracts statistical features, compares operating cycles with healthy reference data, and identifies cycles that may require further inspection.

## Overview

The project uses hydraulic system sensor data to analyze system behavior across multiple operating cycles. It includes data preprocessing, feature extraction, statistical analysis, distance-based comparisons, and an interactive Streamlit dashboard.

## Features

- **Sensor Data Analysis:** Processes measurements from 17 hydraulic system sensors.
- **Feature Extraction:** Calculates five statistical features for each sensor: mean, minimum, maximum, standard deviation, and root mean square (RMS).
- **Data Standardization:** Standardizes features using statistics calculated from training data.
- **Correlation Analysis:** Examines relationships between features and identifies highly correlated variables.
- **Healthy Reference Comparison:** Compares operating cycles with healthy reference data using distance-based methods.
- **Alert Classification:** Assigns alert levels to cycles based on their distance from healthy reference behavior.
- **Cycle Comparison:** Identifies cycles with similar characteristics and highlights significant changes between consecutive cycles.
- **Evaluation:** Reports evaluation metrics for the implemented detection approach.
- **Interactive Dashboard:** Provides a Streamlit interface for exploring analysis results.

## Technologies Used

- Python
- NumPy
- Pandas
- Streamlit

## Project Structure

```text
Hydraulic-System-Monitoring/
├── app.py
├── logic.py
├── dataset/
│   ├── CE.txt
│   ├── CP.txt
│   ├── EPS1.txt
│   ├── FS1.txt
│   ├── FS2.txt
│   ├── PS1.txt
│   ├── PS2.txt
│   ├── PS3.txt
│   ├── PS4.txt
│   ├── PS5.txt
│   ├── PS6.txt
│   ├── SE.txt
│   ├── TS1.txt
│   ├── TS2.txt
│   ├── TS3.txt
│   ├── TS4.txt
│   ├── VS1.txt
│   ├── profile.txt
│   ├── description.txt
│   └── documentation.txt
├── outputs/
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## Dataset

The project processes measurements from 17 sensors and uses five equipment-condition parameters from `profile.txt`.

For each operating cycle, five statistical features are extracted from each sensor, resulting in **85 features per cycle**.

The analysis uses the sensor measurements and equipment-condition profiles to examine operating behavior and compare cycles against healthy reference data.

**Note:** Check the dataset's original license and redistribution terms before sharing the raw data publicly.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YasamanKarbalaiiH/Hydraulic-System-Monitoring.git
```

### 2. Navigate to the project directory

```bash
cd Hydraulic-System-Monitoring
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit dashboard with:

```bash
streamlit run app.py
```

The application will open in your default browser.

Make sure the required dataset files are available in the expected `dataset/` directory before running the application.

## Output

The analysis can produce CSV files containing results such as:

- Extracted features
- Standardization parameters
- Training and test data
- Correlation analysis
- Healthy reference distances
- Cycle changes and similar cycles
- Alert classifications
- Evaluation results

These outputs support further analysis of system behavior and potential anomalies.

## Limitations

The results depend on the quality and representativeness of the available data and healthy reference cycles. Distance-based alerts are indicators for further investigation, not definitive diagnoses of equipment faults.

Evaluation metrics should be interpreted in the context of the number of healthy reference samples, class imbalance, and the evaluation procedure.

## Author

**Yasaman Karbalaii**

GitHub: [YasamanKarbalaiiH](https://github.com/YasamanKarbalaiiH)

---

_This project was developed for educational and data analysis purposes._
