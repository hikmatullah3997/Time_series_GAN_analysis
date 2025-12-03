# TimeGAN Experimental Study: Household Power Consumption

## 📋 Project Overview
This repository contains the implementation of TimeGAN (Time-series Generative Adversarial Networks) for synthetic household power consumption data generation. The project implements and evaluates TimeGAN's capability to capture temporal dynamics in real-world time series data.

## 📁 File Structure

### 📊 Data Files
- `household_power_consumption.txt` - Original dataset from Kaggle
- `household_power_consumption_cleaned.csv` - Preprocessed and cleaned data
- `scaler.pkl` - MinMaxScaler object for normalization
- `real_data_clean.npy` - Processed real sequences (1423×24×7)
- `synthetic_data_clean.npy` - Generated synthetic sequences (1423×24×7)
- `real_data_small.npy` / `synthetic_data_small.npy` - Smaller versions for testing

### 🔧 Preprocessing Scripts
- `data_preprocessing.py` - Main data cleaning pipeline:
  - Loads raw data
  - Handles missing values
  - Resamples to hourly frequency
  - Normalizes features (MinMax scaling)
  - Saves cleaned dataset

- `deep_clean_data.py` - Advanced data cleaning:
  - Removes NaN/Inf values
  - Creates sequence windows (24-hour)
  - Ensures data quality
  - Generates synthetic data for comparison

### 🤖 Model & Training
- `model_training_simple.py` - Simplified TimeGAN implementation:
  - Creates synthetic data by adding noise to real sequences
  - Uses GRU-based architecture
  - Configurable parameters (hidden_dim=24, seq_len=24)
  - Saves generated synthetic data

- `timegan.py` - TimeGAN model implementation (TensorFlow compatible)

### 📈 Analysis & Visualization
- `visualization_simple.py` - Main analysis script:
  - Loads real and synthetic data
  - Calculates metrics (Wasserstein, MSE, Correlation, DTW, MMD)
  - Generates visualizations:
    - Sequence comparisons
    - Distribution plots
    - Temporal dependency analysis (ACF)
    - Daily pattern analysis

- `advanced_metrics.py` - Advanced temporal metrics:
  - Dynamic Time Warping (DTW) calculation
  - Maximum Mean Discrepancy (MMD) computation
  - Comprehensive evaluation framework

### 📊 Report Generation
- `generate_report.py` - Generates experimental study report
- `generate_complete_report.py` - Creates comprehensive report with all metrics

## 🚀 Setup Instructions

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install statsmodels  # For temporal dependency analysis
