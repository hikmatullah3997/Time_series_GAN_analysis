# model_training_simple.py - Simplified approach for testing
import pandas as pd
import numpy as np
import time

# 1. Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv('household_power_consumption_cleaned.csv', index_col='DateTime', parse_dates=True)
print(f"Data shape: {df.shape}")

# 2. Prepare data (we'll use this for visualization later)
sequence_length = 24
features = df.shape[1]

# Create sequences
data = []
for i in range(0, len(df) - sequence_length + 1):
    data.append(df.iloc[i:(i + sequence_length)].values)

data = np.array(data)
print(f"Data shape after sequencing: {data.shape}")

# 3. For now, let's create some simple synthetic data for testing
# This is just a placeholder so we can proceed with visualization
print("Creating simple synthetic data for testing...")

# Simple approach: add small random noise to real data
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, 0.1, data.shape)  # Small noise
synthetic_data = data + noise

# Clip values to ensure they stay in [0,1] range (due to normalization)
synthetic_data = np.clip(synthetic_data, 0, 1)

print(f"Original data shape: {data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

# 4. Save the synthetic data
np.save('synthetic_data.npy', synthetic_data)
print("Synthetic data saved to 'synthetic_data.npy'")

print("\n--- Data Preparation Complete ---")
print("We now have both real and synthetic data for visualization!")
