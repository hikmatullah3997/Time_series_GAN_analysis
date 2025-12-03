# deep_clean_data.py - Thorough data cleaning
import pandas as pd
import numpy as np
import joblib

print("=== Deep Cleaning Data ===")

# 1. Load the original cleaned data and check for issues
print("Loading data...")
df_clean = pd.read_csv('household_power_consumption_cleaned.csv', index_col='DateTime', parse_dates=True)
print(f"Original data shape: {df_clean.shape}")

# 2. Check for NaN values
print(f"NaN values in original data: {df_clean.isnull().sum().sum()}")

# 3. Remove any remaining NaN values
print("Removing NaN values...")
df_clean = df_clean.dropna()
print(f"Data shape after NaN removal: {df_clean.shape}")

# 4. Check for infinite values and replace with bounds
print("Checking for infinite values...")
for col in df_clean.columns:
    if np.isinf(df_clean[col]).any():
        print(f"Found infinite values in {col}, replacing with min/max...")
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# 5. Ensure all values are within [0,1] range (due to normalization)
print("Ensuring values are in [0,1] range...")
for col in df_clean.columns:
    df_clean[col] = np.clip(df_clean[col], 0.0, 1.0)

# 6. Create sequences - but only if we have enough data
sequence_length = 24
print(f"Creating sequences of length {sequence_length}...")

real_data = []
for i in range(0, len(df_clean) - sequence_length + 1, sequence_length):  # Step by sequence_length to avoid overlap
    sequence = df_clean.iloc[i:(i + sequence_length)].values
    # Only add sequence if it has no NaN and proper length
    if len(sequence) == sequence_length and not np.isnan(sequence).any():
        real_data.append(sequence)

real_data = np.array(real_data)
print(f"Final real data sequences shape: {real_data.shape}")

# 7. Create synthetic data that's clearly different but realistic
print("Creating synthetic data...")
np.random.seed(42)

synthetic_data = real_data.copy()

# Add different types of variations to make synthetic data distinct but realistic
for i in range(synthetic_data.shape[0]):
    # Add Gaussian noise
    noise = np.random.normal(0, 0.03, synthetic_data[i].shape)
    synthetic_data[i] += noise

    # Add some temporal smoothing to make it more realistic
    for j in range(synthetic_data.shape[2]):  # For each feature
        # Apply slight smoothing to maintain temporal coherence
        if j == 0:  # Main feature - keep more structure
            smoothness = 0.7
        else:  # Other features - can vary more
            smoothness = 0.9

        # Simple moving average to maintain temporal patterns
        original = synthetic_data[i, :, j].copy()
        for k in range(1, len(original) - 1):
            synthetic_data[i, k, j] = (original[k - 1] + original[k] * 2 + original[k + 1]) / 4

# Final clip to ensure valid range
synthetic_data = np.clip(synthetic_data, 0.001, 0.999)

print(f"Synthetic data shape: {synthetic_data.shape}")

# 8. Final quality check
print("\nFinal Data Quality Check:")
print(f"Real data - NaN count: {np.isnan(real_data).sum()}")
print(f"Synthetic data - NaN count: {np.isnan(synthetic_data).sum()}")
print(f"Real data - Inf count: {np.isinf(real_data).sum()}")
print(f"Synthetic data - Inf count: {np.isinf(synthetic_data).sum()}")

# 9. Verify basic statistics work
print("\nBasic Statistics Verification:")
real_flat = real_data.reshape(-1, real_data.shape[-1])
synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])

feature_names = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("\nMean Values:")
print("Feature\t\t\tReal\t\tSynthetic")
print("-" * 55)
for i, name in enumerate(feature_names):
    real_mean = np.mean(real_flat[:, i])
    synth_mean = np.mean(synth_flat[:, i])
    print(f"{name:20} {real_mean:.4f}\t\t{synth_mean:.4f}")

# 10. Save the cleaned datasets
print("\nSaving cleaned datasets...")
np.save('real_data_clean.npy', real_data)
np.save('synthetic_data_clean.npy', synthetic_data)

# Also save a smaller version for faster testing
np.save('real_data_small.npy', real_data[:1000])
np.save('synthetic_data_small.npy', synthetic_data[:1000])

print("✓ real_data_clean.npy saved")
print("✓ synthetic_data_clean.npy saved")
print("✓ Small versions saved for testing")

print(f"\n=== Deep Cleaning Complete ===")
print(f"Final dataset sizes:")
print(f"Real data: {real_data.shape}")
print(f"Synthetic data: {synthetic_data.shape}")