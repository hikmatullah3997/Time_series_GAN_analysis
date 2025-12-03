# data_preprocessing.py
import pandas as pd
import numpy as np

# 1. Load the Data
print("Loading data...")
# Note: This file is separated by semicolons ';' and has missing values marked as '?'
df = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False, na_values=['?'])

print(f"Data loaded. Original shape: {df.shape}")

# 2. Handle Missing Values
print("Handling missing values...")
# Count missing values before cleaning
print(f"Missing values before cleaning:\n{df.isnull().sum()}")

# Drop rows with any missing values (this is a simple approach, you can get more sophisticated later)
df_clean = df.dropna()

print(f"Data shape after removing missing values: {df_clean.shape}")

# 3. Create a DateTime Index
print("Creating datetime index...")
# Combine the Date and Time columns into a single datetime object
df_clean['DateTime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time'])
df_clean = df_clean.set_index('DateTime')  # Set this new column as the index
df_clean = df_clean.drop(['Date', 'Time'], axis=1)  # Remove the old Date and Time columns

# 4. Convert Data to Numeric (just to be safe)
print("Converting data to numeric types...")
for col in df_clean.columns:
    df_clean[col] = pd.to_numeric(df_clean[col])

# 5. Resample the Data (Let's start with hourly averages to make it manageable)
print("Resampling to hourly frequency...")
# This averages the values for each hour, reducing the data size and smoothing it.
df_hourly = df_clean.resample('1H').mean()

print(f"Data shape after resampling: {df_hourly.shape}")

# 6. Normalize the Data (CRITICAL for Neural Networks)
print("Normalizing data...")
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_hourly)

# Convert back to a DataFrame for easier handling
df_normalized = pd.DataFrame(data_scaled, columns=df_hourly.columns, index=df_hourly.index)

print("Preview of normalized data:")
print(df_normalized.head())

# 7. Save the Cleaned Data
print("Saving cleaned data...")
df_normalized.to_csv('household_power_consumption_cleaned.csv')
print("Cleaned data saved to 'household_power_consumption_cleaned.csv'")

print("\n--- Preprocessing Complete ---")
print(f"Final dataset shape: {df_normalized.shape}")
print(f"Date range: {df_normalized.index.min()} to {df_normalized.index.max()}")