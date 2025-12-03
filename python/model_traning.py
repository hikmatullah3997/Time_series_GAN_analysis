# model_training.py
import pandas as pd
import numpy as np
from timegan import timegan
from sklearn.model_selection import train_test_split
import time

# 1. Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv('household_power_consumption_cleaned.csv', index_col='DateTime', parse_dates=True)
print(f"Data shape: {df.shape}")

# 2. Prepare data for TimeGAN (Convert to 3D format)
print("Preparing data for TimeGAN...")

# TimeGAN expects data in shape [samples, sequence_length, features]
# Let's use a sequence length of 24 hours (1 day)
sequence_length = 24
features = df.shape[1]  # Should be 7

# Create sequences of 24 hours each
data = []
for i in range(0, len(df) - sequence_length + 1):
    data.append(df.iloc[i:(i + sequence_length)].values)

data = np.array(data)
print(f"Data shape after sequencing: {data.shape}")  # Should be (number_of_sequences, 24, 7)

# 3. Split the data (use first 80% for training, keep 20% for later comparison)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# 4. Set TimeGAN parameters
print("Setting up TimeGAN parameters...")

# These are the hyperparameters we need to define
parameters = {
    'module': 'gru',           # Type of RNN cell: gru or lstm
    'hidden_dim': 24,          # Size of hidden state in RNN (smaller for faster training)
    'num_layer': 3,            # Number of RNN layers
    'iterations': 1000,        # Number of training iterations (start small for testing)
    'batch_size': 128,         # Size of mini-batch
    'seq_len': sequence_length,# Our sequence length (24 hours)
    'learning_rate': 0.001,    # Learning rate for optimizer
    'metric': 'euclidean',     # Metric for supervised loss
    'gamma': 1,                # Hyperparameter for unsupervised loss
}

print("TimeGAN Parameters:")
for key, value in parameters.items():
    print(f"  {key}: {value}")

# 5. Train TimeGAN
print("\nStarting TimeGAN training...")
print("This will take a while (especially without GPU)...")
print("You can monitor progress by watching the loss values...")

start_time = time.time()

# Train the model - REMOVED the model_name parameter
synthetic_data = timegan(train_data, parameters)

end_time = time.time()
training_time = (end_time - start_time) / 60  # Convert to minutes
print(f"Training completed in {training_time:.2f} minutes!")

print(f"Original training data shape: {train_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

# 6. Save the synthetic data
print("Saving synthetic data...")
np.save('synthetic_data.npy', synthetic_data)
print("Synthetic data saved to 'synthetic_data.npy'")

print("\n--- Model Training Complete ---")