# advanced_metrics.py - Calculate DTW and MMD metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_kernels
import warnings

warnings.filterwarnings('ignore')

print("=== CALCULATING DTW & MMD METRICS ===\n")

# Load data
real_data = np.load('real_data_clean.npy')
synthetic_data = np.load('synthetic_data_clean.npy')

print(f"Real data shape: {real_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

feature_names = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']


# 1. DTW (Dynamic Time Warping) Implementation
def dtw_distance(seq1, seq2):
    """Calculate DTW distance between two sequences"""
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # insertion
                                          dtw_matrix[i, j - 1],  # deletion
                                          dtw_matrix[i - 1, j - 1])  # match
    return dtw_matrix[n, m]


def calculate_dtw_batch(real_seqs, synth_seqs, n_samples=100):
    """Calculate average DTW distance between sample sequences"""
    np.random.seed(42)
    real_samples = real_seqs[np.random.choice(len(real_seqs), n_samples, replace=False)]
    synth_samples = synth_seqs[np.random.choice(len(synth_seqs), n_samples, replace=False)]

    dtw_distances = []
    for i in range(n_samples):
        # Calculate DTW for each feature and average
        feature_dtws = []
        for feature_idx in range(real_seqs.shape[2]):
            dtw_val = dtw_distance(real_samples[i, :, feature_idx],
                                   synth_samples[i, :, feature_idx])
            feature_dtws.append(dtw_val)
        dtw_distances.append(np.mean(feature_dtws))

    return np.mean(dtw_distances), np.std(dtw_distances)


# 2. MMD (Maximum Mean Discrepancy) Implementation
def mmd_rbf(X, Y, gamma=1.0):
    """Calculate MMD using RBF kernel"""
    XX = pairwise_kernels(X, X, metric='rbf', gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric='rbf', gamma=gamma)
    XY = pairwise_kernels(X, Y, metric='rbf', gamma=gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


def calculate_mmd_features(real_flat, synth_flat, n_samples=1000):
    """Calculate MMD for each feature"""
    np.random.seed(42)

    # Sample for computational efficiency
    if len(real_flat) > n_samples:
        real_idx = np.random.choice(len(real_flat), n_samples, replace=False)
        synth_idx = np.random.choice(len(synth_flat), n_samples, replace=False)
        real_sampled = real_flat[real_idx]
        synth_sampled = synth_flat[synth_idx]
    else:
        real_sampled = real_flat
        synth_sampled = synth_flat

    mmd_results = {}
    for i, name in enumerate(feature_names):
        # Reshape for MMD calculation
        X = real_sampled[:, i].reshape(-1, 1)
        Y = synth_sampled[:, i].reshape(-1, 1)

        mmd_val = mmd_rbf(X, Y)
        mmd_results[name] = mmd_val

    return mmd_results


# 3. Calculate all metrics
print("Calculating DTW distances...")
real_flat = real_data.reshape(-1, real_data.shape[-1])
synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])

# DTW calculation (takes a while, so we use sampling)
dtw_mean, dtw_std = calculate_dtw_batch(real_data, synthetic_data, n_samples=50)
print(f"✓ Average DTW distance: {dtw_mean:.4f} ± {dtw_std:.4f}")

print("\nCalculating MMD distances...")
mmd_results = calculate_mmd_features(real_flat, synth_flat, n_samples=1000)

# 4. Print results
print("\n" + "=" * 60)
print("ADVANCED TEMPORAL METRICS")
print("=" * 60)

print(f"\nDynamic Time Warping (DTW):")
print(f"Average DTW distance: {dtw_mean:.4f} ± {dtw_std:.4f}")
print("(Lower values indicate better temporal pattern matching)")

print(f"\nMaximum Mean Discrepancy (MMD) by feature:")
print("Feature\t\t\t\tMMD")
print("-" * 50)
for name in feature_names:
    print(f"{name:25} {mmd_results[name]:.6f}")

# 5. Visualization: DTW alignment example
print("\nGenerating DTW visualization example...")


def plot_dtw_example(real_seq, synth_seq, feature_name, sequence_idx=0):
    """Plot DTW alignment between two sequences"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot sequences
    ax1.plot(real_seq, 'b-', linewidth=2, label='Real', marker='o')
    ax1.plot(synth_seq, 'r-', linewidth=2, label='Synthetic', marker='s')
    ax1.set_title(f'DTW Comparison: {feature_name} (Sequence {sequence_idx})')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calculate and show cumulative DTW path
    n, m = len(real_seq), len(synth_seq)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(real_seq[i - 1] - synth_seq[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])

    # Plot DTW matrix
    im = ax2.imshow(dtw_matrix[1:, 1:], cmap='viridis', aspect='auto',
                    origin='lower', interpolation='nearest')
    ax2.set_title('DTW Cost Matrix')
    ax2.set_xlabel('Synthetic Sequence Index')
    ax2.set_ylabel('Real Sequence Index')
    plt.colorbar(im, ax=ax2, label='Cumulative Cost')

    plt.tight_layout()
    plt.savefig(f'dtw_comparison_{feature_name.replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


# Plot example for main feature
example_idx = 10  # Use a specific sequence for consistent visualization
plot_dtw_example(real_data[example_idx, :, 0],
                 synthetic_data[example_idx, :, 0],
                 feature_names[0],
                 example_idx)

# 6. Generate comprehensive metrics report
print("\n" + "=" * 60)
print("COMPREHENSIVE METRICS SUMMARY")
print("=" * 60)

from scipy.stats import wasserstein_distance, pearsonr
from sklearn.metrics import mean_squared_error

# Calculate all metrics
comprehensive_metrics = {}
for i, name in enumerate(feature_names):
    comprehensive_metrics[name] = {
        'wasserstein': wasserstein_distance(real_flat[:, i], synth_flat[:, i]),
        'mse': mean_squared_error(real_flat[:, i], synth_flat[:, i]),
        'correlation': pearsonr(real_flat[:, i], synth_flat[:, i])[0],
        'mmd': mmd_results[name]
    }

print("\nFeature-wise Comprehensive Metrics:")
print("Feature\t\t\tWasserstein\tMSE\t\tCorrelation\tMMD")
print("-" * 85)
for name in feature_names:
    m = comprehensive_metrics[name]
    print(f"{name:20} {m['wasserstein']:.4f}\t\t{m['mse']:.4f}\t\t{m['correlation']:.4f}\t\t{m['mmd']:.6f}")

print(f"\nOverall DTW: {dtw_mean:.4f} ± {dtw_std:.4f}")

# Save metrics to file for report
metrics_summary = {
    'dtw_mean': dtw_mean,
    'dtw_std': dtw_std,
    'feature_metrics': comprehensive_metrics
}

import json

with open('advanced_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print("\n✓ Advanced metrics calculated and saved!")
print("✓ DTW visualization generated")
print("✓ Metrics saved to 'advanced_metrics.json'")
