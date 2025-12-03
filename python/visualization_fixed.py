# visualization_simple.py - Simple and robust visualization WITH TEMPORAL ANALYSIS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance, pearsonr
import warnings

warnings.filterwarnings('ignore')

print("=== Simple Visualization & Analysis ===\n")

# 1. Load the CLEAN data
print("Loading clean data...")
try:
    real_data = np.load('real_data_clean.npy')
    synthetic_data = np.load('synthetic_data_clean.npy')
    print("✓ Loaded full datasets")
except:
    # Fallback to small datasets
    real_data = np.load('real_data_small.npy')
    synthetic_data = np.load('synthetic_data_small.npy')
    print("✓ Loaded small datasets")

print(f"Real data shape: {real_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

# Quick validation
assert np.isnan(real_data).sum() == 0, "Real data has NaN!"
assert np.isnan(synthetic_data).sum() == 0, "Synthetic data has NaN!"

# 2. Flatten for analysis
real_flat = real_data.reshape(-1, real_data.shape[-1])
synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])

feature_names = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 3. Basic Statistics
print("\n" + "=" * 50)
print("BASIC STATISTICS COMPARISON")
print("=" * 50)

print("\nMean Values:")
print("Feature\t\t\tReal\t\tSynthetic\tDiff")
print("-" * 65)
for i, name in enumerate(feature_names):
    real_mean = np.mean(real_flat[:, i])
    synth_mean = np.mean(synth_flat[:, i])
    diff = abs(real_mean - synth_mean)
    print(f"{name:20} {real_mean:.4f}\t\t{synth_mean:.4f}\t\t{diff:.4f}")

# 4. Quantitative Metrics
print("\n" + "=" * 50)
print("QUANTITATIVE METRICS")
print("=" * 50)

print("\nWasserstein Distance (lower = better):")
print("Feature\t\t\tDistance")
print("-" * 40)
for i, name in enumerate(feature_names):
    w_dist = wasserstein_distance(real_flat[:, i], synth_flat[:, i])
    print(f"{name:20} {w_dist:.6f}")

print("\nMean Squared Error (lower = better):")
print("Feature\t\t\tMSE")
print("-" * 40)
for i, name in enumerate(feature_names):
    mse = mean_squared_error(real_flat[:, i], synth_flat[:, i])
    print(f"{name:20} {mse:.6f}")

print("\nPearson Correlation (higher = better):")
print("Feature\t\t\tCorrelation")
print("-" * 40)
for i, name in enumerate(feature_names):
    corr, _ = pearsonr(real_flat[:, i], synth_flat[:, i])
    print(f"{name:20} {corr:.4f}")

# 5. Simple Visualization 1: Sequence Comparison
print("\n" + "=" * 50)
print("GENERATING VISUALIZATIONS")
print("=" * 50)

print("Creating sequence comparison plot...")
plt.figure(figsize=(12, 8))

# Plot first 3 sequences of main feature
for i in range(3):
    plt.subplot(3, 2, i * 2 + 1)
    plt.plot(real_data[i, :, 0], 'b-', linewidth=2, label='Real')
    plt.title(f'Real Sequence {i + 1}')
    plt.xlabel('Hour')
    plt.ylabel('Global Active Power')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 2, i * 2 + 2)
    plt.plot(synthetic_data[i, :, 0], 'r-', linewidth=2, label='Synthetic')
    plt.title(f'Synthetic Sequence {i + 1}')
    plt.xlabel('Hour')
    plt.ylabel('Global Active Power')
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig('sequence_comparison_simple.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ sequence_comparison_simple.png saved")

# 6. Simple Visualization 2: Distribution Comparison
print("Creating distribution comparison plot...")
plt.figure(figsize=(15, 10))

for i, name in enumerate(feature_names):
    plt.subplot(3, 3, i + 1)
    plt.hist(real_flat[:, i], bins=50, alpha=0.7, label='Real', density=True, color='blue')
    plt.hist(synth_flat[:, i], bins=50, alpha=0.7, label='Synthetic', density=True, color='red')
    plt.title(f'Distribution: {name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_comparison_simple.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ distribution_comparison_simple.png saved")

# 7. TEMPORAL DEPENDENCY ANALYSIS (NEW SECTION)
print("\n" + "=" * 50)
print("TEMPORAL DEPENDENCY ANALYSIS")
print("=" * 50)

# Try to import statsmodels for ACF, but provide fallback
try:
    from statsmodels.tsa.stattools import acf

    STATSMODELS_AVAILABLE = True
    print("✓ Statsmodels available - generating ACF plots...")
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠ Statsmodels not available - install with: pip install statsmodels")
    print("Skipping ACF plots...")


def calculate_simple_autocorrelation(series, max_lag=12):
    """Calculate simple autocorrelation without statsmodels"""
    autocorrs = []
    n = len(series)
    mean_val = np.mean(series)

    for lag in range(max_lag + 1):
        if lag == 0:
            autocorrs.append(1.0)
        else:
            numerator = np.sum((series[lag:] - mean_val) * (series[:-lag] - mean_val))
            denominator = np.sum((series - mean_val) ** 2)
            autocorrs.append(numerator / denominator)

    return autocorrs


def plot_temporal_dependencies(real_flat, synth_flat, feature_names):
    """Plot autocorrelation functions for temporal dependency analysis"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flat

    max_lag = 12  # Analyze up to 12-hour dependencies

    for i, name in enumerate(feature_names):
        if STATSMODELS_AVAILABLE:
            # Use statsmodels for accurate ACF
            real_acf = acf(real_flat[:, i], nlags=max_lag, fft=True)
            synth_acf = acf(synth_flat[:, i], nlags=max_lag, fft=True)
        else:
            # Use simple autocorrelation calculation
            real_acf = calculate_simple_autocorrelation(real_flat[:, i], max_lag)
            synth_acf = calculate_simple_autocorrelation(synth_flat[:, i], max_lag)

        # Plot ACF comparison
        lags = range(len(real_acf))
        axes[i].plot(lags, real_acf, 'bo-', label='Real', linewidth=2, markersize=4)
        axes[i].plot(lags, synth_acf, 'ro-', label='Synthetic', linewidth=2, markersize=4)
        axes[i].set_title(f'Temporal Dep: {name}')
        axes[i].set_xlabel('Lag (hours)')
        axes[i].set_ylabel('Autocorrelation')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Add correlation value for lag-1 (most important temporal dependency)
        lag1_corr_real = real_acf[1] if len(real_acf) > 1 else 0
        lag1_corr_synth = synth_acf[1] if len(synth_acf) > 1 else 0
        axes[i].text(0.05, 0.95, f'Lag-1: R={lag1_corr_real:.3f}\nS={lag1_corr_synth:.3f}',
                     transform=axes[i].transAxes, fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Remove empty subplots
    for i in range(len(feature_names), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('temporal_dependency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ temporal_dependency_analysis.png saved")


# Generate temporal dependency plots
plot_temporal_dependencies(real_flat, synth_flat, feature_names)

# 8. Additional Temporal Analysis: Lag Analysis
print("\nPerforming lag correlation analysis...")


def analyze_lag_correlations(real_flat, synth_flat, feature_names):
    """Analyze how well temporal dependencies are preserved across lags"""
    print("\nLag Correlation Analysis:")
    print("Feature\t\t\tLag-1 Corr Real\tLag-1 Corr Synth\tDifference")
    print("-" * 80)

    for i, name in enumerate(feature_names):
        if STATSMODELS_AVAILABLE:
            real_acf = acf(real_flat[:, i], nlags=3, fft=True)
            synth_acf = acf(synth_flat[:, i], nlags=3, fft=True)
        else:
            real_acf = calculate_simple_autocorrelation(real_flat[:, i], 3)
            synth_acf = calculate_simple_autocorrelation(synth_flat[:, i], 3)

        lag1_real = real_acf[1] if len(real_acf) > 1 else 0
        lag1_synth = synth_acf[1] if len(synth_acf) > 1 else 0
        diff = abs(lag1_real - lag1_synth)

        print(f"{name:20} {lag1_real:.4f}\t\t{lag1_synth:.4f}\t\t{diff:.4f}")


analyze_lag_correlations(real_flat, synth_flat, feature_names)

# 9. Seasonality Analysis
print("\nPerforming basic seasonality analysis...")


def analyze_daily_patterns(real_data, synthetic_data):
    """Analyze daily patterns by averaging across sequences"""
    real_daily_avg = np.mean(real_data, axis=0)  # Average across all days
    synth_daily_avg = np.mean(synthetic_data, axis=0)

    plt.figure(figsize=(15, 10))

    for i, name in enumerate(feature_names[:4]):  # Plot first 4 features
        plt.subplot(2, 2, i + 1)
        plt.plot(real_daily_avg[:, i], 'b-', linewidth=2, label='Real Daily Avg')
        plt.plot(synth_daily_avg[:, i], 'r-', linewidth=2, label='Synthetic Daily Avg')
        plt.title(f'Daily Pattern: {name}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Calculate pattern similarity
        pattern_corr = np.corrcoef(real_daily_avg[:, i], synth_daily_avg[:, i])[0, 1]
        plt.text(0.05, 0.95, f'Pattern Corr: {pattern_corr:.3f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig('daily_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ daily_pattern_analysis.png saved")


analyze_daily_patterns(real_data, synthetic_data)

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE!")
print("=" * 50)
print("\nGenerated files:")
print("✓ sequence_comparison_simple.png")
print("✓ distribution_comparison_simple.png")
print("✓ temporal_dependency_analysis.png")
print("✓ daily_pattern_analysis.png")
print("\nKey metrics calculated:")
print("✓ Mean comparison")
print("✓ Wasserstein Distance")
print("✓ Mean Squared Error")
print("✓ Pearson Correlation")
print("✓ Temporal dependency analysis (ACF)")
print("✓ Lag correlation analysis")
print("✓ Daily pattern analysis")
print("\nYou now have COMPLETE temporal dynamics analysis for your report!")