import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

df = pd.read_csv('WindSpeed.csv')
ws = df['WindSpeed'].values
n = len(ws)

print("=" * 60)
print("WIND SPEED DATASET — COMPREHENSIVE EDA")
print("=" * 60)

# ===================== 1. BASIC STATISTICS =====================
print("\n1. BASIC STATISTICS")
print("-" * 40)
print(f"   Total observations: {n}")
print(f"   Range: {ws.min():.2f} — {ws.max():.2f} m/s")
print(f"   Mean: {ws.mean():.4f} m/s")
print(f"   Median: {np.median(ws):.4f} m/s")
print(f"   Std Dev: {ws.std():.4f} m/s")
print(f"   Variance: {ws.var():.4f}")
print(f"   Skewness: {stats.skew(ws):.4f}")
print(f"   Kurtosis: {stats.kurtosis(ws):.4f}")
print(f"   Missing values: {np.isnan(ws).sum()}")
print(f"   IQR: {np.percentile(ws, 75) - np.percentile(ws, 25):.4f}")

q25, q75 = np.percentile(ws, 25), np.percentile(ws, 75)
iqr = q75 - q25
lower = q25 - 1.5 * iqr
upper = q75 + 1.5 * iqr
outliers = np.sum((ws < lower) | (ws > upper))
print(f"   Outliers (1.5*IQR): {outliers} ({100*outliers/n:.2f}%)")

# ===================== 2. DISTRIBUTION ANALYSIS =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution Analysis', fontsize=14, fontweight='bold')

# Histogram
ax = axes[0, 0]
ax.hist(ws, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
x_range = np.linspace(ws.min(), ws.max(), 200)
ax.plot(x_range, stats.norm.pdf(x_range, ws.mean(), ws.std()), 'r-', lw=2, label='Normal fit')
ax.plot(x_range, stats.lognorm.pdf(x_range, *stats.lognorm.fit(ws)), 'g--', lw=2, label='Log-normal fit')
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Density')
ax.set_title('Histogram + Distribution Fit')
ax.legend()

# Q-Q Plot
ax = axes[0, 1]
stats.probplot(ws, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normal)')
ax.get_lines()[0].set(markerfacecolor='steelblue', markeredgecolor='steelblue', markersize=2)
ax.get_lines()[1].set(color='red', linewidth=1.5)

# Box plot
ax = axes[1, 0]
bp = ax.boxplot(ws, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set(facecolor='lightsteelblue')
bp['medians'][0].set(color='red', linewidth=2)
ax.set_ylabel('Wind Speed (m/s)')
ax.set_title('Box Plot')
ax.set_xticklabels(['WindSpeed'])

# ECDF
ax = axes[1, 1]
sorted_ws = np.sort(ws)
ecdf = np.arange(1, n + 1) / n
ax.plot(sorted_ws, ecdf, color='steelblue', linewidth=1.5)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Median')
ax.axvline(x=np.median(ws), color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Empirical CDF')
ax.legend()

plt.tight_layout()
plt.savefig('eda_01_distribution.png', bbox_inches='tight')
plt.close()
print("\n   Saved: eda_01_distribution.png")

# ===================== 3. TIME SERIES PLOT =====================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle('Time Series Overview', fontsize=14, fontweight='bold')

# Full series
ax = axes[0]
ax.plot(ws, linewidth=0.3, color='steelblue', alpha=0.8)
ax.set_ylabel('Wind Speed (m/s)')
ax.set_title('Full Time Series')
ax.set_xlabel('Time Step')
ax.grid(True, alpha=0.3)

# Rolling statistics
ax = axes[1]
window = 50
rolling_mean = pd.Series(ws).rolling(window).mean()
rolling_std = pd.Series(ws).rolling(window).std()
ax.plot(ws, linewidth=0.3, color='steelblue', alpha=0.4, label='Raw')
ax.plot(rolling_mean, color='red', linewidth=1.5, label=f'Rolling Mean ({window})')
ax.fill_between(range(n), rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std,
                alpha=0.2, color='red', label='±2σ Band')
ax.set_ylabel('Wind Speed (m/s)')
ax.set_title('Rolling Statistics (window=50)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# First difference
ax = axes[2]
diff = np.diff(ws)
ax.plot(diff, linewidth=0.3, color='darkorange', alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Δ Wind Speed')
ax.set_title('First Difference (differenced series)')
ax.set_xlabel('Time Step')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_02_timeseries.png', bbox_inches='tight')
plt.close()
print("   Saved: eda_02_timeseries.png")

# ===================== 4. AUTOCORRELATION =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Autocorrelation Analysis', fontsize=14, fontweight='bold')

# ACF
ax = axes[0, 0]
lags = 100
acf_vals = acf(ws, nlags=lags)
ax.bar(range(lags + 1), acf_vals, width=0.8, color='steelblue', alpha=0.7)
ci = 1.96 / np.sqrt(n)
ax.axhline(y=ci, color='red', linestyle='--', linewidth=1)
ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1)
ax.set_title('Autocorrelation Function (ACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_xlim(0, lags)

# PACF
ax = axes[0, 1]
pacf_vals = pacf(ws, nlags=lags)
ax.bar(range(lags + 1), pacf_vals, width=0.8, color='darkorange', alpha=0.7)
ax.axhline(y=ci, color='red', linestyle='--', linewidth=1)
ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1)
ax.set_title('Partial Autocorrelation Function (PACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('PACF')
ax.set_xlim(0, lags)

# ACF of differenced series
ax = axes[1, 0]
acf_diff = acf(diff, nlags=lags)
ax.bar(range(lags + 1), acf_diff, width=0.8, color='steelblue', alpha=0.7)
ax.axhline(y=ci, color='red', linestyle='--', linewidth=1)
ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1)
ax.set_title('ACF of Differenced Series')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_xlim(0, lags)

# Lag plot (lag=1)
ax = axes[1, 1]
ax.scatter(ws[:-1], ws[1:], s=1, alpha=0.3, color='steelblue')
ax.plot([ws.min(), ws.max()], [ws.min(), ws.max()], 'r--', linewidth=1)
ax.set_xlabel('Wind Speed(t)')
ax.set_ylabel('Wind Speed(t+1)')
ax.set_title('Lag Plot (lag=1)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_03_autocorrelation.png', bbox_inches='tight')
plt.close()
print("   Saved: eda_03_autocorrelation.png")

# ===================== 5. STATIONARITY TESTS =====================
print("\n2. STATIONARITY TESTS")
print("-" * 40)

# ADF test on original
adf_orig = adfuller(ws, autolag='AIC')
print(f"   Original Series:")
print(f"     ADF Statistic: {adf_orig[0]:.4f}")
print(f"     p-value: {adf_orig[1]:.6f}")
print(f"     Lags used: {adf_orig[2]}")
for k, v in adf_orig[4].items():
    print(f"     Critical Value ({k}): {v:.4f}")
print(f"     Stationary: {'Yes' if adf_orig[1] < 0.05 else 'No'}")

# ADF test on differenced
adf_diff = adfuller(diff, autolag='AIC')
print(f"\n   Differenced Series:")
print(f"     ADF Statistic: {adf_diff[0]:.4f}")
print(f"     p-value: {adf_diff[1]:.6f}")
print(f"     Stationary: {'Yes' if adf_diff[1] < 0.05 else 'No'}")

# ===================== 6. SPECTRAL ANALYSIS =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Spectral Analysis', fontsize=14, fontweight='bold')

# Periodogram
ax = axes[0]
fft_vals = np.fft.rfft(ws - ws.mean())
freqs = np.fft.rfftfreq(n)
power = np.abs(fft_vals) ** 2
ax.plot(1/freqs[1:], power[1:], color='steelblue', linewidth=0.8)
ax.set_xlabel('Period (time steps)')
ax.set_ylabel('Power')
ax.set_title('Periodogram (FFT)')
ax.set_xlim(0, 500)
ax.grid(True, alpha=0.3)

# Cumulative periodogram
ax = axes[1]
cum_power = np.cumsum(power[1:]) / np.sum(power[1:])
ax.plot(1/freqs[1:], cum_power, color='darkorange', linewidth=1.5)
ax.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
ax.set_xlabel('Period (time steps)')
ax.set_ylabel('Cumulative Power')
ax.set_title('Cumulative Periodogram')
ax.set_xlim(0, 500)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_04_spectral.png', bbox_inches='tight')
plt.close()
print("\n   Saved: eda_04_spectral.png")

# Dominant periods
top_idx = np.argsort(power[1:])[::-1][:5]
print("\n3. DOMINANT PERIODS (FFT)")
print("-" * 40)
for i, idx in enumerate(top_idx):
    period = 1 / freqs[idx + 1]
    print(f"   #{i+1}: Period = {period:.1f} steps, Power = {power[idx+1]:.2f}")

# ===================== 7. WINDOW ANALYSIS =====================
print("\n4. WINDOW SIZE ANALYSIS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Window Size Analysis', fontsize=14, fontweight='bold')

for idx, ws_size in enumerate([7, 14, 21, 28]):
    ax = axes[idx // 2, idx % 2]
    # Autocorrelation decay
    autocorr = [np.corrcoef(ws[:-lag], ws[lag:])[0, 1] for lag in range(1, ws_size + 1)]
    ax.bar(range(1, ws_size + 1), autocorr, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'Autocorrelation (lags 1-{ws_size})')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3)

    print(f"   Window={ws_size}: Corr at lag 1 = {autocorr[0]:.4f}, "
          f"lag {ws_size} = {autocorr[-1]:.4f}")

plt.tight_layout()
plt.savefig('eda_05_window_analysis.png', bbox_inches='tight')
plt.close()
print("   Saved: eda_05_window_analysis.png")

# ===================== 8. NORMALITY TESTS =====================
print("\n5. NORMALITY TESTS")
print("-" * 40)

# Shapiro-Wilk (on sample)
sample = ws[np.random.RandomState(42).choice(n, min(500, n), replace=False)]
sw_stat, sw_p = stats.shapiro(sample)
print(f"   Shapiro-Wilk: stat={sw_stat:.4f}, p={sw_p:.6f} → {'Normal' if sw_p > 0.05 else 'Non-normal'}")

# D'Agostino
dag_stat, dag_p = stats.normaltest(ws)
print(f"   D'Agostino: stat={dag_stat:.4f}, p={dag_p:.6f} → {'Normal' if dag_p > 0.05 else 'Non-normal'}")

# Anderson-Darling
ad_result = stats.anderson(ws, dist='norm')
print(f"   Anderson-Darling: stat={ad_result.statistic:.4f}")
for i in range(len(ad_result.critical_values)):
    sl = ad_result.significance_level[i]
    cv = ad_result.critical_values[i]
    print(f"     {sl}%: critical={cv:.4f} → {'Reject' if ad_result.statistic > cv else 'Accept'}")

# ===================== 9. PEAK ANALYSIS =====================
print("\n6. PEAK/TROUGH ANALYSIS")
print("-" * 40)

threshold_high = np.percentile(ws, 95)
threshold_low = np.percentile(ws, 5)
peaks = ws[ws > threshold_high]
troughs = ws[ws < threshold_low]

print(f"   95th percentile (high): {threshold_high:.4f} m/s ({len(peaks)} occurrences)")
print(f"   5th percentile (low): {threshold_low:.4f} m/s ({len(troughs)} occurrences)")
print(f"   Peak-to-trough ratio: {threshold_high/threshold_low:.2f}x")

# ===================== 10. SUMMARY TABLE =====================
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Metric':<30} {'Value':<20}")
print("-" * 50)
print(f"{'Observations':<30} {n}")
print(f"{'Mean (m/s)':<30} {ws.mean():.4f}")
print(f"{'Std Dev (m/s)':<30} {ws.std():.4f}")
print(f"{'Min (m/s)':<30} {ws.min():.4f}")
print(f"{'Max (m/s)':<30} {ws.max():.4f}")
print(f"{'Skewness':<30} {stats.skew(ws):.4f}")
print(f"{'Kurtosis':<30} {stats.kurtosis(ws):.4f}")
print(f"{'ADF p-value (original)':<30} {adf_orig[1]:.6f}")
print(f"{'ADF p-value (differenced)':<30} {adf_diff[1]:.6f}")
print(f"{'Stationary':<30} {'Yes (after differencing)' if adf_diff[1] < 0.05 else 'No'}")
print(f"{'Normal distribution':<30} {'No' if dag_p < 0.05 else 'Yes'}")
print(f"{'Outliers':<30} {outliers} ({100*outliers/n:.2f}%)")
print(f"{'Best lag for correlation':<30} 1")
print("=" * 60)

print("\nAll EDA plots saved to eda_01-05_*.png")
