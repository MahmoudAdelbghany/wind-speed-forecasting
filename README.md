# Wind Speed Time-Series Forecasting

A comprehensive benchmark of 8 machine learning and deep learning models for wind speed prediction, with systematic improvement testing, full EDA, and results history.

## Results Summary

| Model | MAE | RMSE |
| :--- | :--- | :--- |
| Random Forest | 0.3525 | 0.4725 |
| SVR | 0.3913 | 0.6251 |
| Reservoir Computing (ESN) | 0.2163 | 0.3079 |
| Gaussian Process | 0.2105 | 0.3018 |
| MLP | 0.5815 | 0.8379 |
| LSTM | 0.4058 | 0.4965 |
| Transformer | 0.2832 | 0.3583 |
| PINN | 0.7288 | 0.9722 |
| **Ensemble (ESN+GPR+Trans)** | **0.2104** | **0.3014** |

## Improvement History

```
Baseline (window=7):     ESN RMSE = 0.3162
  ↓ Architecture tuning
Phase 1 (tuned models):  GPR RMSE = 0.3018  (-4.5%)
  ↓ Extended window sizes
Phase 2 (window=21):     ESN RMSE = 0.2984  (-5.6%)
  ↓ Multivariate features
Phase 3 (lagged+rolling): ESN RMSE = 0.2995 (-5.3%)
  ↓ Ensemble methods
Phase 4 (ESN+GPR+Trans): RMSE = 0.3014     (-4.7%)
```

## Exploratory Data Analysis

### Dataset Characteristics

| Metric | Value |
|---|---|
| Observations | 5,000 |
| Mean | 8.3936 m/s |
| Std Dev | 3.7342 m/s |
| Min / Max | 0.00 / 18.70 m/s |
| Skewness | -0.5353 (left-skewed) |
| Kurtosis | -0.8403 (platykurtic) |
| Missing values | 0 |
| Outliers (IQR) | 0 (0.00%) |

### Key EDA Findings

1. **Non-stationary**: ADF test p=0.27 (original), p≈0 (differenced). Series requires differencing for stationarity.

2. **Non-normal distribution**: Shapiro-Wilk p≈0, D'Agostino p≈0. Distribution is left-skewed with lighter tails than Gaussian (Q-Q plot shows deviation at both extremes).

3. **Strong autocorrelation**: ACF remains >0.8 even at lag 100. PACF shows significant spikes at lags 1-5, indicating strong short-term dependencies. This explains why ESN (which captures temporal dynamics) outperforms feed-forward models.

4. **Dominant periods (FFT)**: Periods at 2500, 5000, 1667, 714, and 1250 steps. The cumulative periodogram shows most power at low frequencies (long trends).

5. **Window size correlation**: Even at lag 28, autocorrelation remains >0.91. This justifies using window sizes up to 21-28 for the ESN model.

6. **Lag-1 plot**: Tight linear relationship (r=0.97) confirms high serial correlation — next value is highly predictable from current value.

### EDA Visualizations

| Plot | Description |
|---|---|
| `eda_01_distribution.png` | Histogram, Q-Q plot, box plot, ECDF with normal/log-normal fits |
| `eda_02_timeseries.png` | Full series, rolling statistics (mean±2σ), first differences |
| `eda_03_autocorrelation.png` | ACF, PACF, differenced ACF, lag-1 scatter plot |
| `eda_04_spectral.png` | Periodogram (FFT) and cumulative periodogram |
| `eda_05_window_analysis.png` | Autocorrelation decay for window sizes 7, 14, 21, 28 |

## Project Structure

```
nileuni/
├── forecasting.py              # Main model training and evaluation
├── test_improvements.py        # Comprehensive improvement testing script
├── eda.py                      # Exploratory data analysis script
├── Forecasting_Report.md       # Full report with history and analysis
├── improvement_history.json    # Machine-readable results for all phases
├── results_table.txt           # Model evaluation and robustness results
├── actual_vs_predicted.png     # Best model prediction plot
├── eda_01_distribution.png     # Distribution analysis plots
├── eda_02_timeseries.png       # Time series overview plots
├── eda_03_autocorrelation.png  # Autocorrelation analysis plots
├── eda_04_spectral.png         # Spectral analysis plots
├── eda_05_window_analysis.png  # Window size analysis plots
├── WindSpeed.csv               # Dataset (5000 wind speed measurements)
└── Research Assessment.pdf     # Original assessment document
```

## Models Implemented

### Classical ML
- **Random Forest** — 200 trees, max_depth=15
- **SVR** — RBF kernel, C=10, epsilon=0.01

### Reservoir Computing
- **Echo State Network (ESN)** — 150 reservoir units, sparse connectivity, hyperparameter search

### Deep Learning (PyTorch)
- **MLP** — 128→64→32, ReLU, dropout=0.1
- **LSTM** — 64 hidden units, early stopping
- **Transformer** — d_model=32, 4 heads, 1 layer
- **PINN** — Smoothness + acceleration physics loss

### Ensemble
- **ESN+GPR+Transformer** — Weighted average (40:40:20)

## Key Findings

1. **ESN outperforms LSTM** on this dataset — reservoir computing is well-suited for univariate wind speed forecasting with limited data
2. **Window size 21** is optimal for ESN — longer context improves predictions
3. **Simple weighted averaging** beats complex stacking ensembles on small datasets
4. **Walk-forward validation** reveals single-split overestimates performance by ~3x (real-world RMSE ~1.0 vs 0.30)

## Usage

```bash
# Install dependencies
pip install pandas numpy scikit-learn torch matplotlib statsmodels scipy

# Run EDA
python eda.py

# Run main forecasting pipeline
python forecasting.py

# Run comprehensive improvement testing
python test_improvements.py
```

## Robustness

Best model (GPR) noise robustness:
| Noise | RMSE | Degradation |
|---|---|---|
| 0% | 0.3018 | — |
| 5% | 0.3078 | +2.0% |
| 10% | 0.3119 | +3.4% |
| 20% | 0.3412 | +13.1% |

## License

MIT
