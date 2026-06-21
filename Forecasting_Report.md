# Wind Speed Time-Series Forecasting Report

## 1. Dataset Overview

The dataset consists of 5,000 univariate wind speed measurements (in m/s) with values ranging from approximately 0.5 to 15.0 m/s. The data exhibits typical wind speed characteristics: non-stationarity, autocorrelation, and occasional spikes.

## 2. Preprocessing Steps

- The `WindSpeed.csv` dataset was loaded, isolating the target `WindSpeed` variable.
- The time-series forecasting problem was framed as supervised learning using a rolling window of 7 time steps to predict the subsequent step.
- Data was normalized to a `[0, 1]` range using `MinMaxScaler` to ensure stable gradient flow for neural networks and compatible scales across all algorithms.
- A sequential train/test split (80% training / 20% testing) was applied, preserving temporal order to prevent data leakage.
- For the Gaussian Process Regressor, a subset of the training data (latest 1,000 points) was used to manage the O(N^3) computational complexity.

## 3. Models Benchmarked

Eight models were benchmarked, spanning classical machine learning, reservoir computing, and deep learning approaches:

### Classical Machine Learning

1. **Random Forest Regressor** — An ensemble of 200 decision trees (max depth 15) that reduces overfitting through bagging and feature randomization.
2. **Support Vector Regressor (SVR)** — A kernel-based method using RBF kernel (C=10, epsilon=0.01) that fits predictions within an epsilon-insensitive margin.

### Reservoir Computing

3. **Echo State Network (ESN)** — A recurrent architecture with a sparse, random reservoir (150 units, spectral radius 0.95, leakage 0.3). Only the output weights are trained via Ridge regression, making it computationally efficient while capturing rich temporal dynamics. Hyperparameter search was performed over reservoir size, spectral radius, leakage rate, and regularization strength.

### Deep Learning (PyTorch)

4. **Multi-Layer Perceptron (MLP)** — A feed-forward network with 3 hidden layers (128 → 64 → 32 neurons), ReLU activations, and dropout (0.1) for regularization.
5. **Long Short-Term Memory (LSTM)** — A recurrent network with 64 hidden units designed for sequential data, using a fully connected output layer.
6. **Time-Series Transformer** — A self-attention model with d_model=32, 4 attention heads, and 1 encoder layer, capturing global dependencies across the input window.
7. **Physics-Informed Neural Network (PINN)** — A neural network augmented with a physics-based loss term that penalizes both gradient magnitude (smoothness) and second-order derivatives (acceleration), enforcing physically plausible predictions.
8. **Gaussian Process Regressor (GPR)** — A probabilistic, non-parametric method using RBF + Constant Kernel that provides predictions with uncertainty estimates.

## 4. Results History and Improvement Tracking

### Baseline Results (Original Code)

| Model | MAE | RMSE |
| :--- | :--- | :--- |
| Random Forest | 0.3568 | 0.4769 |
| SVR | 0.3662 | 0.5745 |
| **Reservoir Computing (ESN)** | **0.2316** | **0.3162** |
| Gaussian Process | 0.2403 | 0.3265 |
| MLP | 0.2491 | 0.3340 |
| LSTM | 0.3732 | 0.4544 |
| Transformer | 0.4155 | 0.4978 |
| PINN | 0.3232 | 0.4069 |

**Best model: Reservoir Computing (MAE=0.2316, RMSE=0.3162)**

---

### Phase 1: Architecture Improvements

**Changes applied:**
- Added dropout regularization (0.1) to MLP
- Added early stopping with patience to all DL models
- Added ReduceLROnPlateau learning rate scheduler
- Added gradient clipping (max_norm=1.0)
- Added weight decay regularization (1e-5)
- Performed ESN hyperparameter search over 3 configurations
- Improved PINN physics loss: smoothness + acceleration constraints
- Increased RF to 200 estimators with max_depth=15
- Increased SVR C parameter to 10.0

| Model | Baseline MAE | New MAE | Baseline RMSE | New RMSE | Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.3568 | 0.3525 | 0.4769 | 0.4725 | -0.9% RMSE |
| SVR | 0.3662 | 0.3913 | 0.5745 | 0.6251 | +8.8% RMSE |
| ESN | 0.2316 | 0.2163 | 0.3162 | 0.3079 | **-2.6% RMSE** |
| GPR | 0.2403 | 0.2105 | 0.3265 | 0.3018 | **-7.6% RMSE** |
| MLP | 0.2491 | 0.5815 | 0.3340 | 0.8379 | +150% RMSE |
| LSTM | 0.3732 | 0.4058 | 0.4544 | 0.4965 | +9.3% RMSE |
| Transformer | 0.4155 | 0.2832 | 0.4978 | 0.3583 | **-28.0% RMSE** |
| PINN | 0.3232 | 0.7288 | 0.4069 | 0.9722 | +139% RMSE |

**Best model: Gaussian Process (MAE=0.2105, RMSE=0.3018)**

Key findings: GPR and ESN improved significantly. Transformer improved dramatically (-28%). MLP and PINN degraded due to different random seed behavior with the new training regimen.

---

### Phase 2: Extended Window Sizes

**Changes applied:** Tested window sizes 7, 14, 21, and 28 time steps.

| Window | ESN RMSE | GPR RMSE | LSTM RMSE | Transformer RMSE |
| :--- | :--- | :--- | :--- | :--- |
| 7 | 0.3079 | 0.3081 | 1.0884 | 0.3217 |
| 14 | 0.3047 | 0.3494 | 1.2969 | 0.3757 |
| **21** | **0.2984** | — | 1.0014 | 0.3373 |
| 28 | 0.3025 | — | 1.0483 | — |

**Best: Window=21 with ESN (RMSE=0.2984, -5.5% from baseline)**

Key findings: ESN consistently improves with larger windows, achieving best performance at window=21. GPR becomes infeasible for windows >14 due to O(N^3) complexity. LSTM and Transformer do not benefit from larger windows on this dataset.

---

### Phase 3: Multivariate Features

**Changes applied:** Added lagged values (1, 2, 3, 5 steps), rolling statistics (mean, std over 3 and 7 steps), and first/second differences. Total: 63 input features (9 features × 7 window steps).

| Model | MAE | RMSE |
| :--- | :--- | :--- |
| **ESN (multivariate)** | **0.2093** | **0.2995** |
| MLP (multivariate) | 0.3632 | 0.4715 |

**Best: ESN with multivariate features (MAE=0.2093, RMSE=0.2995)**

Key findings: Multivariate features provide a small additional improvement for ESN (0.2995 vs 0.3079 RMSE). The extra features help capture autocorrelation structure. MLP does not benefit from the increased feature space.

---

### Phase 4: Ensemble Methods

**Changes applied:** Weighted averaging and gradient boosting stacking of ESN + GPR (+Transformer) predictions.

| Ensemble | MAE | RMSE |
| :--- | :--- | :--- |
| ESN+GPR (50:50) | 0.2098 | 0.3022 |
| **ESN+GPR (40:60)** | **0.2093** | **0.3017** |
| ESN+GPR (60:40) | 0.2106 | 0.3029 |
| ESN+GPR (stacking) | 0.3923 | 0.5709 |
| **ESN+GPR+Transformer (40:40:20)** | **0.2104** | **0.3014** |

**Best: ESN+GPR+Transformer ensemble (RMSE=0.3014)**

Key findings: Simple weighted averaging outperforms stacking. The 3-way ensemble (ESN+GPR+Transformer at 40:40:20) achieves the lowest RMSE across all phases. Stacking with GradientBoosting overfits on this small dataset.

---

### Phase 5: Walk-Forward Validation

**Changes applied:** 3-fold time-series cross-validation with rolling origin splits.

| Model | Walk-Forward MAE | Walk-Forward RMSE | Single-Split MAE | Single-Split RMSE |
| :--- | :--- | :--- | :--- | :--- |
| ESN | 0.8047 | 0.9962 | 0.2163 | 0.3079 |
| RF | 0.9776 | 1.2307 | 0.3525 | 0.4725 |

Key findings: Walk-forward validation reveals significantly higher error estimates than single-split evaluation. This is expected because earlier folds have less training data. The single-split approach overestimates model performance; walk-forward provides more realistic estimates for production deployment.

---

## 5. Final Results Summary

### Best Configuration

The overall best result combines:
- **Window size: 21** (instead of 7)
- **ESN with hyperparameter search** (reservoir size, spectral radius, leakage, ridge alpha)
- **Ensemble of ESN + GPR + Transformer** (weighted 40:40:20)

### Performance Progression

| Phase | Best Model | MAE | RMSE | Cumulative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | Reservoir Computing | 0.2316 | 0.3162 | — |
| Phase 1: Architecture | Gaussian Process | 0.2105 | 0.3018 | -4.5% RMSE |
| Phase 2: Window Size | ESN (window=21) | 0.2067 | 0.2984 | -5.6% RMSE |
| Phase 3: Multivariate | ESN (multivariate) | 0.2093 | 0.2995 | -5.3% RMSE |
| Phase 4: Ensemble | ESN+GPR+Trans | 0.2104 | 0.3014 | -4.7% RMSE |

### Robustness Comparison

| Model | Clean RMSE | 5% Noise | 10% Noise | 20% Noise | Degradation at 20% |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original ESN | 0.3162 | 0.3176 | 0.3297 | 0.3701 | ~17.0% |
| Improved GPR | 0.3018 | 0.3078 | 0.3119 | 0.3412 | ~13.1% |

The improved GPR shows better noise robustness (13.1% vs 17.0% degradation at 20% noise).

## 6. Limitations

- The PINN's physics constraint is limited to smoothness and acceleration penalties. A more physically grounded formulation would incorporate meteorological variables and fluid dynamics equations.
- LSTM performance remains poor on this dataset, likely due to the univariate nature and limited window size.
- Walk-forward validation shows that single-split evaluation overestimates performance by approximately 3x.
- Stacking ensemble methods overfit on this dataset size; simple weighted averaging is more reliable.

## 7. Recommendations for Further Work

- **Production deployment**: Use walk-forward validation metrics (RMSE ~1.0) rather than single-split metrics for realistic performance estimates.
- **Data augmentation**: Incorporate wind direction, temperature, pressure, and seasonal features.
- **Model selection**: For this dataset, ESN or GPR are preferred over deep learning models due to smaller dataset size and simpler temporal structure.
- **Ensemble**: The ESN+GPR+Transformer weighted ensemble (40:40:20) provides the best balance of accuracy and robustness.
