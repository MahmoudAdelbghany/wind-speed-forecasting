# Wind Speed Time-Series Forecasting

A comprehensive benchmark of 8 machine learning and deep learning models for wind speed prediction, with systematic improvement testing and full results history.

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

## Project Structure

```
nileuni/
├── forecasting.py          # Main model training and evaluation
├── test_improvements.py    # Comprehensive improvement testing script
├── Forecasting_Report.md   # Full report with history and analysis
├── improvement_history.json # Machine-readable results for all phases
├── results_table.txt       # Model evaluation and robustness results
├── actual_vs_predicted.png # Best model prediction plot
├── WindSpeed.csv           # Dataset (5000 wind speed measurements)
└── Research Assessment.pdf # Original assessment document
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
pip install pandas numpy scikit-learn torch matplotlib

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
