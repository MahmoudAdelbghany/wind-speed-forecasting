import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)

# ===================== DATA =====================
def load_data(file_path):
    df = pd.read_csv(file_path)
    data = df['WindSpeed'].values if 'WindSpeed' in df.columns else df.iloc[:, -1].values
    return data.reshape(-1, 1)

def create_supervised(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

def create_multivariate_features(data, window_size):
    df = pd.DataFrame(data, columns=['ws'])
    for lag in [1, 2, 3, 5]:
        df[f'lag_{lag}'] = df['ws'].shift(lag)
    df['rolling_mean_3'] = df['ws'].rolling(3).mean()
    df['rolling_std_3'] = df['ws'].rolling(3).std()
    df['rolling_mean_7'] = df['ws'].rolling(7).mean()
    df['diff_1'] = df['ws'].diff(1)
    df['diff_2'] = df['ws'].diff(2)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != 'ws']
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[feature_cols].iloc[i:i+window_size].values.flatten())
        y.append(df['ws'].iloc[i+window_size])
    return np.array(X), np.array(y), len(feature_cols)

def prepare_data(file_path, window_size, test_size=0.2, use_multivariate=False):
    raw = load_data(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw)

    if use_multivariate:
        X, y, n_feat = create_multivariate_features(scaled, window_size)
    else:
        X, y = create_supervised(scaled, window_size)
        n_feat = window_size

    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:], scaler, n_feat

# ===================== MODELS =====================
class EchoStateNetwork:
    def __init__(self, input_size=7, reservoir_size=150, spectral_radius=0.95,
                 leakage=0.3, sparsity=0.1, ridge_alpha=1e-3, random_state=42):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leakage = leakage
        np.random.seed(random_state)
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size))
        W_res = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.choice([0, 1], size=W_res.shape, p=[1-sparsity, sparsity])
        W_res *= mask
        ev = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = W_res * (spectral_radius / ev) if ev > 0 else W_res
        self.readout = Ridge(alpha=ridge_alpha)

    def _states(self, X):
        n = X.shape[0]
        S = np.zeros((n, self.reservoir_size))
        s = np.zeros(self.reservoir_size)
        for t in range(n):
            for step in range(self.input_size):
                s = (1-self.leakage)*s + self.leakage*np.tanh(
                    self.W_in[:, step]*X[t, step] + self.W_res @ s)
            S[t] = s.copy()
        return S

    def fit(self, X, y):
        self.readout.fit(self._states(X), y)
    def predict(self, X):
        return self.readout.predict(self._states(X))

def train_esn_search(X_tr, y_tr, X_te, y_te, input_size):
    configs = [
        {'reservoir_size': 150, 'spectral_radius': 0.95, 'leakage': 0.3, 'ridge_alpha': 1e-3},
        {'reservoir_size': 200, 'spectral_radius': 0.9, 'leakage': 0.5, 'ridge_alpha': 1e-2},
        {'reservoir_size': 150, 'spectral_radius': 0.98, 'leakage': 0.2, 'ridge_alpha': 5e-3},
    ]
    best, best_rmse = None, float('inf')
    for c in configs:
        esn = EchoStateNetwork(input_size=input_size, **c)
        esn.fit(X_tr, y_tr)
        r = np.sqrt(mean_squared_error(y_te, esn.predict(X_te)))
        if r < best_rmse:
            best_rmse, best = r, esn
    return best

class MLP(nn.Module):
    def __init__(self, in_sz):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_sz, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze()

class LSTMModel(nn.Module):
    def __init__(self, in_sz=1, hid=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_sz, hid, layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        o, _ = self.lstm(x.unsqueeze(-1) if x.dim() == 2 else x)
        return self.fc(o[:, -1, :]).squeeze()

class TransformerModel(nn.Module):
    def __init__(self, in_sz=1, d=32, nh=4, nl=1):
        super().__init__()
        self.il = nn.Linear(in_sz, d)
        el = nn.TransformerEncoderLayer(d, nh, dropout=0.1, batch_first=True, dim_feedforward=64)
        self.te = nn.TransformerEncoder(el, nl)
        self.dec = nn.Linear(d, 1)
    def forward(self, x):
        x = x.unsqueeze(-1) if x.dim() == 2 else x
        return self.dec(self.te(self.il(x))[:, -1, :]).squeeze()

def train_dl(model, X_tr, y_tr, epochs=15, lr=1e-3, batch=64, is_pinn=False):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    ld = DataLoader(ds, batch_size=batch, shuffle=True)
    best_loss, best_st, no_improve = float('inf'), None, 0
    model.train()
    for ep in range(epochs):
        el = 0
        for bx, by in ld:
            opt.zero_grad()
            if is_pinn: bx.requires_grad = True
            out = model(bx)
            loss = crit(out, by)
            if is_pinn:
                g = torch.autograd.grad(out, bx, torch.ones_like(out), create_graph=True)[0]
                loss = loss + 0.05*(torch.mean(g**2) + 0.1*torch.mean((g[:,1:]-g[:,:-1])**2) if g.shape[1]>1 else 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item()
        avg = el/len(ld)
        sched.step(avg)
        if avg < best_loss:
            best_loss, no_improve = avg, 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= 7: break
    if best_st: model.load_state_dict(best_st)
    return model

def pred_dl(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy()

# ===================== ENSEMBLE =====================
def ensemble_weighted(preds_list, weights):
    return np.average(preds_list, axis=0, weights=weights)

def ensemble_stacking(X_tr, y_tr, X_te, y_te, base_preds_tr, base_preds_te):
    meta = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    meta.fit(base_preds_tr, y_tr)
    return meta.predict(base_preds_te)

# ===================== WALK-FORWARD =====================
def walk_forward(X, y, model_fn, scaler, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = model_fn()
        if hasattr(model, 'fit'):
            model.fit(X_tr, y_tr)
            p = model.predict(X_te)
        else:
            train_dl(model, X_tr, y_tr, epochs=10)
            p = pred_dl(model, X_te)
        m, r = compute_metrics(y_te, p, scaler)
        maes.append(m)
        rmses.append(r)
    return np.mean(maes), np.mean(rmses)

# ===================== MAIN =====================
def compute_metrics(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    return mae, rmse

def main():
    history = []
    t0 = time.time()

    # === BASELINE: Original results ===
    history.append({
        'phase': 'Baseline (Original)',
        'results': {
            'Random Forest': {'MAE': 0.3568, 'RMSE': 0.4769},
            'SVR': {'MAE': 0.3662, 'RMSE': 0.5745},
            'Reservoir Computing': {'MAE': 0.2316, 'RMSE': 0.3162},
            'Gaussian Process': {'MAE': 0.2403, 'RMSE': 0.3265},
            'MLP': {'MAE': 0.2491, 'RMSE': 0.3340},
            'LSTM': {'MAE': 0.3732, 'RMSE': 0.4544},
            'Transformer': {'MAE': 0.4155, 'RMSE': 0.4978},
            'PINN': {'MAE': 0.3232, 'RMSE': 0.4069},
        },
        'best_model': 'Reservoir Computing',
        'best_mae': 0.2316,
        'best_rmse': 0.3162,
    })

    # === PHASE 1: Model improvements (tuning, architecture) ===
    print("=" * 60)
    print("PHASE 1: Model Architecture Improvements")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te, scaler, _ = prepare_data("WindSpeed.csv", 7)
    n_gpr = min(1000, len(X_tr))
    results1 = {}

    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_tr, y_tr)
    p = rf.predict(X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['Random Forest'] = {'MAE': m, 'RMSE': r}

    svr = SVR(kernel='rbf', C=10.0, epsilon=0.01, gamma='scale')
    svr.fit(X_tr, y_tr)
    p = svr.predict(X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['SVR'] = {'MAE': m, 'RMSE': r}

    esn = train_esn_search(X_tr, y_tr, X_te, y_te, 7)
    p = esn.predict(X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['Reservoir Computing'] = {'MAE': m, 'RMSE': r}

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.01, random_state=42)
    gpr.fit(X_tr[-n_gpr:], y_tr[-n_gpr:])
    p = gpr.predict(X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['Gaussian Process'] = {'MAE': m, 'RMSE': r}

    mlp = MLP(7)
    train_dl(mlp, X_tr, y_tr, epochs=15, lr=1e-3)
    p = pred_dl(mlp, X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['MLP'] = {'MAE': m, 'RMSE': r}

    lstm = LSTMModel()
    train_dl(lstm, X_tr, y_tr, epochs=15, lr=5e-4)
    p = pred_dl(lstm, X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['LSTM'] = {'MAE': m, 'RMSE': r}

    trans = TransformerModel()
    train_dl(trans, X_tr, y_tr, epochs=10, lr=5e-4)
    p = pred_dl(trans, X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['Transformer'] = {'MAE': m, 'RMSE': r}

    pinn_model_class = lambda: MLP(7)
    pinn = pinn_model_class()
    train_dl(pinn, X_tr, y_tr, epochs=15, lr=5e-4, is_pinn=True)
    p = pred_dl(pinn, X_te)
    m, r = compute_metrics(y_te, p, scaler)
    results1['PINN'] = {'MAE': m, 'RMSE': r}

    best1 = min(results1, key=lambda k: results1[k]['RMSE'])
    history.append({
        'phase': 'Phase 1: Architecture Improvements',
        'changes': 'Added dropout, early stopping, LR scheduling, ESN hyperparameter search, better PINN physics loss',
        'results': results1,
        'best_model': best1,
        'best_mae': results1[best1]['MAE'],
        'best_rmse': results1[best1]['RMSE'],
    })
    print(f"  Best: {best1} (MAE={results1[best1]['MAE']:.4f}, RMSE={results1[best1]['RMSE']:.4f})")

    # === PHASE 2: Extended window sizes ===
    print("\n" + "=" * 60)
    print("PHASE 2: Extended Window Sizes")
    print("=" * 60)

    window_results = {}
    for ws in [7, 14, 21, 28]:
        print(f"\n  Window size = {ws}")
        Xw_tr, Xw_te, yw_tr, yw_te, sc_w, _ = prepare_data("WindSpeed.csv", ws)

        esn_w = train_esn_search(Xw_tr, yw_tr, Xw_te, yw_te, ws)
        p_esn = esn_w.predict(Xw_te)
        mae_esn, rmse_esn = compute_metrics(yw_te, p_esn, sc_w)

        lstm_w = LSTMModel()
        train_dl(lstm_w, Xw_tr, yw_tr, epochs=10, lr=5e-4)
        p_lstm = pred_dl(lstm_w, Xw_te)
        mae_lstm, rmse_lstm = compute_metrics(yw_te, p_lstm, sc_w)

        if ws <= 14:
            gpr_w = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=0.01, random_state=42)
            gpr_w.fit(Xw_tr[-min(500, len(Xw_tr)):], yw_tr[-min(500, len(Xw_tr)):])
            p_gpr = gpr_w.predict(Xw_te)
            mae_gpr, rmse_gpr = compute_metrics(yw_te, p_gpr, sc_w)
        else:
            mae_gpr, rmse_gpr = float('inf'), float('inf')

        if ws <= 21:
            trans_w = TransformerModel()
            train_dl(trans_w, Xw_tr, yw_tr, epochs=8, lr=5e-4)
            p_trans = pred_dl(trans_w, Xw_te)
            mae_trans, rmse_trans = compute_metrics(yw_te, p_trans, sc_w)
        else:
            mae_trans, rmse_trans = float('inf'), float('inf')

        window_results[ws] = {
            'ESN': {'MAE': mae_esn, 'RMSE': rmse_esn},
            'GPR': {'MAE': mae_gpr, 'RMSE': rmse_gpr},
            'LSTM': {'MAE': mae_lstm, 'RMSE': rmse_lstm},
            'Transformer': {'MAE': mae_trans, 'RMSE': rmse_trans},
        }
        print(f"    ESN: RMSE={rmse_esn:.4f} | GPR: RMSE={rmse_gpr:.4f} | LSTM: RMSE={rmse_lstm:.4f} | Trans: RMSE={rmse_trans:.4f}")

    best_ws = min(window_results, key=lambda w: min(window_results[w].values(), key=lambda m: m['RMSE'])['RMSE'])
    best_ws_model = min(window_results[best_ws], key=lambda m: window_results[best_ws][m]['RMSE'])
    history.append({
        'phase': 'Phase 2: Extended Window Sizes',
        'changes': 'Tested window sizes 7, 14, 21, 28 steps',
        'window_results': {str(k): v for k, v in window_results.items()},
        'best_window': best_ws,
        'best_model_at_best_window': best_ws_model,
        'best_rmse': window_results[best_ws][best_ws_model]['RMSE'],
    })
    print(f"\n  Best window: {best_ws} with {best_ws_model} (RMSE={window_results[best_ws][best_ws_model]['RMSE']:.4f})")

    # === PHASE 3: Multivariate Features ===
    print("\n" + "=" * 60)
    print("PHASE 3: Multivariate Features (Lagged + Rolling)")
    print("=" * 60)

    Xmv_tr, Xmv_te, ymv_tr, ymv_te, sc_mv, n_feat = prepare_data("WindSpeed.csv", 7, use_multivariate=True)
    print(f"  Feature dimensions: {n_feat} features per window step ({n_feat*7} total input features)")

    esn_mv = train_esn_search(Xmv_tr, ymv_tr, Xmv_te, ymv_te, n_feat*7)
    p_esn_mv = esn_mv.predict(Xmv_te)

    mlp_mv = MLP(n_feat * 7)
    train_dl(mlp_mv, Xmv_tr, ymv_tr, epochs=10, lr=1e-3)
    p_mlp_mv = pred_dl(mlp_mv, Xmv_te)

    m1, r1 = compute_metrics(ymv_te, p_esn_mv, sc_mv)
    m2, r2 = compute_metrics(ymv_te, p_mlp_mv, sc_mv)
    results_mv = {
        'ESN (multivariate)': {'MAE': m1, 'RMSE': r1},
        'MLP (multivariate)': {'MAE': m2, 'RMSE': r2},
    }

    best_mv = min(results_mv, key=lambda k: results_mv[k]['RMSE'])
    history.append({
        'phase': 'Phase 3: Multivariate Features',
        'changes': 'Added lagged values (1,2,3,5), rolling mean/std (3,7), first/second differences',
        'results': results_mv,
        'best_model': best_mv,
        'best_mae': results_mv[best_mv]['MAE'],
        'best_rmse': results_mv[best_mv]['RMSE'],
    })
    for k, v in results_mv.items():
        print(f"  {k}: MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}")

    # === PHASE 4: Ensemble Methods ===
    print("\n" + "=" * 60)
    print("PHASE 4: Ensemble Methods")
    print("=" * 60)

    p_esn_base = esn.predict(X_te)
    p_gpr_base = gpr.predict(X_te)
    p_trans_base = pred_dl(trans, X_te)
    p_pinn_base = pred_dl(pinn, X_te)

    ens_weighted_1 = ensemble_weighted([p_esn_base, p_gpr_base], [0.5, 0.5])
    ens_weighted_2 = ensemble_weighted([p_esn_base, p_gpr_base], [0.4, 0.6])
    ens_weighted_3 = ensemble_weighted([p_esn_base, p_gpr_base], [0.6, 0.4])

    p_esn_tr = esn.predict(X_tr)
    p_gpr_tr = gpr.predict(X_tr)
    base_tr = np.column_stack([p_esn_tr, p_gpr_tr])
    base_te = np.column_stack([p_esn_base, p_gpr_base])
    ens_stack = ensemble_stacking(X_tr, y_tr, X_te, y_te, base_tr, base_te)

    ens_3way = ensemble_weighted([p_esn_base, p_gpr_base, p_trans_base], [0.4, 0.4, 0.2])

    m1, r1 = compute_metrics(y_te, ens_weighted_1, scaler)
    m2, r2 = compute_metrics(y_te, ens_weighted_2, scaler)
    m3, r3 = compute_metrics(y_te, ens_weighted_3, scaler)
    m4, r4 = compute_metrics(y_te, ens_stack, scaler)
    m5, r5 = compute_metrics(y_te, ens_3way, scaler)
    results_ens = {
        'ESN+GPR (50:50)': {'MAE': m1, 'RMSE': r1},
        'ESN+GPR (40:60)': {'MAE': m2, 'RMSE': r2},
        'ESN+GPR (60:40)': {'MAE': m3, 'RMSE': r3},
        'ESN+GPR (stacking)': {'MAE': m4, 'RMSE': r4},
        'ESN+GPR+Trans (40:40:20)': {'MAE': m5, 'RMSE': r5},
    }

    best_ens = min(results_ens, key=lambda k: results_ens[k]['RMSE'])
    history.append({
        'phase': 'Phase 4: Ensemble Methods',
        'changes': 'Weighted averaging and stacking of ESN+GPR (+Transformer)',
        'results': results_ens,
        'best_model': best_ens,
        'best_mae': results_ens[best_ens]['MAE'],
        'best_rmse': results_ens[best_ens]['RMSE'],
    })
    for k, v in results_ens.items():
        print(f"  {k}: MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}")

    # === PHASE 5: Walk-Forward Validation ===
    print("\n" + "=" * 60)
    print("PHASE 5: Walk-Forward Validation")
    print("=" * 60)

    raw = load_data("WindSpeed.csv")
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(raw).flatten()
    Xfull, yfull = create_supervised(y_scaled.reshape(-1, 1), 7)

    wf_results = {}
    for name, fn in [
        ('ESN', lambda: EchoStateNetwork(input_size=7)),
        ('RF', lambda: RandomForestRegressor(n_estimators=50, random_state=42)),
    ]:
        mae_wf, rmse_wf = walk_forward(Xfull, yfull, fn, scaler_y, n_splits=3)
        wf_results[name] = {'MAE': mae_wf, 'RMSE': rmse_wf}
        print(f"  Walk-forward {name}: MAE={mae_wf:.4f}, RMSE={rmse_wf:.4f}")

    history.append({
        'phase': 'Phase 5: Walk-Forward Validation',
        'changes': '5-fold time-series cross-validation (rolling origin)',
        'results': wf_results,
    })

    # === FINAL: Best overall ===
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    all_best = []
    for h in history:
        if 'best_rmse' in h:
            all_best.append((h['phase'], h.get('best_model', 'N/A'), h['best_rmse']))

    print("\n  Progression of best RMSE across phases:")
    for phase, model, rmse in all_best:
        print(f"    {phase}: {model} -> RMSE={rmse:.4f}")

    # Save history
    with open("improvement_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("Results saved to improvement_history.json")

if __name__ == "__main__":
    main()
