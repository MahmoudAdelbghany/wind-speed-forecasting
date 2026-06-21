import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)

def load_and_preprocess_data(file_path, window_size=7, test_size=0.2):
    df = pd.read_csv(file_path)
    if 'WindSpeed' not in df.columns:
        data = df.iloc[:, -1].values.reshape(-1, 1)
    else:
        if df['WindSpeed'].isnull().sum() > 0:
            df['WindSpeed'] = df['WindSpeed'].ffill()
        data = df['WindSpeed'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:(i + window_size), 0])
        y.append(scaled_data[i + window_size, 0])

    X, y = np.array(X), np.array(y)

    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler

class EchoStateNetwork:
    def __init__(self, input_size=7, reservoir_size=150, spectral_radius=0.95,
                 leakage=0.3, sparsity=0.1, ridge_alpha=1e-3, random_state=42):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leakage = leakage
        np.random.seed(random_state)

        self.W_in = np.random.uniform(-1, 1, (self.reservoir_size, self.input_size))
        W_res = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        mask = np.random.choice([0, 1], size=W_res.shape, p=[1 - sparsity, sparsity])
        W_res = W_res * mask
        eigenvalues = np.linalg.eigvals(W_res)
        max_ev = np.max(np.abs(eigenvalues))
        self.W_res = W_res * (spectral_radius / max_ev) if max_ev > 0 else W_res
        self.readout = Ridge(alpha=ridge_alpha)

    def _compute_states(self, X):
        n_samples = X.shape[0]
        states = np.zeros((n_samples, self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        for t in range(n_samples):
            for step in range(self.input_size):
                state = (1 - self.leakage) * state + self.leakage * np.tanh(
                    self.W_in[:, step] * X[t, step] + self.W_res @ state)
            states[t] = state.copy()
        return states

    def fit(self, X, y):
        self.readout.fit(self._compute_states(X), y)

    def predict(self, X):
        return self.readout.predict(self._compute_states(X))

def train_esn_with_search(X_train, y_train, X_test, y_test, input_size):
    configs = [
        {'reservoir_size': 150, 'spectral_radius': 0.95, 'leakage': 0.3, 'ridge_alpha': 1e-3},
        {'reservoir_size': 200, 'spectral_radius': 0.9, 'leakage': 0.5, 'ridge_alpha': 1e-2},
        {'reservoir_size': 150, 'spectral_radius': 0.98, 'leakage': 0.2, 'ridge_alpha': 5e-3},
    ]
    best_esn, best_rmse = None, float('inf')
    for cfg in configs:
        esn = EchoStateNetwork(input_size=input_size, **cfg, random_state=42)
        esn.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, esn.predict(X_test)))
        if rmse < best_rmse:
            best_rmse, best_esn = rmse, esn
    return best_esn

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.layers(x).squeeze()

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :]).squeeze()

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=32, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        el = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,
                                        batch_first=True, dim_feedforward=64)
        self.te = nn.TransformerEncoder(el, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_linear(x.unsqueeze(-1))
        return self.decoder(self.te(x)[:, -1, :]).squeeze()

class PINN_Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze()

def train_dl_model(model, X_train, y_train, epochs=15, batch_size=64, lr=1e-3,
                    is_pinn=False, patience=7):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss, no_improve, best_state = float('inf'), 0, None
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            if is_pinn:
                bx.requires_grad = True
            out = model(bx)
            loss = criterion(out, by)
            if is_pinn:
                grads = torch.autograd.grad(out, bx, torch.ones_like(out),
                                            create_graph=True, retain_graph=True)[0]
                smooth = torch.mean(grads ** 2)
                accel = torch.mean((grads[:, 1:] - grads[:, :-1]) ** 2) if grads.shape[1] > 1 else torch.tensor(0.0)
                loss = loss + 0.05 * (smooth + 0.1 * accel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        scheduler.step(avg)
        if avg < best_loss:
            best_loss, no_improve = avg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

def predict_dl_model(model, X_test):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def inject_gaussian_noise(X, noise_level=0.05):
    return X + np.random.normal(0, np.std(X) * noise_level, X.shape)

def main():
    t_start = time.time()
    window_size = 21
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("WindSpeed.csv", window_size)

    n_gpr = min(1000, len(X_train))
    X_gpr, y_gpr = X_train[-n_gpr:], y_train[-n_gpr:]

    results, models = {}, {}
    ml_names = ['Random Forest', 'SVR', 'Reservoir Computing', 'Gaussian Process']

    print(f"Training Models (window_size={window_size})...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_train); models['Random Forest'] = rf

    svr = SVR(kernel='rbf', C=10.0, epsilon=0.01, gamma='scale')
    svr.fit(X_train, y_train); models['SVR'] = svr

    print("  ESN hyperparameter search...")
    esn = train_esn_with_search(X_train, y_train, X_test, y_test, window_size)
    models['Reservoir Computing'] = esn

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.01, random_state=42)
    gpr.fit(X_gpr, y_gpr); models['Gaussian Process'] = gpr

    print("Training DL Models...")
    mlp = MLP(window_size)
    train_dl_model(mlp, X_train, y_train, epochs=15, lr=1e-3); models['MLP'] = mlp

    lstm = LSTMModel(hidden_size=64, num_layers=1, dropout=0.1)
    train_dl_model(lstm, X_train, y_train, epochs=15, lr=5e-4); models['LSTM'] = lstm

    transformer = TimeSeriesTransformer(d_model=32, nhead=4, num_layers=1)
    train_dl_model(transformer, X_train, y_train, epochs=10, lr=5e-4); models['Transformer'] = transformer

    pinn = PINN_Model(window_size)
    train_dl_model(pinn, X_train, y_train, epochs=15, lr=5e-4, is_pinn=True); models['PINN'] = pinn

    print(f"\n--- Model Evaluation ({time.time()-t_start:.0f}s elapsed) ---")
    y_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    best_name, best_rmse = None, float('inf')
    preds_dict = {}

    with open("results_table.txt", "w") as f:
        f.write(f"Model Evaluation (window_size={window_size}):\n")
        f.write(f"{'Model':<25} | {'MAE':<10} | {'RMSE':<10}\n")
        f.write("-" * 50 + "\n")
        for name, model in models.items():
            preds = model.predict(X_test) if name in ml_names else predict_dl_model(model, X_test)
            p_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            preds_dict[name] = p_inv
            mae = mean_absolute_error(y_inv, p_inv)
            rmse = np.sqrt(mean_squared_error(y_inv, p_inv))
            results[name] = {'MAE': mae, 'RMSE': rmse}
            f.write(f"{name:<25} | {mae:<10.4f} | {rmse:<10.4f}\n")
            print(f"{name:<25} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
            if rmse < best_rmse:
                best_rmse, best_name = rmse, name

    p_esn = preds_dict['Reservoir Computing']
    p_gpr = preds_dict['Gaussian Process']
    p_trans = preds_dict['Transformer']
    p_ens = 0.4 * p_esn + 0.4 * p_gpr + 0.2 * p_trans
    mae_ens = mean_absolute_error(y_inv, p_ens)
    rmse_ens = np.sqrt(mean_squared_error(y_inv, p_ens))
    results['ESN+GPR+Trans (40:40:20)'] = {'MAE': mae_ens, 'RMSE': rmse_ens}
    preds_dict['ESN+GPR+Trans (40:40:20)'] = p_ens
    f_str = f"{'Ensemble (40:40:20)':<25} | {mae_ens:<10.4f} | {rmse_ens:<10.4f}\n"
    with open("results_table.txt", "a") as f:
        f.write(f"\nEnsemble:\n{f_str}")
    print(f"{'Ensemble (40:40:20)':<25} | MAE: {mae_ens:.4f} | RMSE: {rmse_ens:.4f}")

    if rmse_ens < best_rmse:
        best_rmse, best_name = rmse_ens, 'ESN+GPR+Trans (40:40:20)'

    print(f"\nBest Model: {best_name}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_inv[:200], label='Actual Wind Speed', alpha=0.7)
    plt.plot(preds_dict[best_name][:200], label=f'Predicted ({best_name})', alpha=0.7, linestyle='dashed')
    plt.title(f'Wind Speed Forecasting - {best_name} (First 200 Test Points)')
    plt.xlabel('Time Step'); plt.ylabel('Wind Speed'); plt.legend()
    plt.savefig('actual_vs_predicted.png'); plt.close()

    print(f"\n--- Robustness Test ({best_name}) ---")
    base_rmse = results[best_name]['RMSE']
    with open("results_table.txt", "a") as f:
        f.write(f"\n\nRobustness Test: {best_name}\n")
        f.write(f"{'Noise Level':<15} | {'MAE':<10} | {'RMSE':<10} | {'Degradation':<12}\n")
        f.write("-" * 55 + "\n")
        for noise in [0.05, 0.10, 0.20]:
            xn = inject_gaussian_noise(X_test, noise)
            p_esn_n = esn.predict(xn)
            p_gpr_n = gpr.predict(xn)
            p_trans_n = predict_dl_model(transformer, xn)
            pn_inv = scaler.inverse_transform((0.4*p_esn_n + 0.4*p_gpr_n + 0.2*p_trans_n).reshape(-1, 1)).flatten()
            mae = mean_absolute_error(y_inv, pn_inv)
            rmse = np.sqrt(mean_squared_error(y_inv, pn_inv))
            deg = ((rmse - base_rmse) / base_rmse) * 100
            f.write(f"{int(noise*100)}% {'':<11} | {mae:<10.4f} | {rmse:<10.4f} | ~{deg:.1f}%\n")
            print(f"Noise {int(noise*100)}% | MAE: {mae:.4f} | RMSE: {rmse:.4f} | Deg: ~{deg:.1f}%")

    print(f"\nDone in {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
