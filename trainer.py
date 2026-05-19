import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from wno_model import train_wno_model, predict_wno

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def create_sequences(prices_series, seq_len):
    """Create sliding window sequences of log returns."""
    log_ret = np.log(prices_series / prices_series.shift(1)).dropna().values
    if len(log_ret) < seq_len + 1:
        return None, None
    X, y = [], []
    for i in range(seq_len, len(log_ret)-1):
        X.append(log_ret[i-seq_len:i])
        y.append(log_ret[i+1])
    return np.array(X), np.array(y)

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Wavelet Neural Operator) ===")
        prices = data_manager.prepare_price_matrix(df, tickers)
        if prices.empty or len(prices) < max(config.WINDOWS) + config.SEQ_LEN + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(prices) < win + config.SEQ_LEN + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            prices_win = prices.iloc[-win:]
            etf_pred = {}
            for etf in tickers:
                if etf not in prices_win.columns:
                    continue
                series = prices_win[etf].dropna()
                X, y = create_sequences(series, config.SEQ_LEN)
                if X is None or len(X) < 20:
                    continue
                split = int(0.8 * len(X))
                X_train = X[:split]
                y_train = y[:split]
                X_train = X_train[:, np.newaxis, :]   # (n, 1, seq_len)
                model = train_wno_model(X_train, y_train, config.SEQ_LEN,
                                        hidden_channels=config.HIDDEN_CHANNELS,
                                        lr=config.LEARNING_RATE,
                                        epochs=config.EPOCHS,
                                        batch_size=config.BATCH_SIZE,
                                        device=device)
                # Predict on the most recent window
                log_ret_series = np.log(series / series.shift(1)).dropna()
                if len(log_ret_series) < config.SEQ_LEN:
                    continue
                last_input = log_ret_series.iloc[-config.SEQ_LEN:].values.reshape(1, 1, -1)
                pred = predict_wno(model, last_input)
                # Ensure pred is scalar
                if isinstance(pred, np.ndarray):
                    pred = pred[0] if pred.size > 0 else 0.0
                elif isinstance(pred, float):
                    pass
                else:
                    pred = float(pred)
                etf_pred[etf] = pred
            window_results[win] = etf_pred
            for etf, pred in etf_pred.items():
                if etf not in best_per_etf or pred > best_per_etf[etf][0]:
                    best_per_etf[etf] = (pred, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            returns = data_manager.prepare_returns_matrix(df, tickers)
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by predicted return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/wavelet_no_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Wavelet Neural Operator Engine complete ===")

if __name__ == "__main__":
    main()
