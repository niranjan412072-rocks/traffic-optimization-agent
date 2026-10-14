"""Traffic volume forecasting (15 / 30 / 60 minutes ahead).

Models: HistGradientBoosting (default, scikit-learn) or XGBoost (optional).
Compared against two naive baselines: persistence and same-time-yesterday.

Run:  python -m src.forecasting --horizons 1 2 4
      python -m src.forecasting --model xgboost
      python -m src.forecasting --metro-csv Metro_Interstate_Traffic_Volume.csv
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


def load_data(path=config.TRAFFIC_CSV):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_metro_csv(path):
    """Adapter for Kaggle 'Metro Interstate Traffic Volume' (hourly data)."""
    raw = pd.read_csv(path, parse_dates=["date_time"])
    raw = raw.drop_duplicates("date_time").set_index("date_time").sort_index()
    df = pd.DataFrame({
        "volume": raw["traffic_volume"],
        "temp": raw["temp"] - 273.15,
        "rain": raw["rain_1h"],
        "is_holiday": (raw["holiday"].astype(str) != "None").astype(int),
    })
    df = df.resample("h").mean().ffill()
    df["is_holiday"] = (df.groupby(df.index.date)["is_holiday"].transform("max") > 0).astype(int)
    return df


def make_features(df, horizon=1, freq_min=config.FREQ_MIN):
    """Features available at time t for predicting volume at t + horizon steps."""
    sph = 60 // freq_min
    v = df["volume"]
    tgt = df.index + pd.Timedelta(minutes=freq_min * horizon)
    hr = tgt.hour.values + tgt.minute.values / 60
    X = pd.DataFrame(index=df.index)
    X["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    X["dow"] = tgt.dayofweek.values
    X["is_weekend"] = (tgt.dayofweek.values >= 5).astype(int)
    X["is_holiday"] = df["is_holiday"].shift(-horizon).fillna(0).values   # calendar is known ahead
    X["vol_now"] = v
    X["vol_prev_step"] = v.shift(1)
    X["vol_1h_mean"] = v.rolling(sph, min_periods=1).mean()
    X["vol_same_time_yesterday"] = v.shift(24 * sph - horizon)
    X["vol_same_time_last_week"] = v.shift(168 * sph - horizon)
    if "speed" in df:
        X["speed_now"] = df["speed"]
    X["temp"] = df["temp"]
    X["rain"] = df["rain"]
    y = v.shift(-horizon)
    return X, y


def build_model(name="hgb", seed=config.SEED):
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=seed)


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE_%": float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100),
            "R2": float(r2_score(y, p))}


def train_and_evaluate(df, horizon, model_name="hgb", freq_min=config.FREQ_MIN, save=True, plot=True):
    X, y = make_features(df, horizon, freq_min)
    data = X.join(y.rename("y")).dropna()
    split = int(len(data) * 0.8)                      # time-ordered split (no leakage)
    train, test = data.iloc[:split], data.iloc[split:]
    feats = list(X.columns)
    model = build_model(model_name)
    model.fit(train[feats], train["y"])
    pred = model.predict(test[feats])

    res = {"horizon_steps": horizon, "horizon_min": horizon * freq_min, "model_name": model_name,
           "n_train": len(train), "n_test": len(test),
           "model": _metrics(test["y"], pred),
           "baseline_persistence": _metrics(test["y"], test["vol_now"]),
           "baseline_same_time_yesterday": _metrics(test["y"], test["vol_same_time_yesterday"])}
    res["MAE_improvement_vs_persistence_%"] = 100 * (
        res["baseline_persistence"]["MAE"] - res["model"]["MAE"]) / res["baseline_persistence"]["MAE"]

    if save:
        joblib.dump({"model": model, "features": feats, "horizon": horizon, "freq_min": freq_min,
                     "p90": float(df["volume"].quantile(0.9))}, config.forecast_model_path(horizon))
    if plot:
        n = min(len(test), 3 * 24 * 60 // freq_min)
        plt.figure(figsize=(11, 3.6))
        plt.plot(test.index[:n], test["y"].values[:n], label="Actual", lw=1.5)
        plt.plot(test.index[:n], pred[:n], label="Predicted", lw=1.2)
        plt.title(f"Traffic volume forecast, {horizon * freq_min} min ahead "
                  f"(MAE {res['model']['MAE']:.1f} veh / {freq_min} min)")
        plt.ylabel("Vehicles per interval"); plt.legend(); plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / f"forecast_h{horizon}.png", dpi=130); plt.close()
    return res


def load_bundle(horizon):
    p = config.forecast_model_path(horizon)
    return joblib.load(p) if p.exists() else None


def predict_latest(bundle, history):
    """Forecast volume `horizon` steps after the last row of `history`."""
    X, _ = make_features(history, bundle["horizon"], bundle["freq_min"])
    return float(bundle["model"].predict(X.iloc[[-1]][bundle["features"]])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--model", default="hgb", choices=["hgb", "xgboost"])
    ap.add_argument("--metro-csv", default=None, help="optional real dataset (hourly)")
    a = ap.parse_args()
    if a.metro_csv:
        df, freq = load_metro_csv(a.metro_csv), 60
    else:
        df, freq = load_data(), config.FREQ_MIN
    out = {}
    for h in a.horizons:
        r = train_and_evaluate(df, h, a.model, freq)
        out[f"h{h}"] = r
        print(f"[{r['horizon_min']:>3} min] MAE={r['model']['MAE']:.1f}  RMSE={r['model']['RMSE']:.1f}  "
              f"R2={r['model']['R2']:.3f}  | persistence MAE={r['baseline_persistence']['MAE']:.1f}  "
              f"(improvement {r['MAE_improvement_vs_persistence_%']:.1f}%)")
    (config.RESULTS_DIR / "forecast_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
