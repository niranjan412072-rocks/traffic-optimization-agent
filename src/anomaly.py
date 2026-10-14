"""Incident / anomaly detection with Isolation Forest (unsupervised).

Idea: an incident shows up as speed far below what is typical for that
day-of-week and time-of-day, usually together with a volume drop.

Run:  python -m src.anomaly
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import config

FEATURES = ["vol_ratio", "speed_ratio"]


def add_ratios(df):
    d = df.copy()
    key = [d.index.dayofweek, d.index.hour, d.index.minute]
    typ_v = d.groupby(key)["volume"].transform("median")
    typ_s = d.groupby(key)["speed"].transform("median")
    d["vol_ratio"] = d["volume"] / typ_v.clip(lower=1)
    d["speed_ratio"] = d["speed"] / typ_s.clip(lower=1)
    return d


def fit_detector(df, contamination=0.01, seed=config.SEED):
    d = add_ratios(df)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(d[FEATURES])
    return model


def detect(model, df):
    d = add_ratios(df)
    d["anomaly"] = (model.predict(d[FEATURES]) == -1).astype(int)
    d["anomaly_score"] = -model.score_samples(d[FEATURES])
    return d


def evaluate(d):
    tp = int(((d.anomaly == 1) & (d.incident == 1)).sum())
    fp = int(((d.anomaly == 1) & (d.incident == 0)).sum())
    fn = int(((d.anomaly == 0) & (d.incident == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # event level: how many distinct incidents had at least one flagged step
    ev = (d.incident.diff().fillna(d.incident) == 1).cumsum() * d.incident
    n_events = int(ev.max())
    caught = int(d.loc[d.anomaly == 1, "incident"].mul(ev[d.anomaly == 1]).replace(0, np.nan).nunique())
    return {"precision": prec, "recall": rec, "f1": f1, "incident_events": n_events,
            "incident_events_detected": caught,
            "event_detection_rate": caught / max(n_events, 1)}


if __name__ == "__main__":
    df = pd.read_csv(config.TRAFFIC_CSV, index_col=0, parse_dates=True)
    model = fit_detector(df)
    joblib.dump(model, config.ANOMALY_MODEL_PATH)
    d = detect(model, df)
    m = evaluate(d)
    (config.RESULTS_DIR / "anomaly_metrics.json").write_text(json.dumps(m, indent=2))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
    d[d.anomaly == 1][["volume", "speed", "vol_ratio", "speed_ratio", "anomaly_score", "incident"]] \
        .to_csv(config.RESULTS_DIR / "detected_incidents.csv")

    w = d.iloc[: 7 * 96]
    plt.figure(figsize=(11, 3.6))
    plt.plot(w.index, w["speed"], lw=1, label="Speed (km/h)")
    plt.scatter(w.index[w.anomaly == 1], w["speed"][w.anomaly == 1], c="red", s=25, label="Flagged anomaly")
    plt.legend(); plt.title("Incident detection (Isolation Forest) - first week")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "anomaly_detection.png", dpi=130)
