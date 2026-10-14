"""Synthetic urban traffic data generator (free, no downloads needed).

Creates 15-minute traffic records with realistic daily/weekly patterns,
weather effects, holidays and injected incidents (accidents / breakdowns).
Swap in a real dataset (e.g. Kaggle "Metro Interstate Traffic Volume") with
`python -m src.forecasting --metro-csv path/to/file.csv`.

Run:  python -m src.data_generator --days 180
"""
import argparse
import numpy as np
import pandas as pd
import config


def _peak(h, mu, sd):
    return np.exp(-0.5 * ((h - mu) / sd) ** 2)


def generate_traffic_data(days=180, freq_min=config.FREQ_MIN, seed=config.SEED,
                          start="2025-01-01", n_incidents=30):
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 // freq_min
    idx = pd.date_range(start, periods=n, freq=f"{freq_min}min")
    hour = idx.hour.values + idx.minute.values / 60
    dow = idx.dayofweek.values

    weekday = 0.25 + 0.9 * _peak(hour, 8.5, 1.5) + 1.0 * _peak(hour, 18, 1.8) + 0.35 * _peak(hour, 13, 3)
    weekend = 0.20 + 0.55 * _peak(hour, 12, 3.5) + 0.35 * _peak(hour, 18, 2.5)
    profile = np.where(dow < 5, weekday, weekend)

    # holidays (calendar information known in advance)
    day_index = idx.normalize()
    uniq_days = day_index.unique()
    hol = uniq_days[rng.choice(len(uniq_days), size=max(1, days // 30), replace=False)]
    is_holiday = day_index.isin(hol).astype(int)

    # weather: seasonal temperature + rainy days
    day_frac = np.arange(n) / (24 * 60 / freq_min)
    temp = 24 + 8 * np.sin(2 * np.pi * day_frac / 365) + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1, n)
    rainy_day = (rng.random(len(uniq_days)) < 0.2)
    rainy_map = dict(zip(uniq_days, rainy_day))
    rain_flag = np.array([rainy_map[d] for d in day_index])
    rain = rain_flag * rng.gamma(2.0, 1.2, n) * (rng.random(n) < 0.6)

    volume = 700 * profile * (1 - 0.40 * is_holiday) * (1 - 0.08 * np.minimum(rain, 5) / 5)
    volume = volume * rng.normal(1.0, 0.06, n)

    speed = 50 * (1 - 0.5 * (volume / 800) ** 2) * (1 - 0.05 * np.minimum(rain, 5) / 5)
    speed = speed * rng.normal(1.0, 0.03, n)

    # incidents: capacity drop -> lower speed and volume
    incident = np.zeros(n, dtype=int)
    for s in rng.choice(n - 10, size=n_incidents, replace=False):
        d = rng.integers(2, 7)
        incident[s:s + d] = 1
    volume = np.where(incident == 1, volume * 0.55, volume)
    speed = np.where(incident == 1, speed * 0.35, speed)

    df = pd.DataFrame({
        "volume": np.clip(volume, 0, None).round(0),
        "speed": np.clip(speed, 3, 60).round(1),
        "temp": temp.round(1),
        "rain": rain.round(2),
        "is_holiday": is_holiday,
        "incident": incident,
    }, index=idx)
    df.index.name = "timestamp"
    df["congestion_index"] = (1 - df["speed"] / 50).clip(0, 1).round(3)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    data = generate_traffic_data(days=a.days, seed=a.seed)
    data.to_csv(config.TRAFFIC_CSV)
    print(f"Saved {len(data):,} rows -> {config.TRAFFIC_CSV}")
    print(data.describe().round(1).T[["mean", "min", "max"]])
