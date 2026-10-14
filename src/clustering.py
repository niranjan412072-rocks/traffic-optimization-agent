"""Unsupervised pattern discovery with K-Means.

1. Daily traffic-profile clustering -> weekday / weekend / holiday-like days.
2. Intersection hotspot clustering -> priority tiers for the signal-control agent.

Run:  python -m src.clustering
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config


def cluster_daily_profiles(df, k=3, seed=config.SEED):
    d = df.copy()
    d["date"] = d.index.date
    d["hour"] = d.index.hour
    prof = d.pivot_table(index="date", columns="hour", values="volume", aggfunc="mean").dropna()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(prof.values)
    prof["cluster"] = km.labels_
    wk = pd.to_datetime(pd.Series(prof.index)).dt.dayofweek.values >= 5
    names, used = {}, {}
    max_peak = km.cluster_centers_.max()
    for c in range(k):
        share_wkend = wk[prof["cluster"].values == c].mean()
        peak = km.cluster_centers_[c].max()
        base = "weekend-like" if share_wkend > 0.6 else ("low-volume / holiday-like" if peak < 0.75 * max_peak else "weekday rush-hour")
        used[base] = used.get(base, 0) + 1
        names[c] = base if used[base] == 1 else f"{base} (variant {used[base]})"
    return km, prof, names


def hotspot_clusters(n=40, k=3, seed=config.SEED):
    """Synthetic city: n intersections with peak delay / volume / variability."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n, 2))
    centre_dist = np.linalg.norm(xy - np.array([5, 5]), axis=1)
    load = np.exp(-centre_dist / 4) + rng.normal(0, 0.08, n)
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                       "peak_delay_s": 20 + 90 * load,
                       "avg_volume": 300 + 700 * load,
                       "variability": 0.1 + 0.4 * load + rng.normal(0, 0.03, n)})
    Z = StandardScaler().fit_transform(df[["peak_delay_s", "avg_volume", "variability"]])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    df["cluster"] = km.labels_
    order = df.groupby("cluster")["peak_delay_s"].mean().sort_values().index.tolist()
    tier = {c: t for c, t in zip(order, ["Tier 3 (low)", "Tier 2 (medium)", "Tier 1 (critical)"][:k])}
    df["priority"] = df["cluster"].map(tier)
    return df


if __name__ == "__main__":
    df = pd.read_csv(config.TRAFFIC_CSV, index_col=0, parse_dates=True)
    km, prof, names = cluster_daily_profiles(df)
    sizes = prof["cluster"].value_counts().to_dict()
    print("Daily profile clusters:", {names[c]: int(n) for c, n in sizes.items()})
    plt.figure(figsize=(8, 3.6))
    for c in range(km.n_clusters):
        plt.plot(range(24), km.cluster_centers_[c], label=f"{names[c]} ({sizes[c]} days)")
    plt.xlabel("Hour of day"); plt.ylabel("Avg vehicles / 15 min"); plt.legend()
    plt.title("Typical daily traffic patterns (K-Means)"); plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "daily_pattern_clusters.png", dpi=130); plt.close()

    hs = hotspot_clusters()
    hs.to_csv(config.RESULTS_DIR / "intersection_hotspots.csv", index=False)
    plt.figure(figsize=(5.5, 5))
    for p, g in hs.groupby("priority"):
        plt.scatter(g.x, g.y, s=60, label=p)
    plt.legend(); plt.title("Congestion hotspot tiers (K-Means)"); plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "hotspot_clusters.png", dpi=130); plt.close()
    summary = {"daily_pattern_clusters": {names[c]: int(n) for c, n in sizes.items()},
               "hotspot_tiers": hs["priority"].value_counts().to_dict()}
    (config.RESULTS_DIR / "clustering_summary.json").write_text(json.dumps(summary, indent=2))
    print("Hotspot tiers:", summary["hotspot_tiers"])
