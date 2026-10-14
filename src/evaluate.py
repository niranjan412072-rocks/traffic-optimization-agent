"""Compare signal-control policies and build the impact summary.

Run:  python -m src.evaluate --episodes 20
Outputs: results/signal_comparison.csv/.png, results/summary.json, results/summary.md
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from src.signal_env import TrafficSignalEnv
from src.rl_agent import FixedTimePolicy, ActuatedPolicy, QLearningPolicy, rollout

SCENARIOS = ["normal", "rush", "imbalanced", "shifting"]
EVAL_SEED = 10_000                      # never used during training


def load_policies():
    pols = [FixedTimePolicy(), ActuatedPolicy()]
    if config.Q_TABLE_PATH.exists():
        pols.append(QLearningPolicy.load())
    if config.SB3_MODEL_PATH.exists():
        try:
            from src.rl_agent import SB3Policy
            pols.append(SB3Policy())
        except Exception as e:                       # optional dependency
            print("Skipping PPO model:", e)
    return pols


def compare(episodes=20):
    rows = []
    for sc in SCENARIOS:
        env = TrafficSignalEnv(scenario=sc)
        for pol in load_policies():
            ms = [rollout(env, pol, seed=EVAL_SEED + i)[1] for i in range(episodes)]
            row = {"scenario": sc, "policy": pol.name}
            row.update({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
            rows.append(row)
    return pd.DataFrame(rows)


def impact(df):
    """Fuel / CO2 saved by the best RL policy relative to fixed-time (planning estimate)."""
    rl_name = [p for p in df.policy.unique() if p.startswith("RL")][-1]
    out = {}
    for sc in SCENARIOS:
        f = df[(df.scenario == sc) & (df.policy == FixedTimePolicy.name)].iloc[0]
        r = df[(df.scenario == sc) & (df.policy == rl_name)].iloc[0]
        saved_s_per_veh = f.avg_delay_s - r.avg_delay_s
        veh_s_saved_per_h = saved_s_per_veh * r.vehicles_per_hour
        fuel_l = veh_s_saved_per_h / 3600 * config.IDLE_FUEL_L_PER_HOUR
        out[sc] = {"delay_reduction_%": 100 * saved_s_per_veh / f.avg_delay_s,
                   "vehicle_hours_saved_per_hour": veh_s_saved_per_h / 3600,
                   "fuel_saved_L_per_hour": fuel_l,
                   "co2_saved_kg_per_hour": fuel_l * config.CO2_KG_PER_LITRE}
    out["rl_policy"] = rl_name
    return out


def plot(df):
    pivot = df.pivot(index="scenario", columns="policy", values="avg_delay_s").loc[SCENARIOS]
    ax = pivot.plot(kind="bar", figsize=(8, 4), rot=0)
    ax.set_ylabel("Average delay per vehicle (s)  - lower is better")
    ax.set_title("Signal control: average vehicle delay by scenario")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "signal_comparison.png", dpi=130); plt.close()


def _md_table(pv):
    cols = list(pv.columns)
    lines = ["| Scenario | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in pv.iterrows():
        lines.append(f"| {idx} | " + " | ".join(f"{row[c]:.1f}" for c in cols) + " |")
    return "\n".join(lines)


def write_summary(sig, imp):
    summary = {"signal_control": sig.round(3).to_dict(orient="records"), "impact_estimate": imp}
    for name in ("forecast_metrics", "anomaly_metrics", "routing_metrics", "clustering_summary"):
        p = config.RESULTS_DIR / f"{name}.json"
        if p.exists():
            summary[name] = json.loads(p.read_text())
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# Results summary (auto-generated)\n", "## Signal control (avg delay per vehicle, seconds)\n",
          _md_table(sig.pivot(index="scenario", columns="policy", values="avg_delay_s").round(1)), "\n",
          f"## Impact of `{imp['rl_policy']}` vs fixed-time signals (per intersection, planning estimate)\n",
          "| Scenario | Delay reduction | Vehicle-hours saved / h | Fuel saved L/h | CO2 saved kg/h |", "|---|---|---|---|---|"]
    for sc in SCENARIOS:
        i = imp[sc]
        md.append(f"| {sc} | {i['delay_reduction_%']:.1f}% | {i['vehicle_hours_saved_per_hour']:.2f} | "
                  f"{i['fuel_saved_L_per_hour']:.2f} | {i['co2_saved_kg_per_hour']:.2f} |")
    if "forecast_metrics" in summary:
        md += ["\n## Forecasting (test set)\n", "| Horizon | MAE | R2 | MAE vs persistence |", "|---|---|---|---|"]
        for k, v in summary["forecast_metrics"].items():
            md.append(f"| {v['horizon_min']} min | {v['model']['MAE']:.1f} | {v['model']['R2']:.3f} | "
                      f"{v['MAE_improvement_vs_persistence_%']:.1f}% better |")
    if "anomaly_metrics" in summary:
        a = summary["anomaly_metrics"]
        md.append(f"\n## Incident detection\nPrecision {a['precision']:.2f}, recall {a['recall']:.2f}, "
                  f"events detected {a['incident_events_detected']}/{a['incident_events']}.")
    if "routing_metrics" in summary:
        r = summary["routing_metrics"]
        md.append(f"\n## Routing\nMean travel-time saving vs shortest-distance route: {r['mean_time_saved_%']:.1f}% "
                  f"over {r['pairs']} trips (synthetic congested city).")
    md.append("\n> All numbers come from simulation / synthetic data. Fuel & CO2 use "
              f"{config.IDLE_FUEL_L_PER_HOUR} L/h idling and {config.CO2_KG_PER_LITRE} kg CO2/L as assumptions.")
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    sig = compare(a.episodes)
    sig.round(3).to_csv(config.RESULTS_DIR / "signal_comparison.csv", index=False)
    plot(sig)
    imp = impact(sig)
    write_summary(sig, imp)
    print(sig.pivot(index="scenario", columns="policy", values="avg_delay_s").round(1))
    for sc in SCENARIOS:
        print(f"{sc:>11}: delay reduction {imp[sc]['delay_reduction_%']:.1f}%")
