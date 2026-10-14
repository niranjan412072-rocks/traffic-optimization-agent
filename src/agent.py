"""The agentic layer: SENSE -> PREDICT -> DECIDE -> ACT -> EXPLAIN -> LEARN.

* SENSE   queue lengths (from the simulator, or from camera counts via src.vision)
* PREDICT short-term demand with the forecasting model; anomaly check on speed/volume
* DECIDE  RL policy picks keep/switch
* ACT     applies the signal phase
* EXPLAIN human-readable reason for every phase change (template or IBM watsonx.ai)
* LEARN   Q-table can be retrained on new data with src.train_rl

Run:  python -m src.agent --scenario rush
"""
import argparse
import numpy as np
import pandas as pd
import config
from src.signal_env import TrafficSignalEnv
from src.rl_agent import QLearningPolicy
from src.forecasting import load_bundle, predict_latest, load_data
from src.explain import get_explainer


class TrafficOptimizationAgent:
    def __init__(self, policy, forecast_bundle=None, explainer=None):
        self.policy, self.forecast = policy, forecast_bundle
        self.explainer = explainer or get_explainer(False)

    def forecast_outlook(self, history):
        if self.forecast is None or history is None:
            return None
        pred = predict_latest(self.forecast, history)
        ratio = pred / self.forecast["p90"]
        level = "HIGH" if ratio >= 0.9 else "MODERATE" if ratio >= 0.5 else "LOW"
        return {"predicted_volume": pred, "ratio": ratio, "level": level}

    def run_episode(self, env, history=None, seed=None):
        env.reset(seed=seed)
        outlook = self.forecast_outlook(history)
        level = outlook["level"] if outlook else None
        log, done = [], False
        while not done:
            g, r, can = env.raw_state()                  # SENSE
            phase_before = env.phase
            action = self.policy.act(env)                # DECIDE
            reason = ""
            if action == 1 and can:                      # EXPLAIN
                reason = self.explainer.explain({"phase": phase_before, "green_q": g, "red_q": r,
                                                 "forecast_level": level})
            _, _, _, done, info = env.step(action)       # ACT
            log.append({"t_sec": env.t * env.step_seconds, "phase": phase_before, "green_queue": g,
                        "red_queue": r, "action": "SWITCH" if (action == 1 and can) else "KEEP",
                        "forecast_level": level, "reason": reason})
        return pd.DataFrame(log), env.metrics(), outlook


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="rush", choices=["normal", "rush", "imbalanced", "shifting"])
    ap.add_argument("--watsonx", action="store_true", help="use IBM watsonx.ai for explanations")
    a = ap.parse_args()
    if not config.Q_TABLE_PATH.exists():
        raise SystemExit("Train first:  python -m src.train_rl")
    history = None
    if config.TRAFFIC_CSV.exists():
        df = load_data()
        # context window ending shortly before an evening rush hour (a Wednesday, 17:00)
        end = df.index[(df.index.dayofweek == 2) & (df.index.hour == 17) & (df.index.minute == 0)][-3]
        history = df.loc[:end].tail(config.STEPS_PER_HOUR * 24 * 8)
    agent = TrafficOptimizationAgent(QLearningPolicy.load(), load_bundle(4), get_explainer(a.watsonx))
    log, m, outlook = agent.run_episode(TrafficSignalEnv(scenario=a.scenario), history, seed=7)
    log.to_csv(config.RESULTS_DIR / "agent_log.csv", index=False)
    if outlook:
        print(f"Forecast (60 min ahead): {outlook['predicted_volume']:.0f} veh/15min -> {outlook['level']}")
    print("Episode metrics:", {k: round(v, 2) for k, v in m.items()})
    print(log[log.action == "SWITCH"].head(5)[["t_sec", "reason"]].to_string(index=False))
    print(f"Full decision log -> {config.RESULTS_DIR / 'agent_log.csv'}")
