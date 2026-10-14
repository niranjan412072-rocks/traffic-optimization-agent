"""Train the Q-learning signal-control agent.

Run:  python -m src.train_rl --episodes 3000
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from src.rl_agent import train_q_learning

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pol, returns = train_q_learning(a.episodes, seed=a.seed)
    pol.save()
    print(f"Saved Q-table -> {config.Q_TABLE_PATH}")
    k = max(1, a.episodes // 60)
    smooth = np.convolve(returns, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(7, 3.6)); plt.plot(smooth)
    plt.xlabel("Training episode"); plt.ylabel("Episode return (higher is better)")
    plt.title("Q-learning training curve"); plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "rl_training_curve.png", dpi=130)
