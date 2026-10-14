"""OPTIONAL: train a PPO (or DQN) agent with Stable-Baselines3 on the same environment.

Needs:  pip install gymnasium stable-baselines3
Run:    python -m src.train_sb3 --algo ppo --timesteps 300000
"""
import argparse
import config
from src.signal_env import TrafficSignalEnv, HAS_GYM

if __name__ == "__main__":
    if not HAS_GYM:
        raise SystemExit("Install first:  pip install gymnasium stable-baselines3")
    from stable_baselines3 import PPO, DQN
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = TrafficSignalEnv(scenario="mixed", seed=a.seed)
    Algo = PPO if a.algo == "ppo" else DQN
    model = Algo("MlpPolicy", env, verbose=1, seed=a.seed)
    model.learn(total_timesteps=a.timesteps)
    model.save(str(config.SB3_MODEL_PATH.with_name(f"{a.algo}_signal.zip")))
    print("Saved to", config.MODELS_DIR)
