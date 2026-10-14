"""Signal-control policies: fixed-time baseline, actuated heuristic, and the
Q-learning reinforcement-learning agent (NumPy only, trains in about a minute).
"""
import numpy as np
import pandas as pd
import config
from src.signal_env import TrafficSignalEnv

EDGES = np.array([1, 4, 8, 14, 22])      # queue-length bins -> 6 levels
N_BINS = len(EDGES) + 1


class FixedTimePolicy:
    """Classic pre-timed signal: 30 s green for each direction."""
    name = "Fixed-time (30s)"

    def __init__(self, green_steps=6):
        self.green_steps = green_steps

    def reset(self): pass

    def act(self, env):
        return int(env.time_in_phase >= self.green_steps and env.yellow_left == 0)


class ActuatedPolicy:
    """Rule-based 'smart' baseline: switch when the red side is clearly longer."""
    name = "Actuated rule-based"

    def __init__(self, margin=3, max_green=12):
        self.margin, self.max_green = margin, max_green

    def reset(self): pass

    def act(self, env):
        g, r, can = env.raw_state()
        if not can:
            return 0
        if g == 0 and r > 0:
            return 1
        if r > g + self.margin:
            return 1
        if env.time_in_phase >= self.max_green and r > 0:
            return 1
        return 0


class QLearningPolicy:
    """Tabular Q-learning agent. State = (green-side queue, red-side queue, can_switch)."""
    name = "RL agent (Q-learning)"
    n_states = N_BINS * N_BINS * 2

    def __init__(self, q_table=None):
        self.Q = np.zeros((self.n_states, 2)) if q_table is None else q_table

    @staticmethod
    def state_index(env):
        g, r, can = env.raw_state()
        return (int(np.digitize(g, EDGES)) * N_BINS + int(np.digitize(r, EDGES))) * 2 + can

    def reset(self): pass

    def act(self, env):
        return int(np.argmax(self.Q[self.state_index(env)]))

    def save(self, path=config.Q_TABLE_PATH):
        np.save(path, self.Q)

    @classmethod
    def load(cls, path=config.Q_TABLE_PATH):
        return cls(np.load(path))


class SB3Policy:
    """Wrapper for a Stable-Baselines3 model (PPO/DQN) trained with src.train_sb3."""
    name = "RL agent (PPO)"

    def __init__(self, path=config.SB3_MODEL_PATH):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(path))

    def reset(self): pass

    def act(self, env):
        a, _ = self.model.predict(env.observation(), deterministic=True)
        return int(a)


def train_q_learning(episodes=3000, alpha=0.1, gamma=0.95, seed=0, log_every=100):
    rng = np.random.default_rng(seed)
    env = TrafficSignalEnv(scenario="mixed", seed=seed)
    pol = QLearningPolicy()
    returns = []
    for ep in range(episodes):
        env.reset()
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))
        s, total = pol.state_index(env), 0.0
        done = False
        while not done:
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(pol.Q[s]))
            _, r, _, done, _ = env.step(a)
            s2 = pol.state_index(env)
            target = r + (0.0 if done else gamma * pol.Q[s2].max())
            pol.Q[s, a] += alpha * (target - pol.Q[s, a])
            s, total = s2, total + r
        returns.append(total)
        if (ep + 1) % log_every == 0:
            print(f"episode {ep + 1:>5}/{episodes}  eps={eps:.2f}  "
                  f"avg return (last {log_every}) = {np.mean(returns[-log_every:]):.1f}")
    return pol, returns


def rollout(env, policy, seed=None):
    """Run one episode; returns (per-step DataFrame, metrics dict)."""
    env.reset(seed=seed)
    policy.reset()
    rows, done = [], False
    while not done:
        a = policy.act(env)
        _, _, _, done, info = env.step(a)
        rows.append({"t": env.t, "phase": env.phase, "q_N": env.q[0], "q_S": env.q[1],
                     "q_E": env.q[2], "q_W": env.q[3], "total_queue": info["queue"], "action": a})
    return pd.DataFrame(rows), env.metrics()
