"""Single 4-way intersection traffic-signal simulator (pure Python, no SUMO needed).

Approaches: N, S, E, W.  Phase 0 = North/South green, phase 1 = East/West green.
One step = 5 seconds.  Action 0 = keep current phase, 1 = switch phase.
Switching costs one lost step (all-red/yellow) and needs a minimum green time.

The class follows the Gymnasium API (reset/step), so it also works with
Stable-Baselines3 when `gymnasium` is installed.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base, HAS_GYM = gym.Env, True
except ImportError:                       # the project also runs without gymnasium
    gym = spaces = None
    _Base, HAS_GYM = object, False

# arrival rates (vehicles per 5-second step) for [N, S, E, W]
SCENARIOS = {
    "normal":     (0.45, 0.45, 0.40, 0.40),
    "rush":       (0.75, 0.75, 0.70, 0.70),
    "imbalanced": (0.95, 0.95, 0.30, 0.30),
    "shifting":   None,                   # NS-heavy first half, EW-heavy second half
}
MIXED = ["normal", "rush", "imbalanced", "shifting"]


class TrafficSignalEnv(_Base):
    def __init__(self, scenario="mixed", episode_steps=360, step_seconds=5,
                 sat_flow=2, min_green=2, yellow_steps=1, max_queue=60, seed=None):
        super().__init__()
        self.scenario = scenario
        self.episode_steps = episode_steps
        self.step_seconds = step_seconds
        self.sat_flow = sat_flow
        self.min_green = min_green
        self.yellow_steps = yellow_steps
        self.max_queue = max_queue
        self.rng = np.random.default_rng(seed)
        if HAS_GYM:
            self.observation_space = spaces.Box(0.0, 5.0, shape=(7,), dtype=np.float32)
            self.action_space = spaces.Discrete(2)
        self.reset()

    # ---------- helpers ----------
    def _rates(self):
        if self.active_scenario == "shifting":
            ns_heavy = self.t < self.episode_steps // 2
            base = (0.9, 0.9, 0.3, 0.3) if ns_heavy else (0.3, 0.3, 0.9, 0.9)
        else:
            base = SCENARIOS[self.active_scenario]
        return np.array(base) * self.jitter

    def raw_state(self):
        """(queue on green approaches, queue on red approaches, can_switch)"""
        q = self.q
        g = q[0] + q[1] if self.phase == 0 else q[2] + q[3]
        r = q[2] + q[3] if self.phase == 0 else q[0] + q[1]
        can = int(self.yellow_left == 0 and self.time_in_phase >= self.min_green)
        return float(g), float(r), can

    def observation(self):
        return np.array([*np.minimum(self.q / 20.0, 5.0), self.phase,
                         min(self.time_in_phase, 20) / 20.0, float(self.yellow_left > 0)], dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.active_scenario = self.rng.choice(MIXED) if self.scenario == "mixed" else self.scenario
        self.jitter = self.rng.uniform(0.9, 1.1)
        self.q = np.zeros(4)
        self.phase = 0
        self.time_in_phase = 0
        self.yellow_left = 0
        self.t = 0
        self.total_queue_steps = 0.0
        self.arrived = 0
        self.departed = 0
        self.overflow = 0
        self.switches = 0
        return self.observation(), {"scenario": self.active_scenario}

    def step(self, action):
        switched = 0
        if int(action) == 1 and self.yellow_left == 0 and self.time_in_phase >= self.min_green:
            self.phase = 1 - self.phase
            self.time_in_phase = 0
            self.yellow_left = self.yellow_steps
            self.switches += 1
            switched = 1

        discharged = 0
        if self.yellow_left > 0:                           # lost time, nobody moves
            self.yellow_left -= 1
        else:
            green = (0, 1) if self.phase == 0 else (2, 3)
            for a in green:
                out = min(self.q[a], self.sat_flow)
                self.q[a] -= out
                discharged += out
            self.time_in_phase += 1

        arrivals = self.rng.poisson(self._rates())         # same draws every step -> paired comparisons
        new_q = self.q + arrivals
        self.overflow += int(np.maximum(new_q - self.max_queue, 0).sum())
        self.q = np.minimum(new_q, self.max_queue)

        self.arrived += int(arrivals.sum())
        self.departed += int(discharged)
        self.total_queue_steps += float(self.q.sum())
        self.t += 1

        reward = -float(self.q.sum()) / 10.0 - 0.3 * switched
        truncated = self.t >= self.episode_steps
        info = {"queue": float(self.q.sum()), "switched": switched}
        return self.observation(), reward, False, truncated, info

    # ---------- metrics ----------
    def metrics(self):
        arrived = max(self.arrived, 1)
        hours = self.t * self.step_seconds / 3600
        return {"avg_delay_s": self.total_queue_steps * self.step_seconds / arrived,
                "avg_queue": self.total_queue_steps / max(self.t, 1),
                "throughput_ratio": self.departed / arrived,
                "vehicles_per_hour": self.arrived / max(hours, 1e-9),
                "overflow": self.overflow,
                "switches": self.switches}
