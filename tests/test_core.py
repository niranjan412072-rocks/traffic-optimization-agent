import unittest
import numpy as np
import cv2
from src.signal_env import TrafficSignalEnv
from src.rl_agent import FixedTimePolicy, ActuatedPolicy, QLearningPolicy, rollout
from src.data_generator import generate_traffic_data
from src.forecasting import make_features, train_and_evaluate
from src.routing import build_city_graph, apply_congestion, hotspot_congestion, \
    fastest_route, shortest_distance_route, route_time
from src.vision import VehicleCounter, zone_of


class TestEnv(unittest.TestCase):
    def test_reset_step_shapes(self):
        env = TrafficSignalEnv(scenario="rush", seed=1)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape, (7,))
        obs, r, term, trunc, info = env.step(0)
        self.assertEqual(obs.shape, (7,))
        self.assertLessEqual(r, 0)
        self.assertFalse(term)

    def test_queues_nonnegative_and_episode_ends(self):
        env = TrafficSignalEnv(scenario="imbalanced")
        tr, m = rollout(env, ActuatedPolicy(), seed=3)
        self.assertEqual(len(tr), env.episode_steps)
        self.assertTrue((tr[["q_N", "q_S", "q_E", "q_W"]].values >= 0).all())

    def test_paired_seeds_give_same_arrivals(self):
        env = TrafficSignalEnv(scenario="rush")
        _, m1 = rollout(env, FixedTimePolicy(), seed=5)
        _, m2 = rollout(env, ActuatedPolicy(), seed=5)
        self.assertAlmostEqual(m1["vehicles_per_hour"], m2["vehicles_per_hour"])

    def test_min_green_enforced(self):
        env = TrafficSignalEnv(scenario="normal", seed=0)
        env.reset(seed=0)
        env.step(1)                      # switch refused: min green not met
        self.assertEqual(env.switches, 0)

    def test_q_policy_runs(self):
        pol = QLearningPolicy()
        _, m = rollout(TrafficSignalEnv(scenario="normal"), pol, seed=1)
        self.assertIn("avg_delay_s", m)


class TestForecast(unittest.TestCase):
    def test_features_no_future_leak_and_training(self):
        df = generate_traffic_data(days=30, seed=1)
        X, y = make_features(df, horizon=2)
        self.assertEqual(len(X), len(y))
        # target is the volume 2 steps ahead
        self.assertEqual(y.iloc[10], df["volume"].iloc[12])
        res = train_and_evaluate(df, 2, save=False, plot=False)
        self.assertLess(res["model"]["MAE"], res["baseline_persistence"]["MAE"])


class TestRouting(unittest.TestCase):
    def test_ai_route_not_slower(self):
        G = build_city_graph(6)
        apply_congestion(G, hotspot_congestion(G))
        s, t = (0, 2), (5, 3)
        self.assertLessEqual(route_time(G, fastest_route(G, s, t)) + 1e-9,
                             route_time(G, shortest_distance_route(G, s, t)))


class TestVision(unittest.TestCase):
    def test_zone_of(self):
        self.assertEqual(zone_of(240, 10, 480, 480), "N")
        self.assertEqual(zone_of(470, 240, 480, 480), "E")

    def test_opencv_detects_moving_box(self):
        vc = VehicleCounter("opencv")
        blank = np.zeros((240, 240, 3), np.uint8)
        for _ in range(40):
            vc.detect(blank)
        frame = blank.copy()
        cv2.rectangle(frame, (100, 20), (140, 60), (255, 255, 255), -1)
        counts, boxes = vc.count_zones(frame)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(counts["N"], 1)


if __name__ == "__main__":
    unittest.main()
