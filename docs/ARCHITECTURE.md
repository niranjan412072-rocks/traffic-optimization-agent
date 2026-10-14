# Architecture

```
            +------------------+      +--------------------+
 Camera --> | vision.py        |      | data_generator.py  |  (or real CSV)
 / video    | YOLOv8 / OpenCV  |      +---------+----------+
            +--------+---------+                |
                     | queue counts             v
                     |                +--------------------+     +------------------+
                     |                | forecasting.py     |     | anomaly.py       |
                     |                | HistGB / XGBoost   |     | IsolationForest  |
                     |                +---------+----------+     +---------+--------+
                     v                          | demand level            | incident flag
            +------------------------------------------------------------------+
            |                       agent.py (orchestrator)                    |
            |   SENSE -> PREDICT -> DECIDE (RL policy) -> ACT -> EXPLAIN       |
            +----------+-------------------------------+-----------------------+
                       |                               |
              +--------v--------+            +---------v----------+
              | signal_env.py   |            | routing.py (A*)    |
              | rl_agent.py     |            | congestion-aware   |
              | Q-learning/PPO  |            | route advice       |
              +--------+--------+            +---------+----------+
                       |                               |
                       +-----------> Streamlit dashboard <---- clustering.py (hotspots)
```

## RL formulation
* **State**: queue on green side, queue on red side (6 bins each), can-switch flag -> 72 states (Q-learning); 7-D vector for PPO/DQN.
* **Actions**: 0 keep phase, 1 switch phase (min green 10 s, 5 s lost time per switch).
* **Reward**: -(total queue)/10 - 0.3 per switch.
* **Training**: mixed scenarios (normal, rush, imbalanced, shifting), epsilon-greedy, 3000 episodes.
* **Evaluation**: unseen random seeds, paired arrivals across policies for a fair comparison.

## Data flow of impact metrics
delay saved per vehicle x vehicles per hour -> vehicle-hours saved -> x idle fuel rate -> litres -> x 2.31 kg CO2/L.
