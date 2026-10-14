# Results summary (auto-generated)

## Signal control (avg delay per vehicle, seconds)

| Scenario | Actuated rule-based | Fixed-time (30s) | RL agent (Q-learning) |
|---|---|---|---|
| imbalanced | 17.2 | 85.3 | 17.9 |
| normal | 13.9 | 20.9 | 14.3 |
| rush | 47.5 | 32.7 | 31.0 |
| shifting | 16.8 | 53.8 | 17.5 |


## Impact of `RL agent (Q-learning)` vs fixed-time signals (per intersection, planning estimate)

| Scenario | Delay reduction | Vehicle-hours saved / h | Fuel saved L/h | CO2 saved kg/h |
|---|---|---|---|---|
| normal | 31.4% | 2.19 | 1.75 | 4.04 |
| rush | 5.2% | 0.98 | 0.79 | 1.82 |
| imbalanced | 79.0% | 33.02 | 26.42 | 61.03 |
| shifting | 67.4% | 17.13 | 13.70 | 31.65 |

## Forecasting (test set)

| Horizon | MAE | R2 | MAE vs persistence |
|---|---|---|---|
| 15 min | 25.0 | 0.976 | 33.7% better |
| 30 min | 26.2 | 0.971 | 50.1% better |
| 60 min | 26.9 | 0.969 | 69.6% better |

## Incident detection
Precision 0.65, recall 0.90, events detected 30/30.

## Routing
Mean travel-time saving vs shortest-distance route: 21.2% over 300 trips (synthetic congested city).

> All numbers come from simulation / synthetic data. Fuel & CO2 use 0.8 L/h idling and 2.31 kg CO2/L as assumptions.