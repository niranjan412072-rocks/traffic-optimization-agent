# AI Traffic Optimization Agent for Smart Cities (SDG 11)

An **agentic AI/ML system** that predicts congestion, detects incidents, counts vehicles from camera
footage, controls traffic signals with **reinforcement learning**, and recommends congestion-aware routes.
Built for the IBM SkillsBuild AI/ML internship final project (SDG 11 - Sustainable Cities and Communities).

**Everything uses free, open-source tools and runs on a normal laptop CPU. No paid API, GPU or dataset
download is required** (a synthetic data generator is included; real datasets are optional).

---------------------------------------------------------------------

## 1. What the agent does

```
 SENSE            PREDICT                 DECIDE              ACT              EXPLAIN
 queues / camera  15-60 min forecast      RL policy:          set signal       human-readable reason
 counts (YOLO /   + incident detection    keep / switch       phase            (template or IBM watsonx.ai)
 OpenCV)          + congestion level      phase
                                   \                                         /
                                    +----> route advisory (A*) <------------+
```

| # | Module | AI/ML technique | File |
|---|--------|-----------------|------|
| 1 | Traffic forecasting | Gradient Boosting (scikit-learn) / XGBoost, lag features | `src/forecasting.py` |
| 2 | Vehicle detection & counting | YOLOv8 (optional) or OpenCV MOG2 background subtraction | `src/vision.py` |
| 3 | Signal control | Reinforcement learning (Q-learning; optional PPO/DQN via Stable-Baselines3) | `src/rl_agent.py`, `src/signal_env.py` |
| 4 | Incident detection | Isolation Forest (unsupervised anomaly detection) | `src/anomaly.py` |
| 5 | Pattern & hotspot discovery | K-Means clustering | `src/clustering.py` |
| 6 | Route recommendation | A* search + BPR volume-delay model on a graph (NetworkX) | `src/routing.py` |
| 7 | Agent orchestration | Sense -> Predict -> Decide -> Act -> Explain loop | `src/agent.py` |
| 8 | Explanations | Template, or IBM watsonx.ai Granite (optional) | `src/explain.py` |
| 9 | Dashboard | Streamlit | `app/streamlit_app.py` |

## 2. Tech stack (all free)

| Purpose | Tool |
|---|---|
| Language | Python 3.10 - 3.12 |
| Data | NumPy, Pandas |
| ML / clustering / anomaly | scikit-learn (+ optional XGBoost) |
| RL | Custom Q-learning (NumPy); optional Stable-Baselines3 + Gymnasium |
| Computer vision | OpenCV; optional Ultralytics YOLOv8 |
| Graphs / routing | NetworkX |
| Plots | Matplotlib |
| Dashboard | Streamlit |
| LLM explanations (optional) | IBM watsonx.ai (check IBM Cloud for current free-tier availability) |
| Dev / hosting | VS Code or Google Colab, GitHub, Streamlit Community Cloud (free) |

## 3. Folder structure

```
traffic-optimization-agent/
├── README.md
├── requirements.txt            # core modules
├── requirements-optional.txt   # xgboost, gymnasium, stable-baselines3, ultralytics, watsonx
├── config.py                   # paths, constants, impact assumptions
├── run_all.sh / run_all.bat    # one-command pipeline
├── src/
│   ├── data_generator.py       # synthetic 15-min traffic data (weather, holidays, incidents)
│   ├── forecasting.py          # forecasting models + baselines
│   ├── vision.py               # vehicle counting per approach (N/S/E/W)
│   ├── signal_env.py           # 4-way intersection simulator (Gym-style)
│   ├── rl_agent.py             # fixed-time, actuated, Q-learning policies
│   ├── train_rl.py             # train Q-learning
│   ├── train_sb3.py            # optional PPO/DQN
│   ├── anomaly.py              # Isolation Forest incident detection
│   ├── clustering.py           # K-Means patterns & hotspots
│   ├── routing.py              # A* congestion-aware routing
│   ├── explain.py              # decision explanations
│   ├── agent.py                # the agent loop
│   └── evaluate.py             # comparison + impact summary
├── app/streamlit_app.py        # dashboard
├── tools/make_demo_video.py    # synthetic intersection video for the vision demo
├── tests/test_core.py          # unit tests
├── data/   models/   results/  # generated data, trained models, charts & metrics
└── docs/                       # architecture + report outline
```

## 4. Setup (step by step)

### 4.1 Create a virtual environment

**Windows (PowerShell / CMD)**
```bash
cd traffic-optimization-agent
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```
**Linux / macOS**
```bash
cd traffic-optimization-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 Install the core modules
```bash
pip install -r requirements.txt
```
Installs: `numpy pandas scikit-learn matplotlib networkx joblib opencv-python streamlit`.

### 4.3 (Optional) Install upgrades
```bash
pip install xgboost                      # alternative forecasting model
pip install gymnasium stable-baselines3  # PPO / DQN agents (also installs PyTorch)
pip install ultralytics                  # YOLOv8 detection (also installs PyTorch)
pip install ibm-watsonx-ai               # IBM watsonx.ai explanations
```
Smaller CPU-only PyTorch first (saves disk space):
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## 5. Run the project

### Option A - everything at once
```bash
bash run_all.sh          # Linux / macOS / Git-Bash
run_all.bat              # Windows
```

### Option B - step by step
```bash
python -m src.data_generator --days 180      # 1. synthetic traffic data  -> data/synthetic_traffic.csv
python -m src.forecasting --horizons 1 2 4   # 2. forecast 15 / 30 / 60 min ahead
python -m src.anomaly                        # 3. incident detection
python -m src.clustering                     # 4. daily-pattern + hotspot clusters
python -m src.routing                        # 5. congestion-aware routing
python tools/make_demo_video.py              # 6. demo video for vision
python -m src.vision --video data/samples/demo_intersection.avi
python -m src.train_rl --episodes 3000       # 7. train the RL signal agent (~1 min)
python -m src.evaluate --episodes 20         # 8. compare policies + impact summary
python -m src.agent --scenario rush          # 9. run the full agent, writes results/agent_log.csv
python -m unittest discover -s tests -t .    # 10. run tests
streamlit run app/streamlit_app.py           # 11. dashboard (http://localhost:8501)
```

### Optional commands
```bash
python -m src.forecasting --model xgboost                         # XGBoost forecaster
python -m src.forecasting --metro-csv Metro_Interstate_Traffic_Volume.csv   # real Kaggle data (hourly)
python -m src.vision --image my_photo.jpg --backend yolo          # YOLOv8 on a photo
python -m src.train_sb3 --algo ppo --timesteps 300000             # PPO agent (then re-run evaluate)
python -m src.agent --scenario shifting --watsonx                 # IBM watsonx.ai explanations
```
For watsonx set `WATSONX_URL`, `WATSONX_API_KEY`, `WATSONX_PROJECT_ID` (and optionally `WATSONX_MODEL_ID`) as environment variables. Without them the agent automatically uses template explanations.

### Real data and real video (recommended for extra marks)
* Forecasting: Kaggle *Metro Interstate Traffic Volume* -> `--metro-csv` (loader included).
* Vision: any fixed-camera traffic clip (e.g. free videos from Pexels, or the Kaggle vehicle-detection datasets) -> `python -m src.vision --video clip.mp4 --backend yolo`.

## 6. Results from the included run (synthetic data / simulation)

Generated by `python -m src.evaluate` (see `results/summary.md` for the live version).

**Signal control - average delay per vehicle (seconds, lower is better)**

| Scenario | Fixed-time | Rule-based actuated | RL agent (Q-learning) | RL vs fixed-time |
|---|---|---|---|---|
| normal | 20.9 | 13.9 | 14.3 | -31% |
| rush | 32.7 | 47.5 | 31.0 | -5% |
| imbalanced | 85.3 | 17.2 | 17.9 | -79% |
| shifting | 53.8 | 16.8 | 17.5 | -67% |

**Forecasting (test set)**: 15 min MAE 25.0 (R2 0.976), 30 min MAE 26.2, 60 min MAE 26.9 (R2 0.969) vehicles per 15-min interval; MAE is 34% / 50% / 70% lower than persistence at 15 / 30 / 60 min.
**Incident detection**: precision 0.65, recall 0.90, all 30 injected incidents detected.
**Routing**: 21% mean travel-time saving versus shortest-distance routing in the simulated congested city.
**Vision demo**: counts vehicles per approach on the included synthetic video (OpenCV backend).

### Honest interpretation (important for your report)
* The RL agent beats the fixed-time signal in every scenario: by a large margin in imbalanced and shifting traffic, and only slightly (about 5%) in heavy rush traffic.
* It performs about the same as the hand-written actuated rule in most scenarios (slightly worse in light traffic, clearly better in rush). Its advantage is that it **learns** the behaviour from experience instead of being hand-tuned, and it can be retrained for any intersection.
* All numbers come from **synthetic data and a simplified simulator**. They show the method works; they are not proof of real-world savings. Validate with real data / SUMO before claiming deployment benefits.
* Fuel/CO2 estimates use planning assumptions in `config.py` (0.8 L/h idling fuel, 2.31 kg CO2 per litre).

## 7. Impact (SDG 11)
Less waiting time, lower fuel use and emissions, faster emergency response and better data for city planners (SDG 11.2 safe & sustainable transport, 11.6 environmental impact of cities).

## 8. Troubleshooting
* `ModuleNotFoundError: src` -> run commands from the project root with `python -m src.<module>`.
* `ModuleNotFoundError: cv2` -> `pip install opencv-python`.
* Dashboard says models are missing -> run steps 1, 2 and 7 first.
* YOLO errors / offline: use `--backend opencv`; YOLO needs internet once to download `yolov8n.pt`.
* Streamlit port busy: `streamlit run app/streamlit_app.py --server.port 8502`.

## 9. Limitations & future work
* Single intersection; extend to a multi-intersection network (SUMO + TraCI, multi-agent RL).
* Q-learning uses discretised queues; try PPO/DQN (included script) or deep RL on camera features.
* Replace synthetic data with city sensor/CCTV data; add pedestrian and emergency-vehicle priority.
* Deploy on Streamlit Community Cloud or an edge device (e.g. Raspberry Pi).

## 10. License
MIT - free to use for learning and academic submission.
