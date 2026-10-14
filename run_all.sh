#!/usr/bin/env bash
# Full pipeline (Linux / macOS / Git-Bash).  Usage: bash run_all.sh
set -e
python -m src.data_generator --days 180
python -m src.forecasting --horizons 1 2 4
python -m src.anomaly
python -m src.clustering
python -m src.routing
python tools/make_demo_video.py
python -m src.vision --video data/samples/demo_intersection.avi
python -m src.train_rl --episodes 3000
python -m src.evaluate --episodes 20
python -m src.agent --scenario rush
python -m unittest discover -s tests -t .
echo "Done. Launch the dashboard with:  streamlit run app/streamlit_app.py"
