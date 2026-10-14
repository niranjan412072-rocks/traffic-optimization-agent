@echo off
REM Full pipeline for Windows (Command Prompt / PowerShell)
python -m src.data_generator --days 180 || exit /b 1
python -m src.forecasting --horizons 1 2 4 || exit /b 1
python -m src.anomaly || exit /b 1
python -m src.clustering || exit /b 1
python -m src.routing || exit /b 1
python tools\make_demo_video.py || exit /b 1
python -m src.vision --video data\samples\demo_intersection.avi || exit /b 1
python -m src.train_rl --episodes 3000 || exit /b 1
python -m src.evaluate --episodes 20 || exit /b 1
python -m src.agent --scenario rush || exit /b 1
python -m unittest discover -s tests -t . || exit /b 1
echo Done. Launch the dashboard with: streamlit run app\streamlit_app.py
