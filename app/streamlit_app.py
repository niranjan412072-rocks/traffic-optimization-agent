"""Streamlit dashboard.   Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config
from src.signal_env import TrafficSignalEnv
from src.rl_agent import FixedTimePolicy, ActuatedPolicy, QLearningPolicy, rollout
from src.forecasting import load_bundle, load_data, make_features
from src.anomaly import add_ratios, detect
from src.vision import VehicleCounter
from src import routing
import joblib
import subprocess


@st.cache_resource
def ensure_models():
    try:
        for p in (ROOT / "models").glob("*.joblib"):
            joblib.load(p)
        return "loaded"
    except Exception:
        steps = [
            ["-m", "src.data_generator", "--days", "180"],
            ["-m", "src.forecasting", "--horizons", "1", "2", "4"],
            ["-m", "src.anomaly"],
        ]
        for step in steps:
            subprocess.run([sys.executable, *step], cwd=ROOT, check=True)
        return "retrained"

st.set_page_config(page_title="AI Traffic Optimization Agent", layout="wide")
with st.spinner("Preparing models (first load can take a minute)..."):
    ensure_models()
st.title("AI Traffic Optimization Agent")
st.caption("SDG 11 - Sustainable Cities | forecasting + computer vision + reinforcement learning")

if not config.TRAFFIC_CSV.exists() or not config.Q_TABLE_PATH.exists():
    st.error("Run the pipeline first:  python -m src.data_generator && python -m src.forecasting && "
             "python -m src.train_rl   (see README.md)")
    st.stop()

tabs = st.tabs(["Overview", "Forecast", "Signal control", "Vision", "Routing", "Incidents"])

# ---------------- Overview ----------------
with tabs[0]:
    p = config.RESULTS_DIR / "summary.md"
    if p.exists():
        st.markdown(p.read_text())
    else:
        st.info("Run `python -m src.evaluate` to generate the summary.")

# ---------------- Forecast ----------------
with tabs[1]:
    h = st.selectbox("Forecast horizon", [1, 2, 4], index=2, format_func=lambda x: f"{x * config.FREQ_MIN} minutes")
    bundle = load_bundle(h)
    if bundle is None:
        st.warning("Run `python -m src.forecasting` first.")
    else:
        df = load_data()
        X, y = make_features(df, h)
        d = X.join(y.rename("y")).dropna().iloc[-3 * 96:]
        pred = bundle["model"].predict(d[bundle["features"]])
        chart = pd.DataFrame({"Actual": d["y"].values, "Predicted": pred}, index=d.index)
        st.line_chart(chart)
        st.metric("MAE on this window (vehicles / 15 min)", f"{np.abs(chart.Actual - chart.Predicted).mean():.1f}")

# ---------------- Signal control ----------------
with tabs[2]:
    c1, c2 = st.columns(2)
    scenario = c1.selectbox("Traffic scenario", ["normal", "rush", "imbalanced", "shifting"], index=1)
    seed = c2.number_input("Random seed", 0, 9999, 7)
    if st.button("Run simulation"):
        pols = [FixedTimePolicy(), ActuatedPolicy(), QLearningPolicy.load()]
        traces, rows = {}, []
        for pol in pols:
            tr, m = rollout(TrafficSignalEnv(scenario=scenario), pol, seed=int(seed))
            traces[pol.name] = tr["total_queue"].values
            rows.append({"Policy": pol.name, "Avg delay (s)": round(m["avg_delay_s"], 1),
                         "Avg queue": round(m["avg_queue"], 1), "Switches": m["switches"]})
        st.line_chart(pd.DataFrame(traces))
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---------------- Vision ----------------
with tabs[3]:
    st.write("Upload a traffic image or a short video clip (fixed camera).")
    backend = st.radio("Detector", ["auto", "opencv", "yolo"], horizontal=True)
    up = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    if up is not None:
        raw = up.read()
        suffix = Path(up.name).suffix.lower()
        try:
            if suffix in (".jpg", ".jpeg", ".png"):
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                vc = VehicleCounter(backend)
                counts, boxes = vc.count_zones(frame)
                st.image(cv2.cvtColor(vc.annotate(frame, boxes, counts), cv2.COLOR_BGR2RGB))
                st.write(f"Backend: **{vc.backend}**  | counts per approach: {counts}")
                if vc.backend == "opencv":
                    st.caption("A single image has no background model; use a video, or install ultralytics for YOLO.")
            else:
                tmp = config.SAMPLE_DIR / f"_upload{suffix}"
                tmp.write_bytes(raw)
                from src.vision import process_video
                df, be = process_video(tmp, backend, out_video=None)
                st.write(f"Backend: **{be}**  | sampled frames: {len(df)}")
                st.line_chart(df.set_index("frame")[["N", "S", "E", "W"]])
        except Exception as e:
            st.error(f"Vision error: {e}")
    demo = config.SAMPLE_DIR / "demo_intersection.avi"
    if demo.exists() and st.button("Analyse built-in demo video"):
        from src.vision import process_video
        df, be = process_video(demo, backend)
        st.write(f"Backend: **{be}**")
        st.line_chart(df.set_index("frame")[["N", "S", "E", "W"]])

# ---------------- Routing ----------------
with tabs[4]:
    strength = st.slider("Congestion strength at the city centre", 0.5, 3.0, 2.2)
    G = routing.build_city_graph()
    routing.apply_congestion(G, routing.hotspot_congestion(G, strength=strength))
    nodes = list(G.nodes)
    a, b = st.columns(2)
    src = a.selectbox("Origin", nodes, index=3)
    dst = b.selectbox("Destination", nodes, index=len(nodes) - 5)
    if src != dst:
        pa, pb = routing.shortest_distance_route(G, src, dst), routing.fastest_route(G, src, dst)
        ta, tb = routing.route_time(G, pa), routing.route_time(G, pb)
        m1, m2 = st.columns(2)
        m1.metric("Shortest-distance route", f"{ta:.1f} min")
        m2.metric("AI fastest route", f"{tb:.1f} min", f"{-(ta - tb):.1f} min")
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        routing.plot_routes(G, src, dst, pa, pb, ax=ax)
        st.pyplot(fig)

# ---------------- Incidents ----------------
with tabs[5]:
    if config.ANOMALY_MODEL_PATH.exists():
        model = joblib.load(config.ANOMALY_MODEL_PATH)
        d = detect(model, load_data())
        flagged = d[d.anomaly == 1][["volume", "speed", "vol_ratio", "speed_ratio", "anomaly_score", "incident"]]
        st.write(f"{len(flagged)} suspicious intervals flagged out of {len(d):,}.")
        st.dataframe(flagged.tail(50))
    else:
        st.warning("Run `python -m src.anomaly` first.")
