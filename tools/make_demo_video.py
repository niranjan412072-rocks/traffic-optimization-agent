"""Create a synthetic top-down intersection video so the vision module can be
tested without any dataset:  python tools/make_demo_video.py
"""
import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

W = H = 480
out = config.SAMPLE_DIR / "demo_intersection.avi"
vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), 15, (W, H))
rng = np.random.default_rng(1)
cars = []                                   # [x, y, vx, vy, colour]
for f in range(450):
    img = np.full((H, W, 3), 60, np.uint8)
    cv2.rectangle(img, (200, 0), (280, H), (40, 40, 40), -1)      # vertical road
    cv2.rectangle(img, (0, 200), (W, 280), (40, 40, 40), -1)      # horizontal road
    if f % 12 == 0:
        side = rng.integers(4)
        col = tuple(int(c) for c in rng.integers(120, 255, 3))
        if side == 0:   cars.append([225, 0, 0, 5, col])           # from N
        elif side == 1: cars.append([245, H, 0, -5, col])          # from S
        elif side == 2: cars.append([W, 225, -5, 0, col])          # from E
        else:           cars.append([0, 245, 5, 0, col])           # from W
    for c in cars:
        c[0] += c[2]; c[1] += c[3]
        vertical = c[2] == 0
        w, h = (28, 48) if vertical else (48, 28)
        cv2.rectangle(img, (int(c[0] - w / 2), int(c[1] - h / 2)), (int(c[0] + w / 2), int(c[1] + h / 2)), c[4], -1)
    cars = [c for c in cars if -60 < c[0] < W + 60 and -60 < c[1] < H + 60]
    vw.write(img)
vw.release()
print("Saved", out)
