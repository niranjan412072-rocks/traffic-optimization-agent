"""Vehicle detection + per-approach counting from camera images / video.

Backends (automatic):
  * YOLOv8 (ultralytics)  - best accuracy, free; downloads yolov8n.pt (~6 MB) on first use
  * OpenCV MOG2 background subtraction - zero extra install, works for fixed cameras

Frame is split into four wedges (N/S/E/W) by its diagonals; counts feed the RL agent.

Run:  python -m src.vision --video data/samples/demo_intersection.avi
      python -m src.vision --image path/to/photo.jpg --backend yolo
"""
import argparse
import cv2
import numpy as np
import pandas as pd
import config

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
ZONES = ["N", "S", "E", "W"]


def zone_of(cx, cy, w, h):
    dx, dy = cx - w / 2, cy - h / 2
    if abs(dy) * w >= abs(dx) * h:
        return "N" if dy < 0 else "S"
    return "W" if dx < 0 else "E"


class VehicleCounter:
    def __init__(self, backend="auto", model_path="yolov8n.pt", conf=0.35, min_area=350):
        self.conf, self.min_area = conf, min_area
        self.model = None
        self.backend = "opencv"
        if backend in ("auto", "yolo"):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.backend = "yolo"
            except Exception as e:
                if backend == "yolo":
                    raise RuntimeError(f"YOLO backend unavailable: {e}")
        if self.backend == "opencv":
            self.bg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=True)

    def detect(self, frame):
        """Returns list of (x1, y1, x2, y2, label)."""
        if self.backend == "yolo":
            res = self.model(frame, conf=self.conf, verbose=False)[0]
            out = []
            for b in res.boxes:
                c = int(b.cls[0])
                if c in COCO_VEHICLES:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    out.append((x1, y1, x2, y2, COCO_VEHICLES[c]))
            return out
        mask = self.bg.apply(frame)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]      # drop shadow pixels (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            if cv2.contourArea(c) >= self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                out.append((x, y, x + w, y + h, "vehicle"))
        return out

    def count_zones(self, frame, boxes=None):
        boxes = self.detect(frame) if boxes is None else boxes
        h, w = frame.shape[:2]
        counts = {z: 0 for z in ZONES}
        for x1, y1, x2, y2, _ in boxes:
            counts[zone_of((x1 + x2) / 2, (y1 + y2) / 2, w, h)] += 1
        return counts, boxes

    @staticmethod
    def annotate(frame, boxes, counts):
        img = frame.copy()
        h, w = img.shape[:2]
        cv2.line(img, (0, 0), (w, h), (90, 90, 90), 1)
        cv2.line(img, (w, 0), (0, h), (90, 90, 90), 1)
        for x1, y1, x2, y2, lab in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, lab, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        txt = "  ".join(f"{k}:{v}" for k, v in counts.items())
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return img


def process_video(path, backend="auto", stride=3, max_frames=600, out_video=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    counter = VehicleCounter(backend)
    writer, rows, i = None, [], 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            counts, boxes = counter.count_zones(frame)
            rows.append({"frame": i, **counts, "total": sum(counts.values())})
            if out_video:
                img = counter.annotate(frame, boxes, counts)
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
                writer.write(img)
        i += 1
    cap.release()
    if writer:
        writer.release()
    return pd.DataFrame(rows), counter.backend


def counts_to_queues(counts):
    """Vision counts -> queue estimate vector [N, S, E, W] for the RL agent."""
    return np.array([counts[z] for z in ZONES], dtype=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--image")
    ap.add_argument("--backend", default="auto", choices=["auto", "yolo", "opencv"])
    a = ap.parse_args()
    if a.image:
        frame = cv2.imread(a.image)
        vc = VehicleCounter(a.backend)
        counts, boxes = vc.count_zones(frame)
        out = config.RESULTS_DIR / "vision_image_result.jpg"
        cv2.imwrite(str(out), vc.annotate(frame, boxes, counts))
        print(f"[{vc.backend}] counts: {counts}  -> {out}")
    elif a.video:
        out_v = config.RESULTS_DIR / "vision_annotated.avi"
        df, be = process_video(a.video, a.backend, out_video=out_v)
        df.to_csv(config.RESULTS_DIR / "vision_counts.csv", index=False)
        print(f"[{be}] processed {len(df)} sampled frames; mean vehicles per frame: "
              f"{df['total'].mean():.1f}  -> {out_v}")
    else:
        ap.error("give --video or --image")
