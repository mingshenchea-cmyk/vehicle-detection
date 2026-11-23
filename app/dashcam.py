"""Dashcam GUI: live vehicle detection with count, FPS and manual recording.

Notebook section 6.2, using the trained YOLO12s export (best.pt).

Examples:
    python app/dashcam.py                     # webcam (index 0)
    python app/dashcam.py --source video/rain1.mp4
"""
import argparse
import datetime
import os
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="0", help="video file path or camera index (default 0)")
parser.add_argument("--model", type=Path, default=config.ROOT / "best.pt")
args = parser.parse_args()

from ultralytics import YOLO

model = YOLO(str(args.model))

# Tkinter UI
root = tk.Tk()
root.geometry("1280x720")
root.configure(bg="black")

# Top Info Bar
info_bar = tk.Frame(root, bg="gray10", height=50)
info_bar.pack(fill="x", side="top")

vehicle_count_var = tk.StringVar(value="Vehicles detected: 0")
vehicle_count_label = tk.Label(info_bar, textvariable=vehicle_count_var, font=("Arial", 16), fg="white", bg="gray10")
vehicle_count_label.pack(side="left", padx=20)

fps_var = tk.StringVar(value="Measured FPS: --")
fps_label = tk.Label(info_bar, textvariable=fps_var, font=("Arial", 12), fg="white", bg="gray10")
fps_label.pack(side="left", padx=8)

record_var = tk.StringVar(value="")
record_label = tk.Label(info_bar, textvariable=record_var, font=("Arial", 16), fg="red", bg="gray10")
record_label.pack(side="right", padx=20)

# Main Video Area
video_frame = tk.Frame(root, bg="black", width=1280, height=620)
video_frame.pack_propagate(False)
video_frame.pack(side="top", fill="none")
video_label = tk.Label(video_frame, bg="black")
video_label.pack(expand=True)
video_label.configure(anchor="center")

# Bottom Info Bar
bottom_bar = tk.Frame(root, bg="gray10", height=50)
bottom_bar.pack(fill="x", side="bottom")

logo_label = tk.Label(bottom_bar, text="📸 Dashcam", font=("Arial", 14, "bold"), fg="white", bg="gray10")
logo_label.pack(side="left", padx=20)

datetime_var = tk.StringVar(value="")
datetime_label = tk.Label(bottom_bar, textvariable=datetime_var, font=("Arial", 14), fg="white", bg="gray10")
datetime_label.pack(side="right", padx=20)

# Manual Record Button
is_recording = False
out_writer = None

record_button = tk.Button(bottom_bar, text="Record", font=("Arial", 12), fg="white", bg="red", width=12)
record_button.pack(side="bottom", pady=5)

# Camera / video initialization
source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open source: {args.source}")

# FPS measurement
ts_deque = deque(maxlen=240)  # store timestamps for about 2-3 seconds
measured_fps = 0.0


def update_measured_fps(ts):
    global measured_fps
    ts_deque.append(ts)
    if len(ts_deque) >= 2:
        duration = ts_deque[-1] - ts_deque[0]
        measured_fps = len(ts_deque) / duration if duration > 0 else float(len(ts_deque))
    else:
        measured_fps = 0.0
    fps_var.set(f"Measured FPS: {measured_fps:.2f}")


def toggle_recording_cmd():
    global is_recording, out_writer
    if not is_recording:
        fps_to_use = measured_fps if measured_fps >= 1.0 else cap.get(cv2.CAP_PROP_FPS) or 20.0
        os.makedirs("recordings", exist_ok=True)
        filename = datetime.datetime.now().strftime("recordings/rec_%Y%m%d_%H%M%S.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        out_writer = cv2.VideoWriter(filename, fourcc, float(fps_to_use), (width, height))
        is_recording = True
        record_var.set("🔴 REC")
        record_button.config(text="Stop Record")
        print(f"[REC] Started {filename} @ {fps_to_use:.2f} FPS")
    else:
        is_recording = False
        record_var.set("")
        record_button.config(text="Record")
        if out_writer:
            out_writer.release()
            out_writer = None
        print("[REC] Stopped")


record_button.config(command=toggle_recording_cmd)


def update_frame():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
        root.after(5, update_frame)
        return

    update_measured_fps(time.time())

    display_frame = frame.copy()
    vehicle_count = 0
    try:
        results = model(frame, conf=0.25, iou=0.5, imgsz=480, verbose=False)
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = r.names[cls_id]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        # If YOLO fails, still proceed and show/write the raw frame
        print("YOLO error (display only):", e)
        vehicle_count = 0

    vehicle_count_var.set(f"Vehicles detected: {vehicle_count}")
    datetime_var.set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if is_recording and out_writer is not None:
        try:
            out_writer.write(display_frame)
        except Exception as e:
            print("Error writing annotated frame, attempting raw frame:", e)
            try:
                out_writer.write(frame)
            except Exception as e2:
                print("Fallback write also failed:", e2)

    # Display in Tkinter, scaled to fit the video area
    h, w = display_frame.shape[:2]
    target_w = video_label.winfo_width()
    target_h = video_label.winfo_height()
    if target_w < 10 or target_h < 10:
        target_w, target_h = 1280, 620
    scale = min(target_w / w, target_h / h)
    display_frame = cv2.resize(display_frame, (int(w * scale), int(h * scale)))

    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    video_label.image_ref = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
    video_label.configure(image=video_label.image_ref)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    root.after(int(1000 / video_fps), update_frame)


if __name__ == "__main__":
    os.makedirs("recordings", exist_ok=True)
    update_frame()
    root.mainloop()

    cap.release()
    if out_writer:
        try:
            out_writer.release()
        except Exception:
            pass
    cv2.destroyAllWindows()
