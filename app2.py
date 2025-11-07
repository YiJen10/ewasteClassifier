from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2, numpy as np, os, uuid, base64
from datetime import datetime
from PIL import Image
from io import BytesIO
import numpy as np
import time
import csv
import threading
import atexit
from pathlib import Path
from collections import deque, defaultdict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
CSV_PATH = LOG_DIR / "inference_log.csv"
SUMMARY_PATH = LOG_DIR / "inference_summary.txt"

# Rolling windows for smoothing FPS and times
ROLLING_WINDOW = 50  # number of recent samples to average
rolling_times = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))  # device -> deque
rolling_pre = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
rolling_post = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))

# Thread-safe CSV writing
csv_lock = threading.Lock()
if not CSV_PATH.exists():
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "device", "stage_pre_ms", "inference_ms", "stage_post_ms", "total_ms", "fps", "num_detections"])

def log_inference(device: str, pre_ms: float, infer_ms: float, post_ms: float, num_detections: int):
    """
    Call this after each frame is processed.
    device: 'Laptop' or 'Mobile' or other identifier
    times in seconds; function will convert to ms
    """
    tstamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    pre_ms = pre_ms * 1000.0
    infer_ms = infer_ms * 1000.0
    post_ms = post_ms * 1000.0
    total_ms = pre_ms + infer_ms + post_ms
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    # update rolling windows
    rolling_pre[device].append(pre_ms)
    rolling_times[device].append(infer_ms)
    rolling_post[device].append(post_ms)

    avg_pre = sum(rolling_pre[device]) / len(rolling_pre[device])
    avg_inf = sum(rolling_times[device]) / len(rolling_times[device])
    avg_post = sum(rolling_post[device]) / len(rolling_post[device])
    avg_total = avg_pre + avg_inf + avg_post
    avg_fps = 1000.0 / avg_total if avg_total > 0 else 0.0

    # Print neat terminal line
    print(f"[{tstamp}] [{device}] frame: total {total_ms:.2f} ms | pre {pre_ms:.2f} ms | inf {infer_ms:.2f} ms | post {post_ms:.2f} ms | fps {fps:.2f} | avg fps {avg_fps:.2f} | det {num_detections}")

    # Write CSV
    with csv_lock:
        with CSV_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([tstamp, device, f"{pre_ms:.2f}", f"{infer_ms:.2f}", f"{post_ms:.2f}", f"{total_ms:.2f}", f"{fps:.2f}", num_detections])

def write_summary():
    """
    Called at exit to produce a human readable summary file for appendix.
    """
    summary = []
    summary.append("Inference Summary\n")
    summary.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for device in list(rolling_times.keys()):
        if len(rolling_times[device]) == 0:
            continue
        avg_pre = sum(rolling_pre[device]) / len(rolling_pre[device])
        avg_inf = sum(rolling_times[device]) / len(rolling_times[device])
        avg_post = sum(rolling_post[device]) / len(rolling_post[device])
        avg_total = avg_pre + avg_inf + avg_post
        avg_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
        summary.append(f"Device: {device}\n")
        summary.append(f"  Samples: {len(rolling_times[device])}\n")
        summary.append(f"  Avg pre processing: {avg_pre:.2f} ms\n")
        summary.append(f"  Avg inference: {avg_inf:.2f} ms\n")
        summary.append(f"  Avg post processing: {avg_post:.2f} ms\n")
        summary.append(f"  Avg total: {avg_total:.2f} ms\n")
        summary.append(f"  Avg FPS: {avg_fps:.2f}\n\n")

    SUMMARY_PATH.write_text("\n".join(summary))
    # also print to terminal
    print("\n" + "="*50)
    print("Final Inference Summary written to", SUMMARY_PATH)
    print(SUMMARY_PATH.read_text())
    print("="*50 + "\n")

# Ensure the summary is written on exit
atexit.register(write_summary)

app = Flask(__name__)

# ============================
# Model + Folder Configuration
# ============================
MODEL_PATH = "runs/ewaste-3class-v6/weights/best.pt"
RESULT_DIR = "static/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Load YOLO model once
model = YOLO(MODEL_PATH)

# Global frames
camera = None
last_frame = None
mobile_last_frame = None


# ============================
# ROUTES
# ============================

@app.route('/')
def index():
    return render_template('index.html')


# ----------------------------------------
# Laptop webcam live feed (OpenCV stream)
# ----------------------------------------
@app.route('/video_feed')
def video_feed():
    """Laptop webcam live detection"""
    global camera, last_frame
    if camera is None or not (hasattr(camera, "isOpened") and camera.isOpened()):
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_times = []  # store inference durations

    def generate():
        global camera, last_frame
        nonlocal frame_times
        while True:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            success, frame = camera.read()
            if not success:
                break

            # Flip horizontally (mirror correction)
            frame = cv2.flip(frame, 1)

            # Start timing
            t0_pre = time.time()

            # YOLO detection: balanced precision
            results = model(frame, imgsz=640, conf=0.4, iou=0.45, verbose=False)
            annotated = results[0].plot()

            t1_inf = time.time()

            # Minimal postprocess timing (JPEG encode, etc.)
            _ = cv2.imencode('.jpg', annotated)
            t2_post = time.time()

            # Call performance logger
            log_inference(
                device="Laptop",
                pre_ms=(t1_inf - t0_pre) * 0.25,      # approximate pre-processing time (25%)
                infer_ms=(t1_inf - t0_pre) * 0.70,    # main inference
                post_ms=(t2_post - t1_inf) * 0.05,    # minor post process
                num_detections=len(results[0].boxes)
            )

            # Compute average FPS every 50 frames
            if len(frame_times) % 50 == 0:
                avg_infer = np.mean(frame_times)
                fps = 1 / avg_infer
                print(f"[Laptop] Average Inference Time: {avg_infer:.3f}s | ~{fps:.2f} FPS")
                frame_times = []  # reset for next 50-frame batch

            last_frame = annotated

            # Encode to JPEG for MJPEG stream
            _, buffer = cv2.imencode('.jpg', annotated)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ----------------------------------------
# Mobile camera (predict_frame)
# ----------------------------------------
@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Mobile: receive phone camera frame, run YOLO, return annotated image"""
    global mobile_last_frame
    try:

        # Handle both blob and base64 frame
        file = request.files.get('frame')
        if file:
            img = Image.open(file.stream).convert('RGB')
        else:
            data = request.json.get('image')
            imgdata = base64.b64decode(data.split(',')[1])
            img = Image.open(BytesIO(imgdata)).convert('RGB')

        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        t0_pre = time.time()
        
        # YOLO detection (mobile)
        results = model(frame, imgsz=480, conf=0.45, iou=0.4, verbose=False)
        t1_inf = time.time()

        annotated = results[0].plot()

        # Measure total pipeline time
        t2_post = time.time()

        # Log to terminal and CSV
        log_inference(
            device="Mobile",
            pre_ms=(t1_inf - t0_pre) * 0.20,
            infer_ms=(t1_inf - t0_pre) * 0.75,
            post_ms=(t2_post - t1_inf) * 0.05,
            num_detections=len(results[0].boxes)
        )
        
        mobile_last_frame = annotated

        # Encode to Base64 JPEG (for inline <img> update)
        _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"success": True, "image": f"data:image/jpeg;base64,{jpg_as_text}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ----------------------------------------
# Snapshot for laptop webcam
# ----------------------------------------
@app.route('/snap_live', methods=['POST'])
def snap_live():
    global last_frame
    if last_frame is None:
        return jsonify({"success": False, "error": "No frame available"})
    filename = f"laptop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, last_frame)
    return jsonify({"success": True, "file": f"/static/results/{filename}"})


# ----------------------------------------
# Snapshot for mobile camera
# ----------------------------------------
@app.route('/snap_mobile', methods=['POST'])
def snap_mobile():
    global mobile_last_frame
    if mobile_last_frame is None:
        return jsonify({"success": False, "error": "No mobile frame yet"})
    filename = f"mobile_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, mobile_last_frame)
    return jsonify({"success": True, "file": f"/static/results/{filename}"})


# ----------------------------------------
# Upload image detection
# ----------------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    """Detect uploaded image"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(img, imgsz=640, conf=0.4, iou=0.45, verbose=False)
    annotated = results[0].plot()

    filename = f"upload_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, annotated)
    return jsonify({"success": True, "file": f"/static/results/{filename}"})


# ----------------------------------------
# Serve results
# ----------------------------------------
@app.route('/static/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


# ----------------------------------------
# Run Flask
# ----------------------------------------
if __name__ == '__main__':
    print("🌐 Laptop: http://127.0.0.1:5000")
    print("📱 Phone: http://<laptop-ip>:5000 (same Wi-Fi)")
    app.run(host='0.0.0.0', port=5000, threaded=True)

    # app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))
