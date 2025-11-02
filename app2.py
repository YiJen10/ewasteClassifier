from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2, numpy as np, os, uuid, base64
from datetime import datetime
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = YOLO("runs/ewaste-3class-v6/weights/best.pt")

RESULT_DIR = "static/results"
os.makedirs(RESULT_DIR, exist_ok=True)

camera = None
last_frame = None
mobile_last_frame = None

# ---- ROUTES ----
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Laptop webcam live detection"""
    global camera, last_frame
    if camera is None or not (hasattr(camera, "isOpened") and camera.isOpened()):
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def generate():
        global camera, last_frame        #  👈  add this line
        while True:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            success, frame = camera.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            results = model(frame, imgsz=640, conf=0.5, verbose=False)
            annotated = results[0].plot()
            last_frame = annotated
            _, buffer = cv2.imencode('.jpg', annotated)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Mobile: receive phone camera frame (base64 or blob)"""
    global last_frame, mobile_last_frame
    try:
        file = request.files.get('frame')
        if file:
            img = Image.open(file.stream).convert('RGB')
        else:
            data = request.json.get('image')
            imgdata = base64.b64decode(data.split(',')[1])
            img = Image.open(BytesIO(imgdata)).convert('RGB')

        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model(frame, imgsz=640, conf=0.5, verbose=False)
        annotated = results[0].plot()
        # last_frame = annotated
        mobile_last_frame = annotated  # save last phone frame
        _, buffer = cv2.imencode('.jpg', annotated)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"success": True, "image": f"data:image/jpeg;base64,{jpg_as_text}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/snap_live', methods=['POST'])
def snap_live():
    """Save latest frame"""
    global last_frame
    if last_frame is None:
        return jsonify({"success": False, "error": "No frame available"})
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, last_frame)
    return jsonify({"success": True, "file": f"/static/results/{filename}"})

@app.route('/snap_mobile', methods=['POST'])
def snap_mobile():
    global mobile_last_frame
    if mobile_last_frame is None:
        return jsonify({"success": False, "error": "No mobile frame yet"})
    filename = f"mobile_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, mobile_last_frame)
    return jsonify({"success": True, "file": f"/static/results/{filename}"})

@app.route('/upload', methods=['POST'])
def upload():
    """Detect uploaded image"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img, imgsz=640, conf=0.5, verbose=False)
    annotated = results[0].plot()
    filename = f"upload_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, annotated)
    return jsonify({"success": True, "file": f"/static/results/{filename}"})


@app.route('/static/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


if __name__ == '__main__':
    print("🌐 Access on laptop: http://127.0.0.1:5000")
    print("📱 Access on phone: http://<laptop-ip>:5000 (same Wi-Fi)")
    app.run(host='0.0.0.0', port=5000, threaded=True)

    # app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))
