import os
import io
import time
import base64
import uuid # For unique filenames
import traceback # Import traceback for detailed error logging
from flask import Flask, render_template_string, Response, jsonify, request
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from PIL import Image # For handling uploads

# --- Configuration ---
MODEL_PATH = "runs/ewaste-3class-v6/weights/best.pt" 
CONF_THRESHOLD = 0.40  
IOU_THRESHOLD = 0.45   
UPLOAD_FOLDER = 'uploads' 
STATIC_FOLDER = 'results' # Processed images will be saved here and served

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --- Initialize Flask App & YOLO Model ---
# Serve files from the 'results' directory under the /results URL path
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path=f'/{STATIC_FOLDER}') 
model = None
DEVICE = 'cpu'
CAMERA_INDEX = -1 
latest_frame = None 

try:
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model not found at {MODEL_PATH}")
    else:
        DEVICE = 0 if torch.cuda.is_available() else 'cpu'
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        print(f"Successfully loaded model '{MODEL_PATH}' onto device: {DEVICE}")
        
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# --- Camera Initialization ---
cap = None
def initialize_camera():
    global cap, CAMERA_INDEX
    if cap is not None and cap.isOpened(): return True 
    indices_to_try = [0, 1, 2, 3] 
    for index in indices_to_try:
        print(f"Attempting webcam index {index}...")
        temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) 
        if temp_cap.isOpened():
            cap = temp_cap
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Webcam initialized successfully on index: {index}")
            CAMERA_INDEX = index
            return True
        else: temp_cap.release() 
    print("Error: Could not open any webcam.")
    cap = None
    CAMERA_INDEX = -1
    return False

# --- Frame Processing ---
def process_frame(frame):
    if model:
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=640, device=DEVICE, verbose=False)
        return results[0].plot() 
    return frame 

def generate_frames_live():
    global latest_frame
    if not initialize_camera():
        # Yield placeholder if camera fails
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "WEBCAM ERROR", (190, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', img); frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)
    # Main loop if camera works
    while True:
        success, frame = cap.read()
        if not success: time.sleep(0.1); continue
        frame = cv2.flip(frame, 1) 
        latest_frame = frame.copy() # Store raw frame for snap
        annotated_frame = process_frame(frame) # Process for stream
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template_string(open('index.html').read())

@app.route('/live_feed')
def live_feed():
    return Response(generate_frames_live(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snap_live', methods=['POST'])
def snap_live():
    """Processes latest frame, saves it, and returns Base64 + URL."""
    global latest_frame
    if latest_frame is None: return jsonify({'error': 'No frame captured yet'}), 400
    try:
        annotated_frame = process_frame(latest_frame) 
        
        # Encode Base64 for immediate display
        _, buffer_display = cv2.imencode('.jpg', annotated_frame)
        img_str_display = base64.b64encode(buffer_display).decode('utf-8')
        
        # Save the file to the results folder
        filename = f"snap_{uuid.uuid4()}.jpg"
        save_path = os.path.join(STATIC_FOLDER, filename)
        save_success = cv2.imwrite(save_path, annotated_frame)
        if not save_success:
             raise IOError(f"Failed to save snapped image to {save_path}")
        print(f"Saved snapped frame to {save_path}")

        # URL for download link (relative to the static folder URL path)
        result_url = f"/{STATIC_FOLDER}/{filename}" 

        return jsonify({
            'image_base64': f'data:image/jpeg;base64,{img_str_display}', # For display
            'image_url': result_url # For download
            }) 
    except Exception as e:
        print(f"--- SNAP LIVE ERROR ---"); traceback.print_exc(); print(f"-----------------------")
        return jsonify({'error': 'Failed to process snapshot'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    """Processes uploaded file, saves it, and returns URL."""
    print("\n--- Received Upload Request ---") 
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    print(f"File received: {file.filename}") 
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    
    if file and model:
        try:
            print("Step 1: Reading image...")
            img_pil = Image.open(file.stream).convert('RGB')
            print("Step 2: Converting...")
            frame = np.array(img_pil); frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print("Step 3: Detecting...")
            annotated_frame = process_frame(frame)
            print("Step 4: Saving...")
            filename = f"upload_{uuid.uuid4()}.jpg" # Changed prefix
            save_path = os.path.join(STATIC_FOLDER, filename)
            save_success = cv2.imwrite(save_path, annotated_frame)
            if not save_success: raise IOError(f"Failed to save image to {save_path}")
            print(f"Step 4 Success: Saved image to {save_path}")
            result_url = f"/{STATIC_FOLDER}/{filename}" 
            print(f"Step 5: Returning result URL: {result_url}")
            return jsonify({'image_url': result_url}) # Only URL needed here
        except Exception as e:
            print(f"--- UPLOAD PROCESSING ERROR ---"); traceback.print_exc(); print(f"-----------------------------")
            return jsonify({'error': f'Failed to process uploaded image: {str(e)}'}), 500
    elif not model: return jsonify({'error': 'Model not loaded'}), 500
    else: return jsonify({'error': 'File processing failed'}), 500

# --- Other Routes ---
@app.route('/status')
def status():
    camera_is_open = cap is not None and cap.isOpened()
    return jsonify({'model_loaded': model is not None, 'cuda_available': torch.cuda.is_available(),'device': str(DEVICE), 'conf': CONF_THRESHOLD, 'iou': IOU_THRESHOLD,'camera_index': CAMERA_INDEX if camera_is_open else -1})

# Serve files directly from the static folder (results)
@app.route(f'/{STATIC_FOLDER}/<filename>')
def serve_result_image(filename):
    # Flask handles serving files from static_folder automatically if static_url_path is set
    # This route might be redundant depending on config, but safe to keep.
     try:
        return app.send_static_file(filename)
     except FileNotFoundError:
        return "File not found", 404

# --- Main Execution ---
if __name__ == '__main__':
    initialize_camera() 
    print("\n\n--- Starting Flask Web Server (Download Feature) ---")
    print(f"Access prototype at: http://127.0.0.1:5000/")
    print("Press Ctrl+C to stop.")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
