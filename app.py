from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import os
from emotion_detection import EmotionDetector

app = Flask(__name__)

# Initialize emotion detector
emotion_detector = EmotionDetector()

# Global variables for camera and emotion history
camera = None
emotion_history = []
camera_active = False

def get_camera():
    global camera
    if camera is None and camera_active:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    global camera_active
    while camera_active:
        camera = get_camera()
        if camera is None:
            break
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Detect emotion
            emotion = emotion_detector.detect_emotion(pil_image)
            
            # Add timestamp and emotion to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            emotion_history.append({
                'timestamp': timestamp,
                'emotion': emotion
            })
            
            # Keep only last 10 emotions
            if len(emotion_history) > 10:
                emotion_history.pop(0)
            
            # Draw emotion on frame
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Wait for 5 seconds before next capture
            import time
            time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if camera_active:
        return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Return a placeholder image or empty response
        return Response(status=204)

@app.route('/get_emotion_history')
def get_emotion_history():
    return jsonify(emotion_history)

@app.route('/capture')
def capture():
    if not camera_active:
        return jsonify({'success': False, 'message': 'Camera is not active'})
    
    camera = get_camera()
    if camera is None:
        return jsonify({'success': False, 'message': 'Camera not available'})
    
    success, frame = camera.read()
    if success:
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Detect emotion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        emotion = emotion_detector.detect_emotion(pil_image)
        
        return jsonify({
            'success': True,
            'image': frame_base64,
            'emotion': emotion,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify({'success': False, 'message': 'Failed to capture image'})

@app.route('/camera/status')
def get_camera_status():
    return jsonify({'active': camera_active})

@app.route('/camera/on')
def turn_camera_on():
    global camera_active
    camera_active = True
    return jsonify({'success': True, 'active': camera_active})

@app.route('/camera/off')
def turn_camera_off():
    global camera_active
    camera_active = False
    release_camera()
    return jsonify({'success': True, 'active': camera_active})

if __name__ == '__main__':
    
    app.run(debug=True) 