from flask import Flask, render_template, Response
from PIL import ImageDraw, Image, ImageFont
from detect import run
from datetime import datetime
import os
import cv2
import numpy as np
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)

capture_interval = 1  # secs
start_time = time.time()
save_path = os.path.join("saved_inference", f'{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}.jpg')


def get_count(count, frame, height, width):
    text = " | ".join([f"{'Boston Kreme' if key== 'Boston Cream' else key}: {value}" for key, value in count.items()])
    frame = Image.fromarray(frame)

    font = ImageFont.truetype("ariblk.ttf", 14)
    draw = ImageDraw.Draw(frame)
    draw.text((10, height-30), text, fill=(255, 255, 255), font=font)
    frame = np.asarray(frame)
    return frame


def generate_frames():
    global save_path, start_time
    while True:
        try:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame, count = run(frame)
                # Concatenate Count
                height, width, channels = frame.shape

                frame = get_count(count, frame, height, width)

                # Save results (image with detections)
                if int(time.time() - start_time) >= capture_interval:
                    cv2.imwrite(save_path, frame)
                    start_time = time.time()
                    save_path = os.path.join("saved_inference",
                                             f'{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}.jpg')

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as ex:
            continue


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
