from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import tensorflow as tf
import numpy as np
import pygame.mixer
import threading
from twilio.rest import Client
import time

app = Flask(__name__)

# Load the pre-trained Keras model
model = tf.keras.models.load_model('inceptionv3.h5')

# Set up Twilio credentials
account_sid = 'ACb75a5ddc6ce2d4537c93240dcfca25d1'
auth_token = 'ea05fdf8156731f86e9518c628ae4647'
twilio_phone_number = '+15856327203'
emergency_phone_number = '+918318693200'
client = Client(account_sid, auth_token)

# Global variables
alarm_on = False
beeping_start_time = None

# Define a function to preprocess each frame before passing it through the model
def preprocess(frame):
    processed_frame = cv2.resize(frame, (80, 80))
    processed_frame = np.expand_dims(processed_frame, axis=0)
    processed_frame = processed_frame / 255.0
    return processed_frame

# Define a function to detect drowsiness in a given frame
def detect_drowsiness(frame):
    global alarm_on, beeping_start_time

    # Preprocess the frame and pass it through the model
    processed_frame = preprocess(frame)
    prediction = model.predict(processed_frame)

    # Check if the person is drowsy
    if prediction[0][0] > prediction[0][1]:
        if not alarm_on:
            pygame.mixer.music.load('alarm.wav')
            pygame.mixer.music.play(loops=-1)
            alarm_on = True
            beeping_start_time = time.time()
        else:
            if time.time() - beeping_start_time > 60:
                # Send SMS message using Twilio
                message = client.messages.create(
                    body='Drowsiness detected! Please check on the person.',
                    from_=twilio_phone_number,
                    to=emergency_phone_number
                )
    else:
        if alarm_on:
            pygame.mixer.music.stop()
            alarm_on = False

    return frame

# Define a function to generate frames from the webcam
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = detect_drowsiness(frame)

        # Encode the frame as JPEG and yield it as a multipart response
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the start route
@app.route('/start')
def start():
    return render_template('start.html')

# Define the video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Define the turnOffAlarm route
@app.route('/turnOffAlarm', methods=['POST'])
def turn_off_alarm():
    global alarm_on
    if alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False
    return 'Alarm turned off successfully'

if __name__ == '__main__':
    pygame.mixer.init()
    app.run(debug=True)
