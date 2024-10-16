import cv2
import pickle
import numpy as np
from flask import Flask, render_template, jsonify
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

app = Flask(__name__)

# Load Pickled Data (names and face encodings)
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Check if lengths of FACES and LABELS are consistent
print(f"Length of FACES: {len(FACES)}")
print(f"Length of LABELS: {len(LABELS)}")

if len(FACES) != len(LABELS):
    min_length = min(len(FACES), len(LABELS))
    FACES = FACES[:min_length]
    LABELS = LABELS[:min_length]
    print(f"Adjusted FACES and LABELS to length: {min_length}")

# Reshape FACES to match the input shape for KNN
FACES = FACES.reshape(FACES.shape[0], -1)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(r"C:\Users\jinesh jain\hh\your_project\haarcascade_frontalface_default.xml")

# Text-to-Speech function
def speak(message):
    speaker = Dispatch(("SAPI.SpVoice"))
    speaker.Speak(message)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])  
def recognize():
    ret, frame = video.read()
    if not ret:
        return jsonify({"status": "failure", "message": "Failed to capture image"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    recognized_name = None
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        # Draw a rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        recognized_name = output[0]

    if recognized_name:
        print("Recognized name:", recognized_name)  # Debug print
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        file_path = f"C:\\Users\\jinesh jain\\hh\\Attendance\\Attendance_{datetime.fromtimestamp(ts).strftime('%d-%m-%Y')}.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        attendance = [recognized_name, timestamp]

        # Write attendance to CSV file
        try:
            print(f"Writing to file: {file_path}")  # Debug print
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if os.path.getsize(file_path) == 0:
                    writer.writerow(['NAME', 'TIME'])
                writer.writerow(attendance)

            speak("Attendance Taken")
            return jsonify({"status": "success", "name": recognized_name, "timestamp": timestamp})

        except Exception as e:
            print(f"Error writing to CSV file: {e}")
            return jsonify({"status": "failure", "message": f"Error writing to file: {str(e)}"})

    return jsonify({"status": "failure", "message": "No face detected"})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        video.release()
        cv2.destroyAllWindows()
