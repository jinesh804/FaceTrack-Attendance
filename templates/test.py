from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(message):
    """Text-to-speech function."""
    speaker = Dispatch(("SAPI.SpVoice"))
    speaker.Speak(message)

# Video Capture and Face Detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(r"C:\Users\jinesh jain\AppData\Roaming\Python\Python311\site-packages\cv2\data\haarcascade_frontalface_default.xml")

# Load Pickled Data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print(f"Number of labels: {len(LABELS)}")

# Ensure FACES and LABELS have the same number of samples
if len(FACES) != len(LABELS):
    min_length = min(len(FACES), len(LABELS))
    FACES = FACES[:min_length]
    LABELS = LABELS[:min_length]
    print(f"Adjusted FACES and LABELS to length: {min_length}")

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image (ensure the path is correct)
imgBackground = cv2.imread("background.png")

# Column Names for CSV
COL_NAMES = ['NAME', 'TIME']

# Ensure Attendance Directory Exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        # Get timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        file_path = f"Attendance/Attendance_15-10-202411.csv"
        exist = os.path.isfile(file_path)

        # Draw rectangles and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Prepare attendance record
        attendance = [str(output[0]), str(timestamp)]
    
    # Overlay video feed on background
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    
    # Key to take attendance
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)

        # Append or Create Attendance File
        try:
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not exist:  # If the file didn't exist before, write the header
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
        except Exception as e:
            print(f"Error occurred while writing to the CSV file: {e}")
    
    # Quit the application
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
