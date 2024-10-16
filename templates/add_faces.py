import cv2
import pickle
import numpy as np
import os

# Initialize video and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(r"C:\Users\jinesh jain\hh\your_project\haarcascade_frontalface_default.xml")

# Ensure 'data/' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Store face data and names
faces_data = []
names = []

while True:
    name = input("Enter Your Name (or 'exit' to finish): ")
    if name.lower() == 'exit':
        break

    i = 0  # Reset counter for each new person
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))

            if len(faces_data) < 100 and i % 10 == 0:  # Capture up to 100 samples per person
                faces_data.append(resized_img)
                names.append(name)

            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)

        if k == ord('q') or len(faces_data) == 100:
            break

# Cleanup
video.release()
cv2.destroyAllWindows()

# Reshape face data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)  # Shape it according to the number of faces captured

# Save names data
names_file = 'data/names.pkl'
faces_file = 'data/faces_data.pkl'

# Save names and faces data
if os.path.exists(names_file):
    with open(names_file, 'rb') as f:
        existing_names = pickle.load(f)
else:
    existing_names = []

if os.path.exists(faces_file):
    with open(faces_file, 'rb') as f:
        existing_faces = pickle.load(f)
else:
    existing_faces = np.empty((0, 2500))  # Adjust the shape according to the resize dimensions

# Combine existing and new names/faces
all_names = existing_names + names
all_faces = np.append(existing_faces, faces_data, axis=0)

# Save updated data
with open(names_file, 'wb') as f:
    pickle.dump(all_names, f)

with open(faces_file, 'wb') as f:
    pickle.dump(all_faces, f)

print("Data saved successfully.")
