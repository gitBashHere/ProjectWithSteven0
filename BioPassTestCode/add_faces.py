import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist



#setting these paths in a way that'll be easier for me to keep track of and type as i need to use them
predictor_path = r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\FinalYear\ComputingProject\finalYearProject\BioPassTestCode\models\shape_predictor_68_face_landmarks.dat" #using this pre trained dlib library to make things a little easier
haarcascade_path = r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\FinalYear\ComputingProject\finalYearProject\BioPassSource\data\haarcascade_frontalface_default.xml" #using haarcascade

# Initialize face detection and landmark predictor
face_cascade = cv2.CascadeClassifier(haarcascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Indices for eyes in the 68-point landmark model
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Blink detection thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
blink_counter = 0
total_blinks = 0

# using the internal camera on my laptop and initialising it
video = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        print("Frame capture unsuccessful")
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (252, 237, 172), 2)

    # Detect faces using dlib for landmark prediction
    dlib_faces = detector(gray)
    for face in dlib_faces:
        # Get landmarks
        landmarks = predictor(gray, face)
        landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

        # Extract eye regions
        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if EAR is below the threshold (blink detected)
        if avg_ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
                blink_counter = 0

        # Display total blink count on the frame
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the video frame with detections and blink counts
    cv2.imshow("Liveness Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
