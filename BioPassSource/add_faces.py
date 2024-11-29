import cv2

# Initialize webcam (0 for internal laptop webcam)
video = cv2.VideoCapture(0)

# Load pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in a frame
def detect_face(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

while True:
    # Capture video frame-by-frame
    ret, frame = video.read()
    if not ret:
        break

    # Call the detect_face function
    faces = detect_face(frame)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (131, 115, 227), 2)  # thicker line for visibilty and aesthetic purposes, pink colour because i like pink random but interesting learning curve i hit when trying to make the square pink i learned openCV uses BGR meaning blue, green, red format not red, green, blue which is what i was using and getting purples and blues so the red and blue need to be swapped for pink leaning shades

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit loop when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video feed and destroy all windows
video.release()
cv2.destroyAllWindows()
