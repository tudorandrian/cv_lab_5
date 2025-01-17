import face_recognition  # Library for face recognition and manipulation
import imutils  # For resizing, rotating, and other image operations
import pickle  # For serializing and deserializing Python objects
import time  # For handling time-related operations
import cv2  # OpenCV library for image and video processing
import os  # For interacting with the operating system
import urllib.request

# URL to download the Haar Cascade XML file
url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# Define the path where the file will be saved
cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_alt2.xml")

# Check if the file already exists to avoid re-downloading
if not os.path.exists(cascade_path):
    try:
        print("Downloading Haar Cascade file...")
        urllib.request.urlretrieve(url, cascade_path)
        print(f"Haar Cascade file downloaded and saved at: {cascade_path}")
    except Exception as e:
        print(f"Failed to download Haar Cascade file: {e}")
else:
    print(f"Haar Cascade file already exists at: {cascade_path}")

# Initialize the CascadeClassifier with the downloaded file
faceCascade = cv2.CascadeClassifier(cascade_path)

# Check if the file loaded correctly
if faceCascade.empty():
    print("Failed to load the Haar Cascade file. Please check the file or path.")
else:
    print("Haar Cascade file loaded successfully.")

# Load the serialized facial embeddings and associated names
data = pickle.loads(open('face_enc', "rb").read())

print("Streaming started")
video_capture = cv2.VideoCapture(0)  # Initialize webcam feed

# Continuously process the video feed
while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for Haar Cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using Haar Cascade
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Scale factor for face detection
        minNeighbors=5,  # Minimum neighbors for detecting a face
        minSize=(60, 60),  # Minimum size for detected faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Convert the frame from BGR to RGB for face recognition processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get facial embeddings for any detected faces
    encodings = face_recognition.face_encodings(rgb)

    # Initialize a list to store names of recognized faces
    names = []

    # Iterate over each facial encoding detected in the frame
    for encoding in encodings:
        # Compare the detected encoding with known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)

        # Default to "Unknown" if no match is found
        name = "Unknown"

        # If there are matches, identify the most frequently matched name
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Count occurrences of each name in the matched indexes
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the name with the highest count
            name = max(counts, key=counts.get)

        # Add the identified name to the list of names
        names.append(name)

    # Draw rectangles around detected faces and display names
    for ((x, y, w, h), name) in zip(faces, names):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the recognized name below the rectangle
        cv2.putText(
            frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
        )

    # Show the processed video feed in a window
    cv2.imshow("Frame", frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()