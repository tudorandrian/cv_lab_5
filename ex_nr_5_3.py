import face_recognition
import imutils
import pickle
import time
import cv2
import os
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

# Load the known faces and embeddings from the serialized file
data = pickle.loads(open('face_enc', "rb").read())

# Find the path to the image you want to analyze and read the image
image = cv2.imread('diversity_3.jpg')  # Replace 'Path-to-img' with the actual path

# Convert the image from BGR to RGB for the face_recognition library
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale for Haarcascade detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image using Haarcascade
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,  # Scale factor for image resizing
    minNeighbors=5,   # Minimum neighbors for bounding box grouping
    minSize=(60, 60), # Minimum size of the detected face
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Compute facial embeddings for the detected faces
encodings = face_recognition.face_encodings(rgb)

# Initialize a list to hold recognized face names
names = []

# Loop over each facial embedding
for encoding in encodings:
    # Compare the current encoding with known encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)

    # Default to "Unknown" if no match is found
    name = "Unknown"

    # If a match is found
    if True in matches:
        # Get the indexes of matches
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]

        # Dictionary to count occurrences of each name
        counts = {}

        # Loop over the matched indexes
        for i in matchedIdxs:
            # Retrieve the name corresponding to the index
            name = data["names"][i]

            # Update the count for the name
            counts[name] = counts.get(name, 0) + 1

        # Determine the name with the highest count
        name = max(counts, key=counts.get)

    # Add the recognized name to the list
    names.append(name)

# Draw rectangles and names around detected faces
for ((x, y, w, h), name) in zip(faces, names):
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Annotate the face with the recognized name
    cv2.putText(
        image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Display the final image with annotated faces
cv2.imshow("Frame", image)
cv2.waitKey(0)
