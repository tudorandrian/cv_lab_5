from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Get paths of all image files in the folder named 'Images'
# Each subfolder within 'Images' is expected to contain images of a specific person
imagePaths = list(paths.list_images('Images'))

# Initialize lists to store face encodings and corresponding names
knownEncodings = []
knownNames = []

# Loop over each image path
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person's name from the folder structure (parent directory of the image file)
    name = imagePath.split(os.path.sep)[-2]

    # Load the image using OpenCV and convert it from BGR color format (default in OpenCV) to RGB (used by face_recognition)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the face locations in the image using the 'hog' model for face detection
    boxes = face_recognition.face_locations(rgb, model='hog')

    # Compute the facial embeddings (128-dimensional feature vectors) for each detected face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over each encoding and associate it with the person's name
    for encoding in encodings:
        knownEncodings.append(encoding)  # Add the face encoding to the list
        knownNames.append(name)  # Add the corresponding name to the list

# Create a dictionary to store the face encodings and their associated names
data = {"encodings": knownEncodings, "names": knownNames}

# Save the dictionary to a file using pickle for later use
with open("face_enc", "wb") as f:
    f.write(pickle.dumps(data))
