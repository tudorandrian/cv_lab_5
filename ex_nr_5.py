import cv2  # Import the OpenCV library for image processing
import os  # Import the OS module for file and path operations

# Define the path to the Haar Cascade classifier file
cascade_path = 'haarcascade_frontalface_default.xml'

# Check if the classifier file exists locally
if not os.path.exists(cascade_path):
    print("Fișierul Haar Cascade nu a fost găsit. Îl descărcăm...")
    import urllib.request  # Import the urllib.request module for downloading files
    haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    # Download the Haar Cascade classifier from the OpenCV GitHub repository
    urllib.request.urlretrieve(haar_url, cascade_path)
    print(f"Fișierul {cascade_path} a fost descărcat.")

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the input image
image_path = "diversity_3.jpg"  # Image to be processed
# image_path = "negative4.jpg"  # Image to be processed
image = cv2.imread(image_path)  # Read the image from the specified path

# Check if the image was loaded successfully
if image is None:
    raise ValueError("Imaginea nu a fost găsită. Verifică calea specificată.")

# Convert the loaded image to grayscale for face detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(
    gray_image,  # Input image
    scaleFactor=1.1,  # Scaling factor for the image
    minNeighbors=5,  # Minimum number of neighbors each rectangle should have to be considered a face
    minSize=(30, 30)  # Minimum size of detected faces
)

# Draw circles around the detected faces
for (x, y, w, h) in faces:
    center = (x + w // 2, y + h // 2)  # Calculate the center of the face
    radius = w // 2  # Use half the width as the radius
    cv2.circle(image, center, radius, (255, 0, 0), 3)  # Draw the circle on the image

# Display the processed image with detected faces
cv2.imshow("Detected Faces", image)  # Create a window to show the image
cv2.waitKey(0)  # Wait for any key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows

# Save the processed image to a file
cv2.imwrite("imagine_rezultat.jpg", image)  # Save the image to the specified path
print("Imaginea procesată a fost salvată ca 'imagine_rezultat.jpg'")  # Notify the user that the image has been saved
