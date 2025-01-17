import cv2
import numpy as np

# Încarcă modelul pre-antrenat
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Încarcă imaginea și redimensionează
image_path = "diversity_3.jpg"  # Înlocuiește cu calea către imaginea ta
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
resized_image = cv2.resize(image, (300, 300))
blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Detectează fețele
net.setInput(blob)
detections = net.forward()

# Parcurge detecțiile și încercuiește fețele
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Prag de încredere
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
        radius = (endX - startX) // 2
        cv2.circle(image, center, radius, (0, 255, 0), 2)

# Afișează și salvează imaginea rezultată
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("imagine_rezultat.jpg", image)
