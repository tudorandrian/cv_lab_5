import cv2
import os
import requests

# Încarcă imaginea
image_path = "diversity_3.jpg"  # Înlocuiește cu calea către imaginea ta
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectează calea implicită pentru clasificatoarele Haar
haarcascade_dir = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascades")
face_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_dir, "haarcascade_frontalface_default.xml"))

# Verificare încărcare clasificator
if face_cascade.empty():
    # URL-ul fișierului Haar Cascade de pe GitHub
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

    # Locația unde va fi salvat fișierul local
    save_path = "haarcascade_frontalface_default.xml"

    # Descărcare fișier
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifică dacă descărcarea a avut succes
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Fișierul a fost descărcat cu succes și salvat ca: {save_path}")
    except Exception as e:
        print(f"Eroare la descărcarea fișierului: {e}")

    print("Eroare: clasificatorul Haar nu a fost încărcat corect. Verifică calea fișierului!")
    exit()

# Detectează fețele
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Încercuiește fețele detectate
for (x, y, w, h) in faces:
    center = (x + w // 2, y + h // 2)
    radius = w // 2
    cv2.circle(image, center, radius, (255, 0, 0), 3)

# Afișează imaginea rezultată
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvează imaginea rezultată
cv2.imwrite("imagine_rezultat.jpg", image)
