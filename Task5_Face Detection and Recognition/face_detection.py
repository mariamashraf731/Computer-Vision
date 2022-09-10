import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_faces(source: np.ndarray, scale_factor: float = 1.1, min_size: int = 50) -> list:

    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # Create the haar cascade
    repo_root = os.path.dirname(os.getcwd())
    sys.path.append(repo_root)

    cascade_path = r"src\haarcascade_frontalface_default.xml"

    # cascade_path = "../src/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        image=src,
        scaleFactor=scale_factor,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


def draw_faces(source: np.ndarray, faces: list, thickness: int = 10) -> np.ndarray:
    """
    Draw rectangle around each face in the given faces list
    :return:
    """

    src = np.copy(source)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img=src, pt1=(x, y), pt2=(x + w, y + h),
                      color=(0, 255, 0), thickness=thickness)

    return src


image = cv2.imread(r"images\faces.jfif")
# image = cv2.imread("../resources/Images/faces/IMG_8117.jpeg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detected_faces = detect_faces(source=image)
faced_image = draw_faces(source=image_rgb, faces=detected_faces)

print(f"Found {len(detected_faces)} Faces!")
print(detected_faces)

plt.imshow(faced_image)
plt.show()