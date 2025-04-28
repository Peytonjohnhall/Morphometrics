# Facial Averaging Program with Manual Landmarking

import fitz  # PyMuPDF
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Extract images from PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            images.append(img_np)
    return images

# Step 2: Click landmarks manually
def click_landmarks(image, num_points=8):
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    clone = image.copy()
    cv2.namedWindow("Click Landmarks")
    cv2.setMouseCallback("Click Landmarks", mouse_callback)

    while True:
        temp_img = clone.copy()
        for point in points:
            cv2.circle(temp_img, point, 3, (0, 255, 0), -1)
        cv2.imshow("Click Landmarks", temp_img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
        if len(points) >= num_points:
            break

    cv2.destroyWindow("Click Landmarks")
    return np.array(points)

# Step 3: Align landmarks

def procrustes(X, Y):
    muX = np.mean(X, 0)
    muY = np.mean(Y, 0)
    X0 = X - muX
    Y0 = Y - muY
    normX = np.linalg.norm(X0)
    normY = np.linalg.norm(Y0)
    X0 /= normX
    Y0 /= normY

    A = np.dot(X0.T, Y0)
    U, S, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    s = S.sum()

    Y_aligned = normX * s * np.dot(Y0, R) + muX
    return Y_aligned

# Step 4: Average faces

def main():
    pdf_path = "Human Side View.pdf"  # Update with your file name
    output_path = "average_face.png"

    images = extract_images_from_pdf(pdf_path)
    print(f"Extracted {len(images)} images.")

    all_landmarks = []
    for idx, img in enumerate(images):
        img_resized = cv2.resize(img, (400, 400))
        landmarks = click_landmarks(img_resized, num_points=8)
        all_landmarks.append(landmarks)

    # Reference: first image's landmarks
    reference = all_landmarks[0]

    aligned_landmarks = []
    for landmarks in all_landmarks:
        aligned = procrustes(reference, landmarks)
        aligned_landmarks.append(aligned)

    mean_landmarks = np.mean(aligned_landmarks, axis=0)

    # Draw average face (updated version)
    avg_face = np.zeros((400, 400), dtype=np.uint8)

    # Draw lines connecting points in order
    mean_landmarks_int = mean_landmarks.astype(int)
    for i in range(len(mean_landmarks_int) - 1):
        cv2.line(avg_face, tuple(mean_landmarks_int[i]), tuple(mean_landmarks_int[i + 1]), 255, 2)

    # Optionally connect last point to first (if closed shape)
    # cv2.line(avg_face, tuple(mean_landmarks_int[-1]), tuple(mean_landmarks_int[0]), 255, 2)

    plt.imshow(avg_face, cmap="gray")
    plt.axis("off")
    plt.title("Average Side View Face (Outline)")
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print(f"Saved average face to {output_path}")

if __name__ == "__main__":
    main()
