from PIL import Image
import numpy as np
import dlib
import os
import shutil

# Adjust threshold for eye openness ratio
eyes_threshold = 0.12

# Initialize Dlib face detector and 68-point predictor
face_detector = dlib.get_frontal_face_detector()
MODEL_PATH = os.path.join("utils/eyes_catch_model")
MODEL_FILE_PATH = os.path.join(MODEL_PATH, "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(MODEL_FILE_PATH)

# Define folder paths for classification
pictures_dir = "./client_pictures"
base_dir = "./log/closed_open_classification"
open_dir = os.path.join(base_dir, "open")
closed_dir = os.path.join(base_dir, "closed")
faces_dir = os.path.join(base_dir, "closed_faces") # Folder to save faces

# Create classification folder if it doesn't exist
os.makedirs(open_dir, exist_ok=True)
os.makedirs(closed_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)

# Function to detect if eyes are open
def is_eye_open(landmarks, left_eye_indices, right_eye_indices):
    def calculate_eye_aspect_ratio(eye_points):
        # EAR calculation formula: ((p2 - p6) + (p3 - p5)) / (2.0 * (p1 - p4))
        A = ((eye_points[1].y - eye_points[5].y) ** 2 + (eye_points[1].x - eye_points[5].x) ** 2) ** 0.5
        B = ((eye_points[2].y - eye_points[4].y) ** 2 + (eye_points[2].x - eye_points[4].x) ** 2) ** 0.5
        C = ((eye_points[0].y - eye_points[3].y) ** 2 + (eye_points[0].x - eye_points[3].x) ** 2) ** 0.5
        return (A + B) / (2.0 * C)

    left_eye = [landmarks.part(idx) for idx in left_eye_indices]
    right_eye = [landmarks.part(idx) for idx in right_eye_indices]

    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)
    print(f"* Face {idx + 1} -> Left eye openness ratio: {left_ear:.3f}, Right eye openness ratio: {right_ear:.3f}")

    # EAR threshold (0.18 is typically suitable for detecting closed eyes)
    return left_ear > eyes_threshold and right_ear > eyes_threshold

# Iterate through all images in the directory
for file_name in os.listdir(pictures_dir):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(pictures_dir, file_name)

        # Use PIL to read the image and convert it to RGB format
        image = Image.open(file_path).convert("RGB")
        img = np.array(image)

        # Ensure the image type is uint8
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)

        # Dlib face detection
        faces = face_detector(img, 0)
        print(f"{file_name}: Detected {len(faces)} face(s)")
        
        all_open = True

        # Check if each face has open eyes
        for idx, face in enumerate(faces):
        # Save the detected face images

            # Pass NumPy image and rectangular region to predictor
            landmarks = predictor(img, face)
            left_eye_indices = [36, 37, 38, 39, 40, 41]
            right_eye_indices = [42, 43, 44, 45, 46, 47]

            eye_open = is_eye_open(landmarks, left_eye_indices, right_eye_indices)

            if not eye_open:
                face_image = image.crop((face.left(), face.top(), face.right(), face.bottom()))
                face_file_name = f"{os.path.splitext(file_name)[0]}_face_{idx + 1}.jpg"
                face_file_path = os.path.join(faces_dir, face_file_name)
                face_image.save(face_file_path)
                all_open = False
                break
        
        if len(faces)==0:
            all_open = False

        # Classify the image based on the results
        if all_open:
            shutil.copy2(file_path, os.path.join(open_dir, file_name))
            print(f"Image {file_name} has been classified into the 'open' folder")
        else:
            shutil.move(file_path, os.path.join(closed_dir, file_name))
            print(f"Image {file_name} has been classified into the 'closed' folder")
    print("=============")
print("Classification complete, and closed-eye faces have been saved!")