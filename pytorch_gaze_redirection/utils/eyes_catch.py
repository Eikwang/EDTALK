import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import torch

import dlib
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def eyes_catch(hps, file_name):

    channels = 3

    # Root directory path of the project

    # Directory for training/validation data
    DATA_PATH = os.path.join("client_pictures/")

    # Directory for model data
    MODEL_PATH = os.path.join("utils/eyes_catch_model")

    MODEL_FILE_PATH = os.path.join(MODEL_PATH, "shape_predictor_68_face_landmarks.dat")

    '''----------face detection----------'''

    # Use dlib's built-in frontal_face_detector as the face detector
    face_detector = dlib.get_frontal_face_detector()

    # Iterate through each image in the folder
    valid_extensions = {".jpg", ".png"} # Supported image formats
    file_path = os.path.join(DATA_PATH, file_name)
    
    # Check if the file is an image
    if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in valid_extensions:
        try:
            # Load image file
            image = Image.open(file_path)
            
            # Convert PIL.Image object to numpy ndarray
            img = np.array(image)
            dets = face_detector(img, 0) # Since the test image is already large, we do not enable upsampling
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    '''----------crop eyes patch----------'''

    predictor = dlib.shape_predictor(MODEL_FILE_PATH) # Use the model path you have set

    picture_eyes_patch = []

    eyes_position = []

    size = []

    # Iterate through each detected face
    for i, d in enumerate(dets):
        # Use Dlib's landmark detector
        landmarks = predictor(img, d)

        # Get the landmark positions for the left and right eyes
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        # Calculate the bounding box for the eye region
        left_eye_x1 = min([p[0] for p in left_eye_points])
        left_eye_y1 = min([p[1] for p in left_eye_points])
        left_eye_x2 = max([p[0] for p in left_eye_points])
        left_eye_y2 = max([p[1] for p in left_eye_points])

        right_eye_x1 = min([p[0] for p in right_eye_points])
        right_eye_y1 = min([p[1] for p in right_eye_points])
        right_eye_x2 = max([p[0] for p in right_eye_points])
        right_eye_y2 = max([p[1] for p in right_eye_points])

        # Calculate the side length of the square, based on the longer side, and enlarge by 1.5 times
        left_eye_width = left_eye_x2 - left_eye_x1
        left_eye_height = left_eye_y2 - left_eye_y1
        left_eye_size = int(1.5 * max(left_eye_width, left_eye_height))

        right_eye_width = right_eye_x2 - right_eye_x1
        right_eye_height = right_eye_y2 - right_eye_y1
        right_eye_size = int(1.5 * max(right_eye_width, right_eye_height))

        # Update the left and right eye bounding boxes to be square, enlarged by 1.5 times, with the center unchanged
        left_eye_center_x = (left_eye_x1 + left_eye_x2) // 2
        left_eye_center_y = (left_eye_y1 + left_eye_y2) // 2
        left_eye_x1 = left_eye_center_x - left_eye_size // 2
        left_eye_x2 = left_eye_center_x + left_eye_size // 2
        left_eye_y1 = left_eye_center_y - left_eye_size // 2
        left_eye_y2 = left_eye_center_y + left_eye_size // 2

        right_eye_center_x = (right_eye_x1 + right_eye_x2) // 2
        right_eye_center_y = (right_eye_y1 + right_eye_y2) // 2
        right_eye_x1 = right_eye_center_x - right_eye_size // 2
        right_eye_x2 = right_eye_center_x + right_eye_size // 2
        right_eye_y1 = right_eye_center_y - right_eye_size // 2
        right_eye_y2 = right_eye_center_y + right_eye_size // 2

        # Assume we have already obtained the bounding box coordinates for the left and right eyes from Dlib
        left_eye = image.crop((left_eye_x1, left_eye_y1, left_eye_x2, left_eye_y2))
        right_eye = image.crop((right_eye_x1, right_eye_y1, right_eye_x2, right_eye_y2))
        size.append(left_eye_size)
        size.append(right_eye_size)

        transform = transforms.Compose([
                transforms.Resize((hps.image_size, hps.image_size)), # Resize the image
                transforms.ToTensor(), # Convert to tensor and normalize pixel values to [0, 1]
                transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels) # Normalize pixel values to [-1, 1]
            ])
        
        left_eye_tensor = transform(left_eye)
        right_eye_tensor = transform(right_eye)

        picture_eyes_patch.append(left_eye_tensor)
        picture_eyes_patch.append(right_eye_tensor)

        eyes_position.append([left_eye_x1, left_eye_y1])
        eyes_position.append([right_eye_x1, right_eye_y1])

    # Convert the list to a Tensor
    picture_eyes_patch = torch.stack(picture_eyes_patch)

    return picture_eyes_patch, eyes_position, size



