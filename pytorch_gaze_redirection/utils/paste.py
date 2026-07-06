import os
import cv2
import numpy as np
from skimage.exposure import match_histograms

def paste(hps, file_name, generated_image, eyes_position, size):
    products_dir = os.path.join('products')
    os.makedirs(products_dir, exist_ok=True)

    count = 0

    picture_path = os.path.join(hps.client_pictures_dir, file_name)
    original_image = cv2.imread(picture_path)

    '''paste eyes patch to original image'''

    for ith in range(len(generated_image)):
        x = eyes_position[ith][0]
        y = eyes_position[ith][1]
        overlay_image = ((generated_image[ith].permute(1, 2, 0).numpy() + 1) * 0.5 * 255).astype(np.uint8)
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

        resized_overlay_image = cv2.resize(overlay_image, (size[ith], size[ith]), interpolation=cv2.INTER_LINEAR)

        target_area = original_image[y : y + size[ith], x : x + size[ith]]
        resized_overlay_image_matched = match_histograms(resized_overlay_image, target_area, channel_axis=-1)
        
        mask = np.zeros((size[ith], size[ith]), dtype=np.float32)
        center = (size[ith] // 2, size[ith] // 2)
        radius = size[ith] // 2
        cv2.circle(mask, center, radius, 1, thickness=-1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        mask = mask[..., np.newaxis]
        blended_patch = (resized_overlay_image_matched * mask + 
                        target_area * (1 - mask)).astype(np.uint8)

        original_image[y : y + size[ith], x : x + size[ith]] = blended_patch
        
        output_path = os.path.join(products_dir, f"processed_{file_name}")
        cv2.imwrite(output_path, original_image)
    count += 1