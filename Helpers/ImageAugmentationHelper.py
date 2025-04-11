import os
import cv2
import uuid
import random

class ImageAugmentationHelper:
    def __init__(self, dataset_path="augmented_images_dataset", num_variations=4):
        self.dataset_path = dataset_path
        self.num_variations = num_variations
        os.makedirs(self.dataset_path, exist_ok=True)
    
    def augment_image(self, image_path, query):
        batch_key = str(uuid.uuid4())
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}")
            return []
        
        augmented_images = []
        for _ in range(self.num_variations):
            angle = random.choice([0, 90, 180, 270])
            alpha = round(random.uniform(0.6, 1.5), 2)
            
            rotated = self._rotate_image(image, angle)
            adjusted = cv2.convertScaleAbs(rotated, alpha=alpha, beta=0)
            img_filename = os.path.join(self.dataset_path, f"{batch_key}/{uuid.uuid4()}.jpg")
            cv2.imwrite(img_filename, adjusted)
            
            augmented_images.append({
                "filepath": img_filename,
                "batch_key": batch_key
            })
        
        return augmented_images

    def _rotate_image(self, image, angle):
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
