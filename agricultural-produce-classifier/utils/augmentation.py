"""
Data augmentation utilities for agricultural produce classification
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


def create_data_generator(produce_type: str = 'mango'):
    """Create a data generator with augmentation for training"""
    
    # Define augmentation parameters based on produce type
    if produce_type.lower() == 'mango':
        # Mangoes can have more rotation since they're roughly spherical
        rotation_range = 30
        width_shift_range = 0.2
        height_shift_range = 0.2
        zoom_range = 0.2
    elif produce_type.lower() == 'plantain':
        # Plantains are elongated, so less rotation
        rotation_range = 15
        width_shift_range = 0.15
        height_shift_range = 0.15
        zoom_range = 0.15
    else:
        # Default parameters
        rotation_range = 20
        width_shift_range = 0.2
        height_shift_range = 0.2
        zoom_range = 0.2
    
    # Create generator
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=0.2,
        zoom_range=zoom_range,
        horizontal_flip=True,
        vertical_flip=False,  # Usually don't flip produce vertically
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],  # Adjust brightness
        channel_shift_range=0.1  # Slight color variations
    )
    
    return datagen


def augment_image(image: np.ndarray, augmentation_params: dict = None) -> np.ndarray:
    """Apply single augmentation to an image"""
    
    if augmentation_params is None:
        augmentation_params = {
            'rotation': np.random.uniform(-20, 20),
            'shift_x': np.random.uniform(-0.1, 0.1),
            'shift_y': np.random.uniform(-0.1, 0.1),
            'zoom': np.random.uniform(0.9, 1.1),
            'brightness': np.random.uniform(0.8, 1.2),
            'flip_horizontal': np.random.choice([True, False])
        }
    
    h, w = image.shape[:2]
    augmented = image.copy()
    
    # Apply rotation
    if 'rotation' in augmentation_params:
        angle = augmentation_params['rotation']
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h))
    
    # Apply translation
    if 'shift_x' in augmentation_params or 'shift_y' in augmentation_params:
        shift_x = augmentation_params.get('shift_x', 0) * w
        shift_y = augmentation_params.get('shift_y', 0) * h
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented = cv2.warpAffine(augmented, translation_matrix, (w, h))
    
    # Apply zoom
    if 'zoom' in augmentation_params:
        zoom = augmentation_params['zoom']
        center_x, center_y = w // 2, h // 2
        new_w, new_h = int(w * zoom), int(h * zoom)
        
        # Resize image
        resized = cv2.resize(augmented, (new_w, new_h))
        
        # Crop or pad to original size
        if zoom > 1:
            # Crop
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            augmented = resized[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad
            augmented = np.zeros_like(image)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            augmented[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    
    # Apply brightness adjustment
    if 'brightness' in augmentation_params:
        brightness = augmentation_params['brightness']
        augmented = np.clip(augmented * brightness, 0, 1)
    
    # Apply horizontal flip
    if augmentation_params.get('flip_horizontal', False):
        augmented = cv2.flip(augmented, 1)
    
    return augmented


def create_augmented_batch(images: np.ndarray, labels: np.ndarray, 
                         augmentations_per_image: int = 5) -> tuple:
    """Create augmented versions of images and labels"""
    
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        # Add original image
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Add augmented versions
        for _ in range(augmentations_per_image):
            aug_img = augment_image(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)


def apply_random_noise(image: np.ndarray, noise_type: str = 'gaussian') -> np.ndarray:
    """Add random noise to image for robustness"""
    
    if noise_type == 'gaussian':
        # Add Gaussian noise
        mean = 0
        sigma = 0.01
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        
    elif noise_type == 'salt_pepper':
        # Add salt and pepper noise
        amount = 0.01
        noisy_image = image.copy()
        
        # Salt noise
        salt_coords = [np.random.randint(0, i - 1, int(amount * image.size / 2))
                      for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1], :] = 1
        
        # Pepper noise
        pepper_coords = [np.random.randint(0, i - 1, int(amount * image.size / 2))
                        for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
        
    else:
        noisy_image = image
    
    return noisy_image


def color_jitter(image: np.ndarray, hue_shift: float = 0.1, 
                saturation_shift: float = 0.2, value_shift: float = 0.2) -> np.ndarray:
    """Apply color jittering to simulate different lighting conditions"""
    
    # Convert to HSV
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Apply shifts
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180  # Hue is in [0, 180]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation_shift), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + value_shift), 0, 255)
    
    # Convert back to RGB
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return rgb.astype(np.float32) / 255.0


def create_validation_generator():
    """Create a data generator for validation (no augmentation)"""
    return ImageDataGenerator()  # No augmentation for validation
