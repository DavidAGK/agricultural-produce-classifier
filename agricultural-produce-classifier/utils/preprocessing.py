"""
Image preprocessing utilities for agricultural produce classification
"""

import cv2
import numpy as np
import os
from typing import List, Tuple


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to specified size"""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] range"""
    return image.astype(np.float32) / 255.0


def preprocess_single_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess a single image"""
    # Load image
    img = load_image(image_path)
    
    # Resize
    img_resized = resize_image(img, target_size)
    
    # Normalize
    img_normalized = normalize_image(img_resized)
    
    return img_normalized


def load_and_preprocess_images(directory: str, produce_type: str) -> np.ndarray:
    """Load and preprocess all images from a directory"""
    # Determine target size based on produce type
    if produce_type.lower() == 'mango':
        target_size = (128, 128)
    elif produce_type.lower() == 'plantain':
        target_size = (256, 256)
    else:
        raise ValueError(f"Unknown produce type: {produce_type}")
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No valid image files found in {directory}")
    
    # Process all images
    processed_images = []
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        try:
            processed_img = preprocess_single_image(image_path, target_size)
            processed_images.append(processed_img)
        except Exception as e:
            print(f"Warning: Could not process {image_file}: {str(e)}")
            continue
    
    # Convert to numpy array
    return np.array(processed_images)


def extract_features(image: np.ndarray) -> dict:
    """Extract basic features from an image"""
    features = {}
    
    # Color features
    features['mean_r'] = np.mean(image[:, :, 0])
    features['mean_g'] = np.mean(image[:, :, 1])
    features['mean_b'] = np.mean(image[:, :, 2])
    
    features['std_r'] = np.std(image[:, :, 0])
    features['std_g'] = np.std(image[:, :, 1])
    features['std_b'] = np.std(image[:, :, 2])
    
    # Convert to HSV for additional features
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    features['mean_h'] = np.mean(hsv[:, :, 0])
    features['mean_s'] = np.mean(hsv[:, :, 1])
    features['mean_v'] = np.mean(hsv[:, :, 2])
    
    return features


def remove_background(image: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Remove white background from image using flood fill"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create mask for white pixels
    mask = gray > threshold
    
    # Apply mask
    result = image.copy()
    result[mask] = [255, 255, 255]
    
    return result


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Apply image enhancement techniques"""
    # Apply histogram equalization to improve contrast
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def create_image_batch(image_paths: List[str], produce_type: str, batch_size: int = 32):
    """Create batches of preprocessed images"""
    # Determine target size
    if produce_type.lower() == 'mango':
        target_size = (128, 128)
    elif produce_type.lower() == 'plantain':
        target_size = (256, 256)
    else:
        raise ValueError(f"Unknown produce type: {produce_type}")
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                img = preprocess_single_image(path, target_size)
                batch_images.append(img)
            except Exception as e:
                print(f"Warning: Could not process {path}: {str(e)}")
                continue
        
        if batch_images:
            yield np.array(batch_images)
