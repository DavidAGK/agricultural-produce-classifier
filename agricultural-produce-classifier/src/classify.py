"""
Classification functions for agricultural produce
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import sys


class ImageClassifier:
    """Handles image classification for agricultural produce"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
            
        model_path = os.path.join(base_path, "models")
        
        # Load mango model if exists
        mango_model_path = os.path.join(model_path, "mangoes.h5")
        if os.path.exists(mango_model_path):
            self.models['mango'] = tf.keras.models.load_model(mango_model_path)
            
        # Load plantain model if exists
        plantain_model_path = os.path.join(model_path, "plantain.h5")
        if os.path.exists(plantain_model_path):
            self.models['plantain'] = tf.keras.models.load_model(plantain_model_path)
    
    def preprocess_image(self, image_path, produce_type):
        """Preprocess image for classification"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Resize based on produce type
        if produce_type.lower() == 'mango':
            img_resized = cv2.resize(img, (128, 128))
        elif produce_type.lower() == 'plantain':
            img_resized = cv2.resize(img, (256, 256))
        else:
            raise ValueError(f"Unknown produce type: {produce_type}")
            
        # Normalize pixel values
        img_normalized = np.array(img_resized / 255.0)
        
        # Add batch dimension
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        return img, img_expanded
    
    def classify_image(self, image_path, produce_type):
        """Classify a single image"""
        # Check if model exists for produce type
        if produce_type.lower() not in self.models:
            raise ValueError(f"No model loaded for produce type: {produce_type}")
            
        # Preprocess image
        original_img, processed_img = self.preprocess_image(image_path, produce_type)
        
        # Get model
        model = self.models[produce_type.lower()]
        
        # Make prediction
        prediction = model.predict(processed_img)
        confidence = prediction[0][0]
        
        # Determine quality
        quality_percentage = confidence * 100
        
        if confidence >= 0.6:
            quality = "Good"
            color = (0, 255, 0)  # Green
        elif confidence >= 0.5:
            quality = "Fair"
            color = (0, 165, 255)  # Orange
        else:
            quality = "Bad"
            color = (0, 0, 255)  # Red
            
        result = {
            'quality': quality,
            'percentage': quality_percentage,
            'confidence': confidence,
            'color': color
        }
        
        return result, original_img
    
    def classify_batch(self, folder_path, produce_type):
        """Classify all images in a folder"""
        results = []
        
        # Get all image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            try:
                result, _ = self.classify_image(image_path, produce_type)
                results.append({
                    'filename': image_file,
                    'quality': result['quality'],
                    'percentage': result['percentage']
                })
            except Exception as e:
                results.append({
                    'filename': image_file,
                    'quality': 'Error',
                    'percentage': 0,
                    'error': str(e)
                })
                
        return results
    
    def visualize_result(self, image, result):
        """Add classification result to image"""
        # Resize image for display
        display_img = cv2.resize(image, (800, 500))
        
        # Add text
        text = f"{result['quality']} Produce, {result['percentage']:.1f}% good"
        cv2.putText(
            display_img,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            result['color'],
            2
        )
        
        return display_img


def calculate_batch_statistics(results):
    """Calculate statistics for batch classification results"""
    total = len(results)
    good = sum(1 for r in results if r['quality'] == 'Good')
    fair = sum(1 for r in results if r['quality'] == 'Fair')
    bad = sum(1 for r in results if r['quality'] == 'Bad')
    errors = sum(1 for r in results if r['quality'] == 'Error')
    
    stats = {
        'total': total,
        'good': good,
        'fair': fair,
        'bad': bad,
        'errors': errors,
        'good_percentage': (good / total * 100) if total > 0 else 0,
        'fair_percentage': (fair / total * 100) if total > 0 else 0,
        'bad_percentage': (bad / total * 100) if total > 0 else 0
    }
    
    return stats
