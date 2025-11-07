"""
Training script for agricultural produce classification models
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from src.model import create_model_for_produce
from utils.preprocessing import load_and_preprocess_images
from utils.augmentation import create_data_generator


def prepare_dataset(data_dir, produce_type, test_size=0.2):
    """Prepare dataset for training"""
    good_dir = os.path.join(data_dir, 'good')
    bad_dir = os.path.join(data_dir, 'bad')
    
    # Check if directories exist
    if not os.path.exists(good_dir) or not os.path.exists(bad_dir):
        raise ValueError(f"Data directories not found. Expected 'good' and 'bad' folders in {data_dir}")
    
    # Load images
    print("Loading good produce images...")
    good_images = load_and_preprocess_images(good_dir, produce_type)
    good_labels = np.ones(len(good_images))
    
    print("Loading bad produce images...")
    bad_images = load_and_preprocess_images(bad_dir, produce_type)
    bad_labels = np.zeros(len(bad_images))
    
    # Combine data
    X = np.concatenate([good_images, bad_images])
    y = np.concatenate([good_labels, bad_labels])
    
    # Shuffle and split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Dataset prepared:")
    print(f"  Training samples: {len(X_train)} ({np.sum(y_train)} good, {len(y_train) - np.sum(y_train)} bad)")
    print(f"  Validation samples: {len(X_val)} ({np.sum(y_val)} good, {len(y_val) - np.sum(y_val)} bad)")
    
    return X_train, X_val, y_train, y_val


def train_model(produce_type, data_dir, batch_size=40, epochs=1000):
    """Train a model for specific produce type"""
    print(f"Training model for {produce_type}...")
    
    # Prepare dataset
    X_train, X_val, y_train, y_val = prepare_dataset(data_dir, produce_type)
    
    # Create model
    classifier = create_model_for_produce(produce_type)
    classifier.build_model()
    
    print("\nModel Architecture:")
    classifier.get_model_summary()
    
    # Train model
    print("\nStarting training...")
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save model to models directory
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{produce_type}s.h5'
    classifier.model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Print final metrics
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\nFinal Training Metrics:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.4f}")
    
    return history


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train agricultural produce classifier')
    parser.add_argument(
        '--produce_type',
        type=str,
        required=True,
        choices=['mango', 'plantain'],
        help='Type of produce to train model for'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training data (should have "good" and "bad" subdirectories)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=40,
        help='Batch size for training (default: 40)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Number of epochs to train (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        args.produce_type,
        args.data_dir,
        args.batch_size,
        args.epochs
    )


if __name__ == "__main__":
    main()
