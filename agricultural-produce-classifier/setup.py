#!/usr/bin/env python
"""
Setup script for Agricultural Produce Classifier
This script creates the necessary directory structure for the project
"""

import os

def create_directories():
    """Create the necessary directory structure"""
    directories = [
        'data/train/good',
        'data/train/bad',
        'data/test/good',
        'data/test/bad',
        'models',
        'logs',
        'output'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
        
        # Create a placeholder file in each directory
        placeholder_path = os.path.join(directory, '.gitkeep')
        with open(placeholder_path, 'w') as f:
            f.write("# This file ensures the directory is tracked by git\n")
    
    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("1. Place your training images in data/train/good and data/train/bad")
    print("2. Place your test images in data/test/good and data/test/bad")
    print("3. Run: python src/train.py --produce_type mango --data_dir data/train")
    print("4. Run: python main.py to start the GUI application")

if __name__ == "__main__":
    create_directories()
