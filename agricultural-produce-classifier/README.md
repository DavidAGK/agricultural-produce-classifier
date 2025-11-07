# Agricultural Produce Classification Tool

A machine learning tool for classifying agricultural produce quality using computer vision and deep learning techniques.

## Overview

This project implements a CNN-based classification system to automatically grade agricultural produce (mangoes and plantains) as good or bad quality, helping to reduce post-harvest waste in Nigerian agriculture.

## Features

- **Real-time Classification**: Classify individual produce images or batch process entire folders
- **High Accuracy**: Achieves 99.31% accuracy in quality detection
- **User-friendly GUI**: Built with Tkinter for easy interaction
- **Transfer Learning**: Utilizes pre-trained models for efficient training
- **Multiple Produce Types**: Currently supports mangoes and plantains

## Tech Stack

- Python 3.8+
- TensorFlow 2.x / Keras
- OpenCV
- NumPy
- Tkinter (GUI)
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agricultural-produce-classifier.git
cd agricultural-produce-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Select the type of produce you want to classify (Mango or Plantain)

3. Choose an image or folder of images to classify

4. View the results showing the quality percentage

## Project Structure

```
agricultural-produce-classifier/
│
├── src/
│   ├── __init__.py
│   ├── model.py          # CNN model architecture
│   ├── train.py          # Training script
│   ├── classify.py       # Classification functions
│   └── gui.py            # Tkinter GUI implementation
│
├── models/
│   ├── mangoes.h5        # Trained model for mangoes
│   └── plantain.h5       # Trained model for plantains
│
├── data/
│   ├── train/
│   │   ├── good/
│   │   └── bad/
│   └── test/
│       ├── good/
│       └── bad/
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py  # Image preprocessing utilities
│   └── augmentation.py   # Data augmentation functions
│
├── notebooks/
│   └── experiments.ipynb # Jupyter notebooks for experiments
│
├── requirements.txt
├── README.md
└── main.py              # Entry point
```

## Model Architecture

The CNN architecture consists of:
- Convolutional layer with 16 filters (3x3 kernel)
- ReLU activation and Batch Normalization
- Flatten layer
- Dense layers (100, 20, and 1 neurons)
- Sigmoid activation for binary classification

## Training

To train the model with your own data:

```bash
python src/train.py --produce_type mango --epochs 1000 --batch_size 40
```

## Results

- **Accuracy**: 99.31%
- **Classification Speed**: < 1 second per image
- **Supported Formats**: JPG, PNG, JPEG

## Future Improvements

- Add support for more agricultural products
- Implement mobile application
- Add IoT integration for real-time field monitoring
- Expand to multi-class classification beyond binary (good/bad)

## Contributors

- Ejike David Chinedu - [GitHub Profile]([https://github.com/DavidAGK])

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Afe Babalola University, Department of Electrical Electronics and Computer Engineering
- Dr. O.O. Omitola (Project Supervisor)
