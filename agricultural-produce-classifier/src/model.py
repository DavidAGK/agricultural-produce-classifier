"""
CNN Model Architecture for Agricultural Produce Classification
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np


class ProduceClassifier:
    """CNN model for classifying agricultural produce quality"""
    
    def __init__(self, input_shape=(184, 184, 3), produce_type='mango'):
        self.input_shape = input_shape
        self.produce_type = produce_type
        self.model = None
        
    def build_model(self):
        """Build the CNN architecture"""
        input_image = Input(self.input_shape, name='img')
        
        # Convolutional layer
        conv = Conv2D(filters=16, kernel_size=(3, 3), padding="valid")(input_image)
        conv = Activation("relu")(conv)
        conv = BatchNormalization()(conv)
        
        # Flatten and Dense layers
        flatten = Flatten()(conv)
        dense1 = Dense(100, activation='relu')(flatten)
        dense2 = Dense(20, activation='relu')(dense1)
        output = Dense(1, activation='sigmoid')(dense2)
        
        # Create model
        self.model = Model(inputs=[input_image], outputs=[output])
        
        # Compile model
        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_valid, y_valid, batch_size=40, epochs=1000):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=20, verbose=1),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.15,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.produce_type}.h5',
                verbose=1,
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_valid, y_valid)
        )
        
        return history
    
    def predict(self, image):
        """Make prediction on a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
            
        # Ensure image has correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Make prediction
        prediction = self.model.predict(image)
        return prediction[0][0]
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


def create_model_for_produce(produce_type='mango'):
    """Factory function to create model for specific produce type"""
    if produce_type.lower() == 'mango':
        return ProduceClassifier(input_shape=(128, 128, 3), produce_type='mango')
    elif produce_type.lower() == 'plantain':
        return ProduceClassifier(input_shape=(256, 256, 3), produce_type='plantain')
    else:
        raise ValueError(f"Unknown produce type: {produce_type}")
