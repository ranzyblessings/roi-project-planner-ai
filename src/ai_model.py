import logging

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModel:
    """Trains a TensorFlow model to learn project selection."""

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.built = False  # Track if the model is built

    def build_model(self):
        """Creates a simple neural network model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),  # Explicit input layer
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")  # Output: probability of selecting project
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.built = True
        logger.info("Model built successfully.")

    def train_model(self, data, labels, epochs=50):
        """Trains the model using project data."""
        if not self.built:
            self.build_model()

        try:
            # Normalize data
            data = np.array(data, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
            data = self.scaler.fit_transform(data)

            # Handle class imbalance (avoid division by zero)
            if np.sum(labels == 1) > 0:
                class_weight = {0: 1.0, 1: np.sum(labels == 0) / np.sum(labels == 1)}
            else:
                class_weight = {0: 1.0, 1: 1.0}  # Avoid error if only one class is present

            self.model.fit(data, labels, epochs=epochs, verbose=1, class_weight=class_weight)
            logger.info("Model training complete.")
        except Exception as e:
            logger.error(f"Error training model: {e}")

    def predict(self, data):
        """Predicts which projects should be selected."""
        if not self.built:
            logger.error("Model is not built yet.")
            return None

        try:
            data = np.array(data, dtype=np.float32)
            data = self.scaler.transform(data)
            predictions = self.model.predict(data)
            selection = (predictions > 0.5).astype(int)  # Convert probabilities to binary selection
            logger.info(f"Predictions: {selection.flatten().tolist()}")
            return selection
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
