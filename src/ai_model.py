import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for model architecture and training parameters."""
    INPUT_DIM = 2
    HIDDEN_LAYERS = [16, 8]
    OUTPUT_DIM = 1
    ACTIVATION_HIDDEN = "relu"
    ACTIVATION_OUTPUT = "sigmoid"
    OPTIMIZER = "adam"
    LOSS = "binary_crossentropy"
    METRICS = ["accuracy"]
    DEFAULT_EPOCHS = 50


class AIModel:
    """
    A robust and extensible TensorFlow-based model for project selection prediction.

    Attributes:
        scaler: MinMaxScaler instance for data normalization.
        model: Compiled TensorFlow Sequential model (lazy-initialized).
    """

    def __init__(self) -> None:
        """Initialize the model with a scaler and unbuilt state."""
        self.scaler = MinMaxScaler()
        self._model: Optional[tf.keras.Model] = None
        self._is_built = False

    @property
    def model(self) -> tf.keras.Model:
        """Lazy-load and return the model, building it if necessary."""
        if not self._is_built:
            self._build_model()
        assert self._model is not None, "Model should be built but is None."
        return self._model

    def _build_model(self) -> None:
        """Construct and compile the neural network architecture."""
        try:
            self._model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(ModelConfig.INPUT_DIM,)),
                *[tf.keras.layers.Dense(units, activation=ModelConfig.ACTIVATION_HIDDEN)
                  for units in ModelConfig.HIDDEN_LAYERS],
                tf.keras.layers.Dense(ModelConfig.OUTPUT_DIM, activation=ModelConfig.ACTIVATION_OUTPUT)
            ])
            self._model.compile(
                optimizer=ModelConfig.OPTIMIZER,
                loss=ModelConfig.LOSS,
                metrics=ModelConfig.METRICS
            )
            self._is_built = True
            logger.info("Neural network model successfully constructed and compiled.")
        except Exception as e:
            logger.error(f"Failed to build model: {str(e)}")
            raise RuntimeError(f"Model construction failed: {str(e)}")

    def _prepare_data(self, data: List[List[float]], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input data and labels for training or prediction.

        Args:
            data: Raw input features.
            labels: Target labels.

        Returns:
            Tuple of normalized features and reshaped labels as numpy arrays.
        """
        try:
            features = np.array(data, dtype=np.float32)
            targets = np.array(labels, dtype=np.float32).reshape(-1, 1)
            return features, targets
        except ValueError as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise ValueError(f"Invalid data format: {str(e)}")

    def _compute_class_weights(self, labels: np.ndarray) -> dict:
        """Calculate class weights to handle imbalance."""
        positive_count = np.sum(labels == 1)
        negative_count = np.sum(labels == 0)
        if positive_count == 0 or negative_count == 0:
            logger.warning("Single-class data detected; using equal weights.")
            return {0: 1.0, 1: 1.0}
        weight_for_positive = negative_count / positive_count
        return {0: 1.0, 1: weight_for_positive}

    def train(self, data: List[List[float]], labels: List[int], epochs: int = ModelConfig.DEFAULT_EPOCHS) -> None:
        """
        Train the model on provided project data.

        Args:
            data: List of feature vectors.
            labels: List of binary labels (0 or 1).
            epochs: Number of training epochs.
        """
        try:
            features, targets = self._prepare_data(data, labels)
            normalized_features = self.scaler.fit_transform(features)
            class_weights = self._compute_class_weights(targets)

            self.model.fit(
                normalized_features,
                targets,
                epochs=epochs,
                verbose=1,
                class_weight=class_weights
            )
            logger.info(f"Training completed successfully for {epochs} epochs.")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Failed to train model: {str(e)}")

    def predict(self, data: List[List[float]]) -> Optional[np.ndarray]:
        """
        Predict project selection outcomes.

        Args:
            data: List of feature vectors to predict on.

        Returns:
            Binary predictions (0 or 1) or None if prediction fails.
        """
        if not self._is_built:
            logger.error("Attempted prediction with unbuilt model.")
            return None

        try:
            features = np.array(data, dtype=np.float32)
            normalized_features = self.scaler.transform(features)
            probabilities = self.model.predict(normalized_features)
            decisions = (probabilities > 0.5).astype(np.int32)
            logger.info(f"Generated predictions: {decisions.flatten().tolist()}")
            return decisions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None
