import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from src.ai_model import AIModel


class TestAIModel(unittest.TestCase):
    """Unit tests for AIModel (success cases only)."""

    def setUp(self):
        """Initialize AIModel before each test."""
        self.model = AIModel()

    def test_build_model_success(self):
        """Test successful model building."""
        self.model.build_model()
        self.assertTrue(self.model.built, "Model should be marked as built")
        self.assertIsNotNone(self.model.model, "Model instance should not be None")

    def test_predict_model_not_built(self):
        """Test prediction when model is not built."""
        data = np.array([[0.1, 0.2]], dtype=np.float32)
        predictions = self.model.predict(data)
        self.assertIsNone(predictions, "Predictions should be None when model is not built")

    @patch.object(AIModel, "build_model")
    def test_train_model_success(self, mock_build):
        """Test successful model training."""
        # Mock build_model to prevent real model creation
        self.model.built = True
        self.model.model = MagicMock()  # Mock the model instance
        self.model.model.fit.return_value = None  # Mock fit() method

        data = np.array([[0.1, 0.2], [0.5, 0.7], [0.8, 0.9]], dtype=np.float32)
        labels = np.array([0, 1, 1], dtype=np.float32)

        self.model.train_model(data, labels)

        self.model.model.fit.assert_called_once()  # Verify fit() was called
        self.assertTrue(self.model.built, "Model should remain built after training")

    @patch("tensorflow.keras.Model.predict")
    def test_predict_success(self, mock_predict):
        """Test successful prediction."""
        self.model.built = True  # Ensure model is built
        self.model.model = MagicMock()  # Mock the model instance
        self.model.model.predict.return_value = np.array([[0.8], [0.3]])  # Mock predictions

        # Mock MinMaxScaler instance on self.model
        self.model.scaler = MagicMock()
        self.model.scaler.transform.return_value = np.array([[0.2, 0.4], [0.6, 0.9]])  # Mock transformation

        data = np.array([[0.2, 0.4], [0.6, 0.9]], dtype=np.float32)
        predictions = self.model.predict(data)

        self.assertIsNotNone(predictions, "Predictions should not be None")
        self.assertEqual(predictions.shape, (2, 1), "Predictions should match input shape")
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary")


if __name__ == "__main__":
    unittest.main()
