import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.ai_model import AIModel, ModelConfig


class TestAIModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test."""
        self.model = AIModel()
        # Mock the logger to avoid real logging output during tests
        self.logger_patcher = patch('src.ai_model.logger')
        self.mock_logger = self.logger_patcher.start()
        # Seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    def tearDown(self):
        """Clean up after each test."""
        self.logger_patcher.stop()

    def test_init(self):
        """Test AIModel initialization."""
        self.assertIsInstance(self.model.scaler, MinMaxScaler)
        self.assertFalse(self.model._is_built)
        self.assertIsNone(self.model._model)

    @patch('tensorflow.keras.Sequential')
    def test_build_model_success(self, mock_sequential):
        """Test successful model building."""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        self.model._build_model()
        self.assertTrue(self.model._is_built)
        self.assertEqual(self.model._model, mock_model)
        mock_model.compile.assert_called_once_with(
            optimizer=ModelConfig.OPTIMIZER,
            loss=ModelConfig.LOSS,
            metrics=ModelConfig.METRICS
        )
        self.mock_logger.info.assert_called_once_with("Neural network model successfully constructed and compiled.")

    @patch('tensorflow.keras.Sequential', side_effect=Exception("Build error"))
    def test_build_model_failure(self, mock_sequential):
        """Test model building failure."""
        with self.assertRaises(RuntimeError) as context:
            self.model._build_model()
        self.assertIn("Model construction failed: Build error", str(context.exception))
        self.mock_logger.error.assert_called_once_with("Failed to build model: Build error")

    def test_model_property_lazy_load(self):
        """Test lazy loading of the model property."""
        with patch.object(self.model, '_build_model') as mock_build:
            # Ensure the mock sets _model to avoid the assertion failure
            mock_build.side_effect = lambda: setattr(self.model, '_model', Mock()) or setattr(self.model, '_is_built',
                                                                                              True)
            _ = self.model.model
            mock_build.assert_called_once()
        self.assertTrue(self.model._is_built)

    def test_prepare_data_valid(self):
        """Test data preprocessing with valid input."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        labels = [0, 1]
        features, targets = self.model._prepare_data(data, labels)
        np.testing.assert_array_equal(features, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        np.testing.assert_array_equal(targets, np.array([[0], [1]], dtype=np.float32))

    def test_prepare_data_invalid(self):
        """Test data preprocessing with invalid input."""
        data = [[1.0, "invalid"], [3.0, 4.0]]
        labels = [0, 1]
        with self.assertRaises(ValueError) as context:
            self.model._prepare_data(data, labels)
        self.assertIn("Invalid data format", str(context.exception))
        self.mock_logger.error.assert_called_once()

    def test_compute_class_weights_balanced(self):
        """Test class weight computation with balanced data."""
        labels = np.array([[0], [1], [0], [1]])
        weights = self.model._compute_class_weights(labels)
        self.assertEqual(weights, {0: 1.0, 1: 1.0})

    def test_compute_class_weights_imbalanced(self):
        """Test class weight computation with imbalanced data."""
        labels = np.array([[0], [0], [0], [1]])
        weights = self.model._compute_class_weights(labels)
        self.assertEqual(weights, {0: 1.0, 1: 3.0})

    def test_compute_class_weights_single_class(self):
        """Test class weight computation with single-class data."""
        labels = np.array([[0], [0], [0]])
        weights = self.model._compute_class_weights(labels)
        self.assertEqual(weights, {0: 1.0, 1: 1.0})
        self.mock_logger.warning.assert_called_once_with("Single-class data detected; using equal weights.")

    @patch.object(AIModel, '_build_model')
    def test_train_success(self, mock_build):
        """Test successful training."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        labels = [0, 1]
        mock_model = Mock()
        self.model._model = mock_model
        self.model._is_built = True
        with patch.object(self.model.scaler, 'fit_transform', return_value=np.array(data)):
            self.model.train(data, labels, epochs=2)
        mock_model.fit.assert_called_once()
        self.mock_logger.info.assert_called_once_with("Training completed successfully for 2 epochs.")

    @patch.object(AIModel, '_build_model')
    def test_train_failure(self, mock_build):
        """Test training failure."""
        data = [[1.0, 2.0]]
        labels = [0]
        mock_model = Mock()
        mock_model.fit.side_effect = Exception("Training error")
        self.model._model = mock_model
        self.model._is_built = True
        with self.assertRaises(RuntimeError) as context:
            self.model.train(data, labels)
        self.assertIn("Failed to train model: Training error", str(context.exception))
        self.mock_logger.error.assert_called_once()

    @patch.object(AIModel, '_build_model')
    def test_predict_success(self, mock_build):
        """Test successful prediction."""
        data = [[1.0, 2.0]]
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.7]])
        self.model._model = mock_model
        self.model._is_built = True
        with patch.object(self.model.scaler, 'transform', return_value=np.array(data)):
            result = self.model.predict(data)
        np.testing.assert_array_equal(result, np.array([[1]], dtype=np.int32))
        self.mock_logger.info.assert_called_once()

    def test_predict_unbuilt_model(self):
        """Test prediction with unbuilt model."""
        data = [[1.0, 2.0]]
        result = self.model.predict(data)
        self.assertIsNone(result)
        self.mock_logger.error.assert_called_once_with("Attempted prediction with unbuilt model.")

    @patch.object(AIModel, '_build_model')
    def test_predict_failure(self, mock_build):
        """Test prediction failure."""
        data = [[1.0, "invalid"]]
        mock_model = Mock()
        self.model._model = mock_model
        self.model._is_built = True
        result = self.model.predict(data)
        self.assertIsNone(result)
        self.mock_logger.error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
