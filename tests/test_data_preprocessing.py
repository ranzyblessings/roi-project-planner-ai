import unittest

import tensorflow as tf

from src.ai_model import AIModel


class TestAIModel(unittest.TestCase):

    def test_build_model(self):
        """Test the build_model method to check if the model is built correctly."""
        ai_model = AIModel()

        # Build the model
        ai_model.build_model()

        # Assert that the model is built and is a Keras Sequential model
        self.assertTrue(ai_model.built)  # Check if the model is marked as built
        self.assertIsInstance(ai_model.model, tf.keras.Sequential)  # Check if the model is a Sequential model


if __name__ == "__main__":
    unittest.main()
