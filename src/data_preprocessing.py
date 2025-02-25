import logging

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessing:
    """Handles project data preprocessing."""

    @staticmethod
    def load_data(file_path):
        """Loads project data from a CSV file."""
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    @staticmethod
    def prepare_data(data):
        """Prepares data for machine learning."""
        if data is None:
            return None

        try:
            # Convert project names to categorical values
            data["name"] = data["name"].astype("category").cat.codes
            return data
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
