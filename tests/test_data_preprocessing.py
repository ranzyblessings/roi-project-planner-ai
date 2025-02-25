import unittest
from unittest.mock import patch

import pandas as pd

from src.data_preprocessing import DataPreprocessing  # Adjust import as needed


class TestDataPreprocessing(unittest.TestCase):
    """Unit tests for DataPreprocessing."""

    @patch("pandas.read_csv")
    def test_load_data_success(self, mock_read_csv):
        """Test successful loading of CSV data."""
        mock_data = pd.DataFrame({"name": ["Project A", "Project B"], "requiredCapital": [100.0, 200.0]})
        mock_read_csv.return_value = mock_data

        file_path = "test.csv"
        result = DataPreprocessing.load_data(file_path)

        self.assertIsInstance(result, pd.DataFrame, "Returned object should be a DataFrame")
        self.assertEqual(len(result), 2, "DataFrame should contain two rows")
        mock_read_csv.assert_called_once_with(file_path)

    @patch("pandas.read_csv")
    def test_load_data_failure(self, mock_read_csv):
        """Test failure during CSV data loading."""
        mock_read_csv.side_effect = FileNotFoundError("File not found.")

        file_path = "nonexistent.csv"
        result = DataPreprocessing.load_data(file_path)

        self.assertIsNone(result, "Should return None on file not found")
        mock_read_csv.assert_called_once_with(file_path)

    def test_prepare_data_success(self):
        """Test successful data preparation with categorical encoding."""
        data = pd.DataFrame({"name": ["Project A", "Project B"], "requiredCapital": [100.0, 200.0]})

        processed_data = DataPreprocessing.prepare_data(data)

        self.assertIsInstance(processed_data, pd.DataFrame, "Returned object should be a DataFrame")
        self.assertIn("name", processed_data.columns, "Processed data should contain 'name' column")
        self.assertTrue(pd.api.types.is_integer_dtype(processed_data["name"]),
                        "Project names should be converted to integers")

    def test_prepare_data_none_input(self):
        """Test prepare_data with None input."""
        result = DataPreprocessing.prepare_data(None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
