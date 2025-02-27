import logging
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_preprocessing import DataPreprocessing, DataPreprocessingConfig


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Set up common test fixtures
        self.config = DataPreprocessingConfig(categorical_columns=["name"])
        self.preprocessor = DataPreprocessing(self.config)

        # Sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "value": [1.0, 2.0, 3.0]
        })

        # Configure logging capture
        self.log_capture = logging.StreamHandler()
        logging.getLogger().addHandler(self.log_capture)

    def tearDown(self):
        # Clean up after each test
        logging.getLogger().removeHandler(self.log_capture)

    def test_init_with_default_config(self):
        """Test initialization with default config"""
        processor = DataPreprocessing()
        self.assertIsInstance(processor.config, DataPreprocessingConfig)
        self.assertEqual(processor.config.categorical_columns, ["name"])
        self.assertEqual(processor.config.encoding, "utf-8")

    def test_init_with_custom_config(self):
        """Test initialization with custom config"""
        custom_config = DataPreprocessingConfig(encoding="latin1", categorical_columns=["id"])
        processor = DataPreprocessing(custom_config)
        self.assertEqual(processor.config, custom_config)

    @patch('pathlib.Path.exists')
    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv, mock_exists):
        """Test successful data loading"""
        mock_exists.return_value = True  # Simulate file existence
        mock_read_csv.return_value = self.sample_df
        file_path = "test.csv"

        result = self.preprocessor.load_data(file_path)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 3))
        mock_read_csv.assert_called_once_with(Path(file_path), encoding="utf-8")

    @patch('pathlib.Path.exists')
    def test_load_data_file_not_found(self, mock_exists):
        """Test loading non-existent file"""
        mock_exists.return_value = False

        result = self.preprocessor.load_data("nonexistent.csv")

        self.assertIsNone(result)

    @patch('pandas.read_csv')
    def test_load_data_read_error(self, mock_read_csv):
        """Test handling of pandas read errors"""
        mock_read_csv.side_effect = ValueError("Invalid CSV")

        result = self.preprocessor.load_data("test.csv")

        self.assertIsNone(result)

    def test_prepare_data_success(self):
        """Test successful data preparation"""
        result = self.preprocessor.prepare_data(self.sample_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result["name"].dtype, "int8")  # Check categorical encoding
        self.assertEqual(len(result), len(self.sample_df))
        pd.testing.assert_series_equal(result["age"], self.sample_df["age"])  # Check unchanged columns

    def test_prepare_data_none_input(self):
        """Test handling of None input"""
        result = self.preprocessor.prepare_data(None)
        self.assertIsNone(result)

    def test_prepare_data_empty_df(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.preprocessor.prepare_data(empty_df)
        self.assertIsNone(result)

    def test_prepare_data_missing_column(self):
        """Test handling of missing categorical column"""
        df = pd.DataFrame({"age": [25, 30]})
        result = self.preprocessor.prepare_data(df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["age"])

    @patch('src.data_preprocessing.DataPreprocessing.load_data')
    @patch('src.data_preprocessing.DataPreprocessing.prepare_data')
    def test_process_pipeline_success(self, mock_prepare, mock_load):
        """Test successful pipeline execution"""
        mock_load.return_value = self.sample_df
        mock_prepare.return_value = self.sample_df

        result = self.preprocessor.process_pipeline("test.csv")

        self.assertIsInstance(result, pd.DataFrame)
        mock_load.assert_called_once()
        mock_prepare.assert_called_once()

    @patch('src.data_preprocessing.DataPreprocessing.load_data')
    def test_process_pipeline_load_failure(self, mock_load):
        """Test pipeline with load failure"""
        mock_load.return_value = None

        result = self.preprocessor.process_pipeline("test.csv")

        self.assertIsNone(result)
        mock_load.assert_called_once()

    def test_categorical_conversion(self):
        """Test proper categorical conversion"""
        df = pd.DataFrame({"name": ["A", "B", "A"]})
        result = self.preprocessor.prepare_data(df)

        self.assertEqual(list(result["name"]), [0, 1, 0])  # Check category codes

    @patch('pathlib.Path.exists')
    @patch('pandas.read_csv')
    def test_encoding_usage(self, mock_read_csv, mock_exists):
        """Test that custom encoding is used"""
        mock_exists.return_value = True  # Simulate file existence
        custom_config = DataPreprocessingConfig(encoding="latin1")
        processor = DataPreprocessing(custom_config)

        processor.load_data("test.csv")
        mock_read_csv.assert_called_once_with(Path("test.csv"), encoding="latin1")


if __name__ == '__main__':
    unittest.main()
