import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_LOG_LEVEL = logging.INFO
ENCODING_UTF8 = "utf-8"

logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing."""
    encoding: str = ENCODING_UTF8
    categorical_columns: list[str] = None

    def __post_init__(self):
        self.categorical_columns = self.categorical_columns or ["name"]


class DataPreprocessing:
    """Handles project data preprocessing with configurable options."""

    def __init__(self, config: Optional[DataPreprocessingConfig] = None):
        self.config = config or DataPreprocessingConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def load_data(self, file_path: str | Path) -> Optional[pd.DataFrame]:
        """
        Loads project data from a CSV file with error handling.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame if successful, None if failed
        """
        try:
            file_path = Path(file_path)  # Ensure consistent path handling
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            df = pd.read_csv(file_path, encoding=self.config.encoding)
            self.logger.info(f"Successfully loaded data from {file_path} with shape {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {str(e)}")
            return None

    def prepare_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepares data for machine learning by converting specified columns to categorical codes.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame if successful, None if failed
        """
        if df is None or df.empty:
            self.logger.warning("Received empty or None DataFrame")
            return None

        try:
            processed_df = df.copy()  # Preserve original data
            for column in self.config.categorical_columns:
                if column in processed_df.columns:
                    processed_df[column] = (processed_df[column]
                                            .astype("category")
                                            .cat.codes)
                else:
                    self.logger.warning(f"Column {column} not found in DataFrame")

            self.logger.info(f"Successfully prepared data with columns: {processed_df.columns.tolist()}")
            return processed_df

        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            return None

    def process_pipeline(self, file_path: str | Path) -> Optional[pd.DataFrame]:
        """
        Executes the complete data processing pipeline.

        Args:
            file_path: Path to the CSV file

        Returns:
            Processed DataFrame if successful, None if failed
        """
        try:
            df = self.load_data(file_path)
            if df is None:
                return None
            return self.prepare_data(df)
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return None
