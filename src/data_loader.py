import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_csv_data(
    file_path: str,
    source_column: str,
    metadata_columns: Optional[List[str]] = None
) ->List:
    if metadata_columns is None:
        pd.read_csv(file_path)
        all_columns = pd.read_csv(file_path).columns.tolist()
        metadata_columns = [col for col in all_columns if col != source_column]
        if source_column not in all_columns:
            raise ValueError(f"Source column '{source_column}' not found in CSV.")
        metadata_columns = [col for col in all_columns if col != source_column]
        logger.info(f"Using metadata columns: {metadata_columns}")

    loader = CSVLoader(
        file_path=file_path,
        source_column=source_column,
        metadata_columns=metadata_columns,
        encoding="utf-8"
    )
    try:
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        logger.error(f"Error: The file at {file_path} was not found.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the CSV file: {e}")
        raise