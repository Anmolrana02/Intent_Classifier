
import argparse
from src.data_loader import load_csv_data
from src.embedding_pipeline import VectorStoreManager
from src.llm_pipeline import IntentClassifier
from src import config
from src.utils.logger import get_logger
import os
import shutil

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Few-Shot Intent Classification Pipeline")
    parser.add_argument("--build-index", action="store_true", help="Build and persist the vector store index.")
    parser.add_argument("--csv-path", type=str, help="Path to the CSV data file for building the index.")
    parser.add_argument("--query", type=str, help="A user query to classify.")
    args = parser.parse_args()

    vector_store_manager = VectorStoreManager()

    if args.build_index:
        if not args.csv_path:
            logger.error("Error: --csv-path is required when using --build-index.")
            return
        
        # Clean up old database directory if it exists
        if os.path.exists(config.VECTOR_STORE_PATH):
            logger.warning(f"Removing existing vector store at {config.VECTOR_STORE_PATH}")
            shutil.rmtree(config.VECTOR_STORE_PATH)

        logger.info("Mode: Build Index")
        documents = load_csv_data(
            file_path=args.csv_path,
            source_column=config.EMBEDDING_SOURCE_COLUMN
        )
        vector_store_manager.create_and_persist_store(documents)
        logger.info("Index building complete.")

    elif args.query:
        logger.info("Mode: Query")
        if not os.path.exists(config.VECTOR_STORE_PATH):
             logger.error(f"Vector store not found at {config.VECTOR_STORE_PATH}. Please build the index first using --build-index.")
             return
        try:
            vector_store = vector_store_manager.load_persistent_store()
            classifier = IntentClassifier(vector_store, vector_store_manager.embedding_function)
            predicted_intent = classifier.predict_intent(args.query)

            print("\n" + "="*50)
            print(f" Query: '{args.query}'")
            print(f" Predicted L3 Intent: '{predicted_intent}'")
            print("="*50 + "\n")

        except Exception as e:
            logger.error(f"Could not run prediction. Details:  {str(e)}")

    else:
        logger.warning("No action specified. Use --build-index or --query. Use -h for help.")

if __name__ == "__main__":
    main()
