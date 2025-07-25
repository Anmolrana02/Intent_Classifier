from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.persist_directory = config.VECTOR_STORE_PATH
        self.embedding_function = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME)
        self.vector_store = None
        logger.info(f"Initialized embedding model: {config.EMBEDDING_MODEL_NAME}")

    def create_and_persist_store(self, documents: List):
        if not documents:
            logger.warning("No documents provided to create vector store.")
            return
        
        logger.info("Creating new vector store...")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        #self.vector_store.persist()
        logger.info(f"Vector store created and persisted at {self.persist_directory}") 

    def load_persistent_store(self) -> Chroma:
        logger.info(f"Loading vector store from {self.persist_directory}...")
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            logger.info("Vector store loaded successfully.")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
        
