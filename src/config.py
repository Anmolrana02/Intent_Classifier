import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_REPO_ID = "gpt-3.5-turbo"


LLM_API_KEY = os.getenv("OPENAI_API_KEY")
if not LLM_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

VECTOR_STORE_PATH = "./chroma_db_store"
EMBEDDING_SOURCE_COLUMN = "CSS_mapping"
NUM_FEW_SHOT_EXAMPLES = 5
INTENT_LABEL_COLUMN = "intent_level_3"
UTTERANCE_COLUMN = "utterance"