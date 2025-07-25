
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from typing import Dict
from src import config
from src.utils.logger import get_logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

logger = get_logger(__name__)

class IntentClassifier:
    def __init__(self, vector_store: Chroma, embedding_function):
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.llm = self._initialize_llm()
        

    def _initialize_llm(self) -> ChatOpenAI:
        logger.info(f"Initializing LLM: {config.LLM_REPO_ID}")
        return ChatOpenAI(
            model_name=config.LLM_REPO_ID,
            temperature=0.1,
            max_tokens=128,
            
        )

    def predict_intent(self, query: str) -> str:
        logger.info(f"Finding relevant examples for query: '{query}'")

        # Step 1: Manually find the N most similar examples
        similar_docs = self.vector_store.similarity_search(
            query, k=config.NUM_FEW_SHOT_EXAMPLES
        )

        # Step 2: Format these examples into a string
        examples_str = "\n\nHere are some examples:\n"
        for doc in similar_docs:
            example_query = doc.metadata.get(config.UTTERANCE_COLUMN, "")
            example_intent = doc.metadata.get(config.INTENT_LABEL_COLUMN, "")
            examples_str += f"\nQuery: {example_query}\nIntent: {example_intent}\n"

        # Step 3: Create the final prompt using a simple f-string
        final_prompt = (
            "You are an expert intent classifier. Given a user query, "
            "classify it into one of the provided intents based on the examples below."
            f"{examples_str}"
            "\nNow, classify the following query:\n"
            f"Query: {query}\n"
            "Intent:"
        )

        logger.info("Running prediction with manually built prompt...")
        try:
            # Step 4: Invoke the LLM directly with the prompt string
            response = self.llm.invoke(final_prompt)
            # The response is now a raw string, no .get("text") needed
            predicted_intent = response.content.strip()
            logger.info(f"LLM Prediction: '{predicted_intent}'")
            return predicted_intent
        except StopIteration:
            #This can happen with some models when they finish generating text.
            logger.error("LLM interference stopped unexpectedly (StopIteration). This might be a model comatibility issue.")
            return "Error: LLM stopped unexpectedly."
        except ValueError as ve:
            logger.error(f"LLM output could not be parsed (ValueError): {ve}")
            return "Error: Could not parse LLM output."
        except Exception as e:
            logger.error(f"Error during LLM inference: {type(e).__name__} - {e}")
            return "Error: Could not get a prediction."
