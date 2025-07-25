
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Configures and returns a logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(name)
