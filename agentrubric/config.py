"""
Centralised LLM configuration for all nodes in the graph.

Single source of truth for model name, temperature defaults, and API key
loading. Every LLM-calling file imports get_llm() from here instead of
hardcoding these values locally.

To switch model across the entire system set the env var:
    GROQ_MODEL=llama-3.3-70b-versatile python run_pipeline.py

load_dotenv() is called here once. No other file should call it.
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from agentrubric.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Single source of truth for model name — override via GROQ_MODEL env var
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY not found in environment. "
        "LLM calls will fail. Add it to your .env file."
    )

# Cache LLM instances by temperature to avoid re-instantiation per call
_llm_cache: dict[float, "ChatGroq"] = {}


def get_llm(temperature: float = 0) -> ChatGroq:
    """Get or create a cached ChatGroq instance.

    Instances are cached by temperature value to avoid repeated
    instantiation across node calls within the same pipeline run.

    Args:
        temperature: 0 for deterministic scoring and judging.
                     0.2 for creative rubric adaptation in rubric_designer.

    Returns:
        Cached ChatGroq instance for the requested temperature.
    """
    if temperature not in _llm_cache:
        _llm_cache[temperature] = ChatGroq(
            model=GROQ_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY,
        )
    return _llm_cache[temperature]
