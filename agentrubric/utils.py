"""
utils.py — Pure helper functions with no dependencies on the agentrubric package.

Import from here for shared utility logic.
Never import from graph/, agents/, or other agentrubric modules here.
"""

from agentrubric.constants import (
    MAX_TASK_DISPLAY,
    MAX_RESPONSE_DISPLAY,
    MAX_RUBRIC_DISPLAY,
    MAX_REASONING_DISPLAY,
)


def truncate(text: str, max_len: int = MAX_TASK_DISPLAY) -> str:
    """Truncate text to max_len characters, adding ellipsis if truncated.

    Args:
        text: The string to truncate.
        max_len: Maximum character length. Defaults to MAX_TASK_DISPLAY (60).

    Returns:
        Original text if len <= max_len, else text[:max_len] + "..."

    Examples:
        truncate("hello world", 5)  → "hello..."
        truncate("hello", 10)       → "hello"
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def truncate_task(task: str) -> str:
    """Truncate a task string for display using MAX_TASK_DISPLAY."""
    return truncate(task, MAX_TASK_DISPLAY)


def truncate_response(response: str) -> str:
    """Truncate a response string for LLM prompts using MAX_RESPONSE_DISPLAY."""
    return truncate(response, MAX_RESPONSE_DISPLAY)


def truncate_rubric(rubric: str) -> str:
    """Truncate rubric text for debug output using MAX_RUBRIC_DISPLAY."""
    return truncate(rubric, MAX_RUBRIC_DISPLAY)


def truncate_reasoning(reasoning: str) -> str:
    """Truncate criterion reasoning for print_result() using MAX_REASONING_DISPLAY."""
    return truncate(reasoning, MAX_REASONING_DISPLAY)
