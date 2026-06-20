"""
Transcript Filter — quality filtering for preference data flywheel.

Rule-based (no LLM) filtering prevents garbage preference pairs from polluting
the training data used in Phase 4 fine-tuning.
"""

from agentrubric.graph.state import AgentState
from langchain_core.messages import HumanMessage
from agentrubric.constants import (
    MIN_RESPONSE_WORDS,
    MIN_TASK_WORDS,
    MIN_REPETITION_NGRAM,
    REPETITION_COUNT_LIMIT,
    ALPHA_RATIO_MIN,
    QUALITY_FLAG_THRESHOLD,
)
from agentrubric.logger import get_logger

logger = get_logger(__name__)


def compute_transcript_quality(task: str, response: str) -> tuple[float, str]:
    """Compute transcript quality score based on rule-based heuristics.

    Args:
        task: The task/question given to the model
        response: The model's response

    Returns:
        Tuple of (quality_score, flag_reason)
        quality_score is 0.0–1.0, flag_reason is "" if no issues
    """
    base_score = 1.0
    reasons = []

    # Check 1: response too short (< MIN_RESPONSE_WORDS)
    response_words = len(response.split())
    if response_words < MIN_RESPONSE_WORDS:
        base_score -= 0.5
        reasons.append("response_too_short")

    # Check 2: task too short (< MIN_TASK_WORDS)
    task_words = len(task.split())
    if task_words < MIN_TASK_WORDS:
        base_score -= 0.3
        reasons.append("task_too_vague")

    # Check 3: response is mostly repetition
    words = response.split()
    if len(words) >= MIN_REPETITION_NGRAM:
        ngram_size = MIN_REPETITION_NGRAM
        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i : i + ngram_size])
            count = sum(1 for j in range(len(words) - ngram_size + 1)
                       if tuple(words[j : j + ngram_size]) == ngram)
            if count >= REPETITION_COUNT_LIMIT:
                base_score -= 0.4
                reasons.append("repetitive_response")
                break

    # Check 4: response contains only numbers/punctuation
    alpha_count = sum(1 for c in response if c.isalpha())
    total_count = len(response)
    if total_count > 0:
        alpha_ratio = alpha_count / total_count
        if alpha_ratio < ALPHA_RATIO_MIN:
            base_score -= 0.6
            reasons.append("non_text_response")

    # Clamp score to [0.0, 1.0]
    final_score = max(0.0, min(1.0, base_score))

    # Return first reason or empty string
    first_reason = reasons[0] if reasons else ""
    return final_score, first_reason


def transcript_filter_node(state: AgentState) -> dict:
    """Filter transcript quality and flag low-quality inputs.

    This is a LangGraph node function. It reads task and response from state
    and returns quality assessment.

    Args:
        state: The shared AgentState

    Returns:
        Dict with updated state keys
    """
    score, reason = compute_transcript_quality(state["task"], state["response"])
    flagged = score < QUALITY_FLAG_THRESHOLD

    msg = (
        f"transcript_filter: quality={score:.2f}, flagged={flagged}"
        + (f", reason={reason}" if flagged else "")
    )
    logger.debug(msg)

    return {
        "transcript_quality_score": score,
        "transcript_flagged": flagged,
        "transcript_flag_reason": reason if flagged else "",
        "run_history": [HumanMessage(content=msg)],
    }
