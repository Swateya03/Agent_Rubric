"""
Hack Detector agent — identifies reward-hacking patterns in rubric variants.

This node:
  1. Computes divergence between proxy reward and gold score
  2. Detects if divergence exceeds the hack threshold
  3. Increments hack counter
  4. On completion, the graph's conditional edge routes to rubric_designer (retry) if hack detected and retries remain, or to preference_store otherwise.
"""

from agentrubric.graph.state import AgentState
from langchain_core.messages import HumanMessage
from agentrubric.constants import (
    DIVERGENCE_ALIGNED,
    DIVERGENCE_MINOR_DRIFT,
    DIVERGENCE_SIGNIFICANT,
)
from agentrubric.logger import get_logger

logger = get_logger(__name__)


def compute_divergence(proxy_score: float, gold_score: float) -> float:
    """Compute divergence between proxy and gold scores.

    Args:
        proxy_score: Score from rubric-based reward (0.0–1.0)
        gold_score: Score from independent gold judge (0.0–1.0)

    Returns:
        Absolute difference rounded to 4 decimal places
    """
    divergence = abs(proxy_score - gold_score)
    return round(divergence, 4)


def classify_divergence(divergence: float) -> str:
    """Classify divergence into human-readable categories.

    Args:
        divergence: The divergence score (0.0–1.0)

    Returns:
        Classification label
    """
    if divergence < DIVERGENCE_ALIGNED:
        return "aligned"
    elif divergence < DIVERGENCE_MINOR_DRIFT:
        return "minor_drift"
    elif divergence < DIVERGENCE_SIGNIFICANT:
        return "significant_drift"
    else:
        return "likely_hack"


def hack_detector_node(state: AgentState) -> dict:
    """Detect potential reward hacking patterns.

    This is a LangGraph node function. It reads scores from state
    and returns a dict of only the keys that changed.

    Args:
        state: The shared AgentState

    Returns:
        Dict with updated state keys (divergence_score, hack_detected, hack_count, run_history)
    """
    # Get proxy score from reward scorer
    if state["current_result"] is not None:
        proxy_score = state["current_result"].overall_score
    else:
        proxy_score = 0.0

    # Get gold score from gold judge
    gold_score = state["gold_score"] if state["gold_score"] is not None else 0.0

    # Compute divergence
    divergence = compute_divergence(proxy_score, gold_score)

    # Determine if hack
    hack_detected = divergence >= state["hack_threshold"]

    # Increment hack count
    new_hack_count = state["hack_count"] + (1 if hack_detected else 0)

    # Build status message
    classification = classify_divergence(divergence)
    msg = (
        f"hack_detector: proxy={proxy_score:.4f}, gold={gold_score:.4f}, "
        f"divergence={divergence:.4f} ({classification}), "
        f"hack={'YES' if hack_detected else 'NO'}, "
        f"iteration={state['iteration']}/{state['max_iterations']}"
    )
    logger.debug(msg)

    return {
        "divergence_score": divergence,
        "hack_detected": hack_detected,
        "hack_count": new_hack_count,
        "run_history": [HumanMessage(content=msg)],
    }
