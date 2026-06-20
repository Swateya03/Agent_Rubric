"""
Reward Scorer node — scores both rubric variants using the Phase 1 LangChain scorer.

This is a LangGraph node, not an agent. It makes no LLM calls itself.
It delegates to score_response() from rubric_scorer.py, which handles
the actual LLM interaction.

What this node does:
  1. Calls score_response() twice — once for rubric_variant_a, once for rubric_variant_b
  2. Compares the two scores
  3. Sets current_result to the winning RubricResult
  4. Increments iteration counter
  5. Writes all results back to state

Why two variants?
  variant_a = the original retrieved rubric template (unchanged)
  variant_b = the LLM-adapted version from rubric_designer
  Scoring both lets rubric_critic compare them and pick the better one.
  This A/B comparison is what drives rubric improvement over iterations.
"""

from agentrubric.graph.state import AgentState
from agentrubric.rubric_scorer import score_response, RubricResult
from langchain_core.messages import HumanMessage
from agentrubric.logger import get_logger

logger = get_logger(__name__)


def score_with_rubric(state: AgentState, rubric_text: str) -> tuple[RubricResult | None, float]:
    """Score the response using a specific rubric.

    Args:
        state: The shared AgentState
        rubric_text: The rubric to use for scoring

    Returns:
        Tuple of (RubricResult, overall_score). Returns (None, 0.0) on error.
    """
    try:
        result = score_response(
            sample_id=state["sample_id"],
            task=state["task"],
            response=state["response"],
            rubric_text=rubric_text,
        )
        return result, result.overall_score
    except Exception as e:
        logger.warning("score_with_rubric failed: %s", e)
        return None, 0.0


def reward_scorer_node(state: AgentState) -> dict:
    """Score the response using both rubric variants and determine winner.

    This is a LangGraph node function. It reads rubric variants from state
    and returns a dict of only the keys that changed.

    Args:
        state: The shared AgentState

    Returns:
        Dict with updated state keys
    """
    result_a, score_a = score_with_rubric(state, state["rubric_variant_a"])
    result_b, score_b = score_with_rubric(state, state["rubric_variant_b"])

    if score_a >= score_b:
        current_result = result_a
        winner = "A"
    else:
        current_result = result_b
        winner = "B"

    return {
        "score_variant_a": score_a,
        "score_variant_b": score_b,
        "current_result": current_result,
        "iteration": state["iteration"] + 1,
        "run_history": [
            HumanMessage(
                content=f"reward_scorer: variant_a={score_a:.4f}, variant_b={score_b:.4f}, "
                f"winner={winner}"
            )
        ],
    }
