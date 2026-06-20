"""
Rubric Critic agent — compares variant rubrics and explains the winner.

This node:
  1. Looks at both variant scores
  2. Uses the LLM to explain which rubric is better
  3. Sets the winning rubric as the new active_rubric
"""

from agentrubric.graph.state import AgentState
from agentrubric.config import get_llm
from langchain_core.messages import HumanMessage
from agentrubric.utils import truncate_response, truncate_rubric
from agentrubric.logger import get_logger

logger = get_logger(__name__)


def build_critic_prompt(
    task: str,
    response: str,
    rubric_a: str,
    rubric_b: str,
    score_a: float,
    score_b: float,
) -> str:
    """Build a prompt asking the LLM to compare two rubrics.

    Args:
        task: The task/question
        response: The model response
        rubric_a: First rubric text
        rubric_b: Second rubric text
        score_a: Score produced by rubric A
        score_b: Score produced by rubric B

    Returns:
        A prompt string for the LLM critic
    """
    task_short = truncate_response(task)
    response_short = truncate_response(response)

    winner = "A" if score_a >= score_b else "B"

    return f"""You are a rubric expert. Compare these two rubrics for evaluating a response.

TASK:
{task_short}

RESPONSE:
{response_short}

RUBRIC A:
{rubric_a}

Score with Rubric A: {score_a:.4f}

RUBRIC B:
{rubric_b}

Score with Rubric B: {score_b:.4f}

Your task:
Explain in ONE SENTENCE why rubric {winner} is better for this task, or if the scores
are equal, which rubric's criteria are more precisely tailored to this task domain.

Return ONLY the one-sentence explanation, no JSON, no lists, no commentary."""


def rubric_critic_node(state: AgentState) -> dict:
    """Compare rubric variants and explain the winner.

    This is a LangGraph node function. It reads variant scores from state
    and returns a dict of only the keys that changed.

    Args:
        state: The shared AgentState

    Returns:
        Dict with updated state keys
    """
    score_a = state["score_variant_a"] or 0.0
    score_b = state["score_variant_b"] or 0.0

    if score_a >= score_b:
        winning_rubric = state["rubric_variant_a"]
        winner_label = "A"
    else:
        winning_rubric = state["rubric_variant_b"]
        winner_label = "B"

    winning_score = max(score_a, score_b)

    llm = get_llm(temperature=0)

    prompt = build_critic_prompt(
        task=state["task"],
        response=state["response"],
        rubric_a=state["rubric_variant_a"],
        rubric_b=state["rubric_variant_b"],
        score_a=score_a,
        score_b=score_b,
    )

    reasoning = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    msg = (
        f"rubric_critic: winner=variant_{winner_label} "
        f"(score={winning_score:.4f}). Reasoning: {reasoning}"
    )
    logger.info(msg)

    return {
        "winning_rubric": winning_rubric,
        "active_rubric": winning_rubric,
        "critic_reasoning": reasoning,
        "run_history": [HumanMessage(content=msg)],
    }
