"""
Conditional edges — routing logic for the LangGraph graph.

These are pure Python functions, not nodes. They are registered with
add_conditional_edges() and called by LangGraph after a node completes
to decide which node runs next. They read state but never write to it
and never make LLM calls.

Two routing decisions:

  1. After transcript_filter:
       flagged  → END   (low quality input, skip entire pipeline)
       clean    → gold_judge  (proceed to independent scoring)

  2. After hack_detector:
       hack detected + retries remaining → rubric_designer  (retry loop)
       clean OR max iterations hit       → preference_store (proceed)

The retry loop in decision 2 is what makes this system agentic rather
than a fixed pipeline. The graph can visit rubric_designer multiple times
on the same sample, each time with a stronger retry context injected into
the prompt, until either the hack resolves or max_iterations is reached.

get_retry_context() is a helper used by rubric_designer_node (not by
LangGraph directly) to build the prompt context for retry runs.
"""

from agentrubric.graph.state import AgentState
from langgraph.graph import END


def route_after_hack_detector(state: AgentState) -> str:
    """Route after hack detection: retry or proceed.

    Args:
        state: The current AgentState

    Returns:
        "rubric_designer" to retry with refined rubric,
        "preference_store" to save results and proceed
    """
    hack = state["hack_detected"]
    iteration = state["iteration"]
    max_iter = state["max_iterations"]

    if hack and iteration < max_iter:
        return "rubric_designer"  # loop back — retry with refined rubric
    else:
        return "preference_store"  # proceed — clean or max retries hit


def route_after_transcript_filter(state: AgentState) -> str:
    """Route after transcript quality filtering: save or skip.

    Args:
        state: The current AgentState

    Returns:
        END to skip (low quality),
        "gold_judge" to proceed to scoring
    """
    if state["transcript_flagged"]:
        return END  # skip — low quality input, do not save preference pair
    else:
        return "gold_judge"  # proceed to scoring


def get_retry_context(state: AgentState) -> str:
    """Generate context for rubric_designer on retry loops.

    When a hack is detected, this context is injected into the prompt so
    the designer knows why it's being retried and what to improve.

    Args:
        state: The current AgentState

    Returns:
        A context string for the designer prompt, or empty string if not retrying
    """
    if state["hack_detected"]:
        proxy_score = (
            state["current_result"].overall_score
            if state["current_result"] is not None
            else 0.0
        )
        gold_score = state["gold_score"] if state["gold_score"] is not None else 0.0

        return (
            f"RETRY CONTEXT (attempt {state['iteration']} of {state['max_iterations']}):\n"
            f"The previous rubric produced a proxy score of {proxy_score:.4f} but the gold judge "
            f"gave {gold_score:.4f} (divergence={state['divergence_score']:.4f}).\n"
            f"The rubric may be too easy to game. Design criteria that are "
            f"harder to satisfy with low-quality responses."
        )
    else:
        return ""
