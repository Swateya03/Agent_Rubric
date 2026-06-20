"""
LangGraph state graph that orchestrates the multi-agent rubric evaluation system.

Phase 2 nodes:
  - Rubric Designer (retrieval + LLM adaptation)
  - Reward Scorer (scores both variants)
  - Rubric Critic (analyzes and selects winner)

Phase 3 nodes:
  - Transcript Filter (quality checking)
  - Gold Judge (independent scoring)
  - Hack Detector (divergence analysis)
  - Preference Store (training data collection)

Supports persistent checkpointing so run history survives between sessions.
"""

import uuid
from langgraph.graph import StateGraph, START, END
from agentrubric.graph.state import AgentState, create_initial_state
from agentrubric.graph.persistence import get_checkpointer
from agentrubric.agents.rubric_designer import rubric_designer_node
from agentrubric.agents.reward_scorer import reward_scorer_node
from agentrubric.agents.rubric_critic import rubric_critic_node
from agentrubric.agents.hack_detector import hack_detector_node
from agentrubric.eval.gold_judge import gold_judge_node
from agentrubric.flywheel.preference_store import preference_store_node
from agentrubric.flywheel.transcript_filter import transcript_filter_node
from agentrubric.graph.conditional_edges import (
    route_after_hack_detector,
    route_after_transcript_filter,
)
from agentrubric.constants import DEFAULT_HACK_THRESHOLD, DEFAULT_MAX_ITERATIONS
from agentrubric.logger import get_logger

logger = get_logger(__name__)


def _validate_inputs(sample_id: str, task: str, response: str) -> None:
    """Validate inputs before starting the graph.

    Called by run_graph() before any nodes are invoked.
    Fails fast with a clear error rather than letting bad inputs
    cause confusing failures deep inside LLM calls.

    Args:
        sample_id: Must be a non-empty string.
        task: Must be at least 10 characters.
        response: Must be at least 5 characters.

    Raises:
        ValueError: If any input fails validation.
    """
    if not sample_id or not sample_id.strip():
        raise ValueError("sample_id cannot be empty")
    if not task or len(task.strip()) < 10:
        raise ValueError(
            f"task too short ({len(task)} chars). "
            "Minimum 10 characters required."
        )
    if not response or len(response.strip()) < 5:
        raise ValueError(
            f"response too short ({len(response)} chars). "
            "Minimum 5 characters required."
        )


def build_graph(use_persistence: bool = True):
    """Build the compiled LangGraph state graph (Phase 2 + Phase 3).

    Node flow:
      START
        → transcript_filter (quality check)
        → (conditional) if flagged → END
                        if clean  → gold_judge
        → gold_judge (independent scoring)
        → rubric_designer (retrieval + adaptation)
        → reward_scorer (variant scoring)
        → rubric_critic (winner analysis)
        → hack_detector (divergence check)
        → (conditional) if hack + retries → rubric_designer (loop)
                        if clean or max  → preference_store
        → preference_store (training data)
        → END

    Args:
        use_persistence: If True, enable checkpointing for run history

    Returns:
        Compiled StateGraph with all nodes and edges connected
    """
    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("transcript_filter", transcript_filter_node)
    builder.add_node("gold_judge", gold_judge_node)
    builder.add_node("rubric_designer", rubric_designer_node)
    builder.add_node("reward_scorer", reward_scorer_node)
    builder.add_node("rubric_critic", rubric_critic_node)
    builder.add_node("hack_detector", hack_detector_node)
    builder.add_node("preference_store", preference_store_node)

    # Start → transcript_filter
    builder.add_edge(START, "transcript_filter")

    # Conditional routing after transcript_filter
    builder.add_conditional_edges(
        "transcript_filter",
        route_after_transcript_filter,
        {"gold_judge": "gold_judge", END: END},
    )

    # Linear flow: gold_judge → rubric_designer → reward_scorer → rubric_critic → hack_detector
    builder.add_edge("gold_judge", "rubric_designer")
    builder.add_edge("rubric_designer", "reward_scorer")
    builder.add_edge("reward_scorer", "rubric_critic")
    builder.add_edge("rubric_critic", "hack_detector")

    # Conditional routing after hack_detector
    builder.add_conditional_edges(
        "hack_detector",
        route_after_hack_detector,
        {
            "rubric_designer": "rubric_designer",
            "preference_store": "preference_store",
        },
    )

    # preference_store → END
    builder.add_edge("preference_store", END)

    if use_persistence:
        checkpointer = get_checkpointer()
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def run_graph(
    sample_id: str,
    task: str,
    response: str,
    thread_id: str = None,
    hack_threshold: float = DEFAULT_HACK_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    use_persistence: bool = True,
) -> tuple[AgentState, str]:
    """Run the complete graph on a sample (Phase 2 + Phase 3).

    Args:
        sample_id: Sample identifier
        task: The task/question
        response: The model response to evaluate
        thread_id: Optional unique identifier for this run. If None, generates a UUID.
        hack_threshold: Divergence threshold above which a hack is flagged (default DEFAULT_HACK_THRESHOLD)
        max_iterations: Maximum hack detection loops before giving up (default DEFAULT_MAX_ITERATIONS)
        use_persistence: If True, checkpoint results to SQLite

    Returns:
        Tuple of (final_state, thread_id)
    """
    _validate_inputs(sample_id, task, response)
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    logger.info(
        "run_graph: starting sample=%s, thread=%s",
        sample_id, thread_id[:8] if thread_id else "new"
    )

    graph = build_graph(use_persistence=use_persistence)
    initial_state = create_initial_state(
        sample_id,
        task,
        response,
        hack_threshold=hack_threshold,
        max_iterations=max_iterations,
    )
    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(initial_state, config=config)

    logger.info(
        "run_graph: complete sample=%s, score=%.4f, hack=%s",
        sample_id,
        final_state["current_result"].overall_score
        if final_state["current_result"] else 0.0,
        final_state["hack_detected"],
    )

    return final_state, thread_id


def print_graph_result(final_state: AgentState) -> None:
    """Print a formatted summary of the graph results (Phase 2 + Phase 3).

    Args:
        final_state: The final state after graph execution
    """
    sample_id = final_state["sample_id"]
    task = final_state["task"]
    iteration = final_state["iteration"]
    score_a = final_state["score_variant_a"] or 0.0
    score_b = final_state["score_variant_b"] or 0.0
    current_result = final_state["current_result"]
    critic_reasoning = final_state["critic_reasoning"]
    run_history = final_state["run_history"]

    # Phase 3 fields
    transcript_quality = final_state["transcript_quality_score"]
    transcript_flagged = final_state["transcript_flagged"]
    gold_score = final_state["gold_score"]
    divergence_score = final_state["divergence_score"]
    hack_detected = final_state["hack_detected"]
    hack_count = final_state["hack_count"]
    preference_saved = final_state["preference_pair_saved"]

    winner = "A" if score_a >= score_b else "B"
    final_score = current_result.overall_score if current_result else 0.0
    status = "[PASS]" if current_result and current_result.passed else "[FAIL]"

    print("=" * 70)
    print("AgentRubric Phase 2 + Phase 3 — Graph Run Summary")
    print("=" * 70)
    print(f"Sample ID  : {sample_id}")
    print(f"Task       : {task[:70]}{'...' if len(task) > 70 else ''}")
    print(f"Iterations : {iteration}")
    print()

    # Phase 2: Scoring
    print("PHASE 2 — Rubric Evaluation:")
    print(f"  Variant A score : {score_a:.4f}")
    print(f"  Variant B score : {score_b:.4f}")
    print(f"  Winning variant : {winner}")
    print(f"  Final score     : {final_score:.4f}  {status}")
    print()

    # Phase 3: Quality and Hack Detection
    print("PHASE 3 — Quality & Hack Detection:")
    transcript_status = "[FLAGGED]" if transcript_flagged else "[OK]"
    print(f"  Transcript quality : {transcript_quality:.2f}  {transcript_status}")
    gold_score_val = gold_score if gold_score is not None else 0.0
    print(f"  Gold judge score   : {gold_score_val:.4f}")
    divergence_val = divergence_score if divergence_score is not None else 0.0
    print(f"  Divergence         : {divergence_val:.4f}")
    hack_status = f"YES (count: {hack_count})" if hack_detected else "NO"
    print(f"  Hack detected      : {hack_status}")
    print(f"  Preference saved   : {preference_saved}")
    print()

    print("Critic reasoning:")
    print(f"  {critic_reasoning}")
    print()

    print("Run history:")
    for msg in run_history:
        print(f"  -> {msg.content}")
    print("=" * 70)

