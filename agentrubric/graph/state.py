"""
Shared state object for the LangGraph multi-agent graph.

AgentState is the single TypedDict that every node reads from and writes to.
It is the only way nodes communicate — no node calls another node directly.
Each node receives the full state, does its work, and returns a dict
containing only the keys it changed. LangGraph merges that dict back
into the state before passing it to the next node.

The run_history field is special: it uses LangGraph's add_messages reducer,
meaning each node appends to the list rather than overwriting it. Every
other field follows last-write-wins.

Field groups and which node owns each:

  Input (set once by run_graph, never modified):
    sample_id, task, response

  Transcript quality (written by transcript_filter_node):
    transcript_quality_score, transcript_flagged, transcript_flag_reason

  Gold judge (written by gold_judge_node):
    gold_score, gold_reasoning

  Rubric variants (written by rubric_designer_node):
    retrieved_rubric_template, active_rubric, rubric_variant_a, rubric_variant_b

  Scoring (written by reward_scorer_node):
    score_variant_a, score_variant_b, current_result, iteration

  Critic (written by rubric_critic_node):
    winning_rubric, critic_reasoning

  Hack detection (written by hack_detector_node):
    divergence_score, hack_detected, hack_count, hack_threshold, max_iterations

  Preference data (written by preference_store_node):
    preference_pair_saved, chosen_rubric, rejected_rubric

  Shared across all nodes:
    run_history
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dataclasses import dataclass
from agentrubric.rubric_scorer import RubricResult
from agentrubric.constants import DEFAULT_HACK_THRESHOLD, DEFAULT_MAX_ITERATIONS


class AgentState(TypedDict):
    """Shared state for multi-agent rubric evaluation graph."""

    # --- Input fields (set once at start, never modified) ---
    sample_id: str
    """The ID of the sample being evaluated. e.g. "sample_001"."""

    task: str
    """The original task/question given to the model."""

    response: str
    """The model's response to evaluate."""

    # --- Rubric fields (written by rubric_designer, read by scorer and critic) ---
    retrieved_rubric_template: str
    """The raw rubric text retrieved by BM25 from the rubric store.
    Empty string by default."""

    active_rubric: str
    """The rubric currently being used for scoring. May be adapted from template.
    Empty string by default."""

    rubric_variant_a: str
    """First rubric variant for critic comparison. Empty string by default."""

    rubric_variant_b: str
    """Second rubric variant for critic comparison. Empty string by default."""

    # --- Scoring fields (written by reward_scorer) ---
    score_variant_a: float | None
    """Score when rubric_variant_a is used. None until scored."""

    score_variant_b: float | None
    """Score when rubric_variant_b is used. None until scored."""

    current_result: RubricResult | None
    """The latest full RubricResult from the reward scorer. None until scored."""

    # --- Critic fields (written by rubric_critic) ---
    winning_rubric: str
    """The rubric that produced the higher score. Empty string until critic runs."""

    critic_reasoning: str
    """One-sentence explanation of why the winning rubric was chosen.
    Empty string until critic runs."""

    # --- Meta fields ---
    iteration: int
    """How many times the graph has scored this sample. Starts at 0."""

    run_history: Annotated[list, add_messages]
    """Log of messages/events appended by each node. Uses LangGraph's
    add_messages reducer so appends don't overwrite."""

    # --- Gold judge fields (written by eval/gold_judge) ---
    gold_score: float | None
    """Score from the independent gold LLM judge (0.0–1.0).
    None until gold_judge runs."""

    gold_reasoning: str
    """One-sentence explanation from the gold judge. Empty string by default."""

    # --- Hack detection fields (written by hack_detector) ---
    divergence_score: float | None
    """Absolute difference between proxy reward and gold score.
    divergence = abs(current_result.overall_score - gold_score)
    None until hack_detector runs."""

    hack_detected: bool
    """True if divergence_score exceeds the hack threshold. False by default."""

    hack_threshold: float
    """The divergence threshold above which a hack is flagged. Default: 0.25"""

    hack_count: int
    """How many times a hack has been detected for this sample.
    Default: 0. Incremented each time hack_detected is True."""

    max_iterations: int
    """Maximum number of hack-detection loops before giving up.
    Default: 3. Prevents infinite loops."""

    # --- Preference data fields (written by preference_store) ---
    preference_pair_saved: bool
    """True if this run produced a preference pair saved to the store.
    False by default."""

    chosen_rubric: str
    """The rubric text that produced the higher-quality, non-hacked score.
    Empty string by default."""

    rejected_rubric: str
    """The rubric text that produced the lower or hacked score.
    Empty string by default."""

    # --- Transcript quality fields (written by transcript_filter) ---
    transcript_quality_score: float | None
    """Score 0.0–1.0 assessing input quality. None until filter runs."""

    transcript_flagged: bool
    """True if this transcript should be excluded from training data.
    False by default."""

    transcript_flag_reason: str
    """Reason for flagging, if any. Empty string by default."""


def create_initial_state(
    sample_id: str,
    task: str,
    response: str,
    hack_threshold: float = DEFAULT_HACK_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentState:
    """Create an AgentState with default values and initial inputs.

    Args:
        sample_id: The sample identifier (e.g. "sample_001")
        task: The task/question text
        response: The model's response text
        hack_threshold: The divergence threshold above which a hack is flagged
        max_iterations: Maximum number of hack-detection loops

    Returns:
        AgentState with all fields initialized to defaults
    """
    return AgentState(
        sample_id=sample_id,
        task=task,
        response=response,
        retrieved_rubric_template="",
        active_rubric="",
        rubric_variant_a="",
        rubric_variant_b="",
        score_variant_a=None,
        score_variant_b=None,
        current_result=None,
        winning_rubric="",
        critic_reasoning="",
        iteration=0,
        run_history=[],
        gold_score=None,
        gold_reasoning="",
        divergence_score=None,
        hack_detected=False,
        hack_threshold=hack_threshold,
        hack_count=0,
        max_iterations=max_iterations,
        preference_pair_saved=False,
        chosen_rubric="",
        rejected_rubric="",
        transcript_quality_score=None,
        transcript_flagged=False,
        transcript_flag_reason="",
    )


@dataclass
class RunSnapshot:
    """Read-only summary of a completed graph run.

    Used by dashboards and reporting code to read run results
    without depending on the full AgentState TypedDict.
    Not used inside the LangGraph graph itself.
    """

    sample_id: str
    """The sample that was evaluated."""

    final_score: float
    """The overall_score from the winning rubric variant."""

    gold_score: float | None
    """Score from the independent gold judge."""

    hack_detected: bool
    """Whether a reward hack was detected on the final iteration."""

    hack_count: int
    """Total number of hack detections across all iterations."""

    preference_saved: bool
    """Whether a preference pair was saved to the store."""

    divergence_score: float | None
    """Final proxy/gold divergence score."""

    iterations: int
    """Number of scoring iterations completed."""

    @classmethod
    def from_state(cls, state: "AgentState") -> "RunSnapshot":
        """Build a RunSnapshot from a completed AgentState.

        Args:
            state: The final state after graph.invoke() returns.

        Returns:
            RunSnapshot with all fields populated from state.
        """
        current_result = state["current_result"]
        return cls(
            sample_id=state["sample_id"],
            final_score=(
                current_result.overall_score
                if current_result is not None
                else 0.0
            ),
            gold_score=state["gold_score"],
            hack_detected=state["hack_detected"],
            hack_count=state["hack_count"],
            preference_saved=state["preference_pair_saved"],
            divergence_score=state["divergence_score"],
            iterations=state["iteration"],
        )
