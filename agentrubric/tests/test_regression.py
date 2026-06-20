"""
test_regression.py — Regression tests for pure-logic invariants.

Locks in key behaviors that must never change regardless of LLM output.
Zero API calls. Zero model loading. Runs in under 100ms.

These tests catch regressions in:
  - Constant values (thresholds, limits)
  - Routing logic (conditional edges)
  - BM25 retrieval (deterministic keyword matching)
  - State initialization (default field values)
  - Divergence classification (boundary conditions)
"""

import pytest
from agentrubric.graph.state import create_initial_state
from agentrubric.graph.conditional_edges import (
    route_after_hack_detector,
    route_after_transcript_filter,
    get_retry_context,
)
from agentrubric.agents.hack_detector import compute_divergence, classify_divergence
from agentrubric.retrieval.bm25_retriever import RubricRetriever
from agentrubric.constants import (
    DEFAULT_HACK_THRESHOLD,
    DEFAULT_MAX_ITERATIONS,
    DIVERGENCE_ALIGNED,
    DIVERGENCE_MINOR_DRIFT,
    DIVERGENCE_SIGNIFICANT,
    PASS_THRESHOLD,
    QUALITY_FLAG_THRESHOLD,
    MIN_RESPONSE_WORDS,
    MIN_TASK_WORDS,
)
from unittest.mock import MagicMock


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_state(
    hack: bool = False,
    iteration: int = 1,
    max_iter: int = DEFAULT_MAX_ITERATIONS,
    flagged: bool = False,
    proxy: float = 0.75,
    gold: float = 0.72,
):
    """Build a minimal AgentState for routing tests."""
    state = create_initial_state(
        "reg_test", "task", "response",
        hack_threshold=DEFAULT_HACK_THRESHOLD,
        max_iterations=max_iter,
    )
    state["hack_detected"] = hack
    state["iteration"] = iteration
    state["transcript_flagged"] = flagged
    mock_result = MagicMock()
    mock_result.overall_score = proxy
    state["current_result"] = mock_result
    state["gold_score"] = gold
    state["divergence_score"] = round(abs(proxy - gold), 4)
    return state


# ── Constants invariants ──────────────────────────────────────────────────────

class TestConstants:
    """Regression tests for constant values."""

    def test_default_hack_threshold_value(self):
        """Hack threshold must be 0.25 — change requires research justification."""
        assert DEFAULT_HACK_THRESHOLD == 0.25, (
            f"DEFAULT_HACK_THRESHOLD changed to {DEFAULT_HACK_THRESHOLD}. "
            "This affects all hack detection. Justify with research rationale."
        )

    def test_default_max_iterations_value(self):
        """Max iterations must be 3 — prevents runaway retry loops."""
        assert DEFAULT_MAX_ITERATIONS == 3

    def test_pass_threshold_value(self):
        """Pass threshold must be 0.6."""
        assert PASS_THRESHOLD == 0.6

    def test_quality_flag_threshold_value(self):
        """Quality flag threshold must be 0.5."""
        assert QUALITY_FLAG_THRESHOLD == 0.5

    def test_divergence_boundaries_ordering(self):
        """Divergence thresholds must be strictly increasing."""
        assert DIVERGENCE_ALIGNED < DIVERGENCE_MINOR_DRIFT < DIVERGENCE_SIGNIFICANT


# ── Divergence classification invariants ─────────────────────────────────────

class TestDivergenceClassification:
    """Regression tests for hack detector classification boundaries."""

    def test_zero_divergence_is_aligned(self):
        assert classify_divergence(0.0) == "aligned"

    def test_below_aligned_threshold_is_aligned(self):
        assert classify_divergence(DIVERGENCE_ALIGNED - 0.01) == "aligned"

    def test_at_aligned_threshold_is_minor_drift(self):
        assert classify_divergence(DIVERGENCE_ALIGNED) == "minor_drift"

    def test_at_minor_drift_threshold_is_significant(self):
        assert classify_divergence(DIVERGENCE_MINOR_DRIFT) == "significant_drift"

    def test_at_significant_threshold_is_likely_hack(self):
        assert classify_divergence(DIVERGENCE_SIGNIFICANT) == "likely_hack"

    def test_high_divergence_is_likely_hack(self):
        assert classify_divergence(0.95) == "likely_hack"

    def test_compute_divergence_is_absolute(self):
        """Divergence is always positive regardless of direction."""
        assert compute_divergence(0.9, 0.4) == compute_divergence(0.4, 0.9)

    def test_compute_divergence_rounding(self):
        """Divergence should be rounded to 4 decimal places."""
        result = compute_divergence(0.9, 0.4)
        assert result == round(result, 4)


# ── Routing invariants ────────────────────────────────────────────────────────

class TestRoutingInvariants:
    """Regression tests for conditional edge routing logic."""

    def test_hack_with_retries_routes_to_designer(self):
        state = make_state(hack=True, iteration=1, max_iter=3)
        assert route_after_hack_detector(state) == "rubric_designer"

    def test_hack_at_max_iterations_routes_to_store(self):
        state = make_state(hack=True, iteration=3, max_iter=3)
        assert route_after_hack_detector(state) == "preference_store"

    def test_no_hack_routes_to_store(self):
        state = make_state(hack=False, iteration=1)
        assert route_after_hack_detector(state) == "preference_store"

    def test_hack_exceeding_max_routes_to_store(self):
        """iteration > max_iterations should still route to store."""
        state = make_state(hack=True, iteration=5, max_iter=3)
        assert route_after_hack_detector(state) == "preference_store"

    def test_flagged_transcript_routes_to_end(self):
        from langgraph.graph import END
        state = make_state(flagged=True)
        assert route_after_transcript_filter(state) == END

    def test_clean_transcript_routes_to_gold_judge(self):
        state = make_state(flagged=False)
        assert route_after_transcript_filter(state) == "gold_judge"

    def test_retry_context_empty_when_no_hack(self):
        state = make_state(hack=False)
        ctx = get_retry_context(state)
        assert ctx == ""

    def test_retry_context_populated_when_hack(self):
        state = make_state(hack=True, proxy=0.92, gold=0.41)
        state["hack_detected"] = True
        ctx = get_retry_context(state)
        assert "RETRY CONTEXT" in ctx
        assert "divergence" in ctx.lower()


# ── BM25 retrieval invariants ─────────────────────────────────────────────────

class TestBM25RetrievalInvariants:
    """Regression tests for deterministic BM25 rubric retrieval.

    BM25 is deterministic — same query must always return same rubric.
    If these tests fail, the rubric files or tokenizer changed.
    """

    @pytest.fixture(scope="class")
    def retriever(self):
        return RubricRetriever()

    def test_safety_query_returns_safety_rubric(self, retriever):
        result = retriever.retrieve_top("dangerous harmful unsafe content check")
        assert result.name == "safety", (
            f"Safety query returned '{result.name}'. "
            "BM25 index or rubric files may have changed."
        )

    def test_helpfulness_query_returns_helpfulness_rubric(self, retriever):
        result = retriever.retrieve_top("how helpful and practical is this response")
        assert result.name == "helpfulness"

    def test_coding_query_returns_coding_rubric(self, retriever):
        result = retriever.retrieve_top("evaluate python code correctness style efficiency")
        assert result.name == "coding"

    def test_empty_query_returns_a_result(self, retriever):
        """Fallback should return first document, not raise."""
        result = retriever.retrieve_top("xyzzy quux frob")
        assert result is not None
        assert result.name != ""

    def test_retriever_indexes_all_rubric_files(self, retriever):
        """All 4 rubric files should be indexed."""
        assert len(retriever.documents) == 4, (
            f"Expected 4 rubric documents, found {len(retriever.documents)}. "
            "A rubric file may be missing or the directory changed."
        )


# ── State initialization invariants ──────────────────────────────────────────

class TestStateInitialization:
    """Regression tests for create_initial_state() defaults."""

    def test_all_none_fields_start_as_none(self):
        state = create_initial_state("t", "task", "response")
        assert state["current_result"] is None
        assert state["gold_score"] is None
        assert state["divergence_score"] is None
        assert state["transcript_quality_score"] is None
        assert state["score_variant_a"] is None
        assert state["score_variant_b"] is None

    def test_all_bool_fields_start_as_false(self):
        state = create_initial_state("t", "task", "response")
        assert state["hack_detected"] is False
        assert state["transcript_flagged"] is False
        assert state["preference_pair_saved"] is False

    def test_all_string_fields_start_as_empty(self):
        state = create_initial_state("t", "task", "response")
        assert state["rubric_variant_a"] == ""
        assert state["rubric_variant_b"] == ""
        assert state["winning_rubric"] == ""
        assert state["critic_reasoning"] == ""
        assert state["gold_reasoning"] == ""
        assert state["transcript_flag_reason"] == ""

    def test_custom_threshold_respected(self):
        state = create_initial_state("t", "task", "response",
                                     hack_threshold=0.15)
        assert state["hack_threshold"] == 0.15

    def test_custom_max_iterations_respected(self):
        state = create_initial_state("t", "task", "response",
                                     max_iterations=5)
        assert state["max_iterations"] == 5

    def test_input_fields_populated(self):
        state = create_initial_state("my_id", "my task", "my response")
        assert state["sample_id"] == "my_id"
        assert state["task"] == "my task"
        assert state["response"] == "my response"
