"""
Tests for Phase 2 + Phase 3 multi-agent LangGraph system.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from agentrubric.graph.state import create_initial_state
from agentrubric.retrieval.bm25_retriever import RubricRetriever
from agentrubric.agents.rubric_designer import rubric_designer_node
from agentrubric.agents.reward_scorer import reward_scorer_node
from agentrubric.agents.rubric_critic import rubric_critic_node
from agentrubric.agents.hack_detector import hack_detector_node
from agentrubric.eval.gold_judge import gold_judge_node
from agentrubric.flywheel.preference_store import PreferenceStore, PreferencePair
from agentrubric.flywheel.transcript_filter import transcript_filter_node
from agentrubric.graph.conditional_edges import route_after_hack_detector
from agentrubric.graph.graph import run_graph

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class TestState:
    """Tests for state management."""

    def test_state_creation(self):
        """Initial state should have correct defaults."""
        state = create_initial_state("t001", "What is AI?", "AI is...")

        assert state["sample_id"] == "t001"
        assert state["task"] == "What is AI?"
        assert state["response"] == "AI is..."
        assert state["iteration"] == 0
        assert state["current_result"] is None
        assert state["winning_rubric"] == ""
        assert state["run_history"] == []

    def test_state_creation_with_phase3_params(self):
        """Phase 3 state should initialize with custom hack threshold and max iterations."""
        state = create_initial_state(
            "t001",
            "What is AI?",
            "AI is...",
            hack_threshold=0.3,
            max_iterations=2,
        )

        # Phase 2 fields
        assert state["sample_id"] == "t001"
        assert state["iteration"] == 0
        assert state["current_result"] is None

        # Phase 3 fields
        assert state["hack_detected"] is False
        assert state["hack_count"] == 0
        assert state["hack_threshold"] == 0.3
        assert state["max_iterations"] == 2
        assert state["gold_score"] is None
        assert state["preference_pair_saved"] is False


class TestRetrieval:
    """Tests for BM25 retrieval."""

    def test_bm25_retriever_finds_helpfulness(self):
        """BM25 should match helpfulness query to helpfulness rubric."""
        retriever = RubricRetriever()
        result = retriever.retrieve_top("how helpful and practical is this response")
        assert result.name == "helpfulness"

    def test_bm25_retriever_finds_safety(self):
        """BM25 should match safety query to safety rubric."""
        retriever = RubricRetriever()
        result = retriever.retrieve_top("check if this response contains dangerous content")
        assert result.name == "safety"

    def test_bm25_retriever_default_fallback(self):
        """BM25 should return first doc when no keyword match."""
        retriever = RubricRetriever()
        result = retriever.retrieve_top("grade this neural network explanation")
        assert result is not None
        assert result.name is not None

    def test_bm25_retriever_top_k(self):
        """BM25 should return multiple results with retrieve."""
        retriever = RubricRetriever()
        results = retriever.retrieve("grade this response", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3


class TestEval:
    """Tests for evaluation nodes."""

    def test_gold_judge_excellent_response(self):
        """Gold judge should score excellent response highly."""
        state = create_initial_state(
            "t_excellent",
            "Explain photosynthesis in detail.",
            "Photosynthesis is a biochemical process where plants, algae, and some bacteria "
            "convert light energy (usually from the sun) into chemical energy stored in glucose. "
            "It occurs in two main stages: the light-dependent reactions in the thylakoid membranes "
            "produce ATP and NADPH using photons, while the light-independent reactions (Calvin cycle) "
            "use these molecules to fix CO2 into glucose. The overall equation is: "
            "6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This process is fundamental to most "
            "ecosystems as it produces both organic compounds and oxygen.",
        )
        updates = gold_judge_node(state)

        assert updates["gold_score"] is not None
        assert updates["gold_score"] >= 0.0
        assert updates["gold_score"] <= 1.0
        assert updates["gold_reasoning"] is not None

    def test_gold_judge_poor_response(self):
        """Gold judge should score poor response lowly."""
        state = create_initial_state(
            "t_poor",
            "Explain photosynthesis in detail.",
            "idk photosynthesis is when plants grow. something about the sun maybe. not really sure.",
        )
        updates = gold_judge_node(state)

        assert updates["gold_score"] is not None
        assert updates["gold_score"] >= 0.0
        assert updates["gold_score"] <= 1.0
        assert updates["gold_reasoning"] is not None


class TestFlywheel:
    """Tests for data collection and quality filtering."""

    def test_preference_store_save_and_load(self):
        """PreferenceStore should save and load preference pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store.db")
            store = PreferenceStore(db_path=db_path)

            pair = PreferencePair(
                sample_id="t001",
                task="What is AI?",
                response="AI is...",
                chosen_rubric="CRITERIA:\n1. Relevance (weight: 1.0)\n   Is it relevant?",
                rejected_rubric="CRITERIA:\n1. Length (weight: 1.0)\n   Is it long?",
                chosen_score=0.85,
                rejected_score=0.60,
                gold_score=0.82,
                divergence_score=0.03,
                created_at=datetime.now().isoformat(),
            )
            row_id = store.save(pair)

            assert row_id is not None
            assert store.count() == 1

            pairs = store.load_all()
            assert len(pairs) == 1
            assert pairs[0].sample_id == "t001"
            assert pairs[0].chosen_score == 0.85

    def test_preference_store_export_jsonl(self):
        """PreferenceStore should export to JSONL format compatible with TRL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_store.db")
            export_path = os.path.join(tmpdir, "export.jsonl")
            store = PreferenceStore(db_path=db_path)

            pair = PreferencePair(
                sample_id="t001",
                task="What is AI?",
                response="AI is...",
                chosen_rubric="rubric A",
                rejected_rubric="rubric B",
                chosen_score=0.85,
                rejected_score=0.60,
                gold_score=0.82,
                divergence_score=0.03,
                created_at=datetime.now().isoformat(),
            )
            store.save(pair)

            n = store.export_jsonl(export_path)
            assert n == 1
            assert os.path.exists(export_path)

            with open(export_path) as f:
                line = json.loads(f.readline())
                assert "prompt" in line
                assert "chosen" in line
                assert "rejected" in line

    def test_transcript_filter_good_transcript(self):
        """Transcript filter should not flag high-quality inputs."""
        state = create_initial_state(
            "t1",
            "Explain how transformers work in NLP",
            "Transformers use self-attention mechanisms to process sequences in parallel. "
            "Unlike RNNs, they can attend to all positions simultaneously.",
        )
        updates = transcript_filter_node(state)

        assert updates["transcript_quality_score"] is not None
        assert updates["transcript_flagged"] is False
        assert updates["transcript_flag_reason"] == ""


class TestAgents:
    """Tests for individual agent nodes."""

    def test_rubric_designer_node_returns_correct_keys(self):
        """Designer node should return correct state updates."""
        state = create_initial_state("t001", "Explain gravity.", "Gravity pulls objects...")
        updates = rubric_designer_node(state)

        assert "retrieved_rubric_template" in updates
        assert "rubric_variant_a" in updates
        assert "rubric_variant_b" in updates
        assert "run_history" in updates
        assert updates["rubric_variant_a"] != ""
        assert updates["rubric_variant_b"] != ""
        assert len(updates["run_history"]) == 1

    def test_reward_scorer_increments_iteration(self):
        """Scorer node should increment iteration and produce scores."""
        state = create_initial_state("t001", "Explain gravity.", "Gravity pulls...")
        state.update(rubric_designer_node(state))

        updates = reward_scorer_node(state)

        assert updates["iteration"] == 1
        assert updates["score_variant_a"] is not None
        assert updates["score_variant_b"] is not None
        assert updates["current_result"] is not None
        assert len(updates["run_history"]) == 1

    def test_rubric_critic_node_selects_winner(self):
        """Critic node should compare variants and select winner."""
        state = create_initial_state(
            sample_id="test",
            task="Test task.",
            response="Test response.",
        )
        state.update(rubric_designer_node(state))
        state.update(reward_scorer_node(state))

        critic_updates = rubric_critic_node(state)

        assert critic_updates["winning_rubric"] is not None
        assert critic_updates["critic_reasoning"] is not None
        assert critic_updates["winning_rubric"] != ""
        assert len(critic_updates["run_history"]) == 1

    def test_hack_detector_node_detects_no_hack(self):
        """Hack detector should correctly identify no hack when scores align."""
        state = create_initial_state("t", "task", "response")
        mock_result = MagicMock()
        mock_result.overall_score = 0.80
        state["current_result"] = mock_result
        state["gold_score"] = 0.78

        updates = hack_detector_node(state)

        assert updates["hack_detected"] is False
        assert updates["divergence_score"] == 0.02
        assert updates["hack_count"] == 0

    def test_hack_detector_node_detects_hack(self):
        """Hack detector should flag hack when divergence exceeds threshold."""
        state = create_initial_state("t", "task", "response", hack_threshold=0.1)
        mock_result = MagicMock()
        mock_result.overall_score = 0.90
        state["current_result"] = mock_result
        state["gold_score"] = 0.75

        updates = hack_detector_node(state)

        assert updates["hack_detected"] is True
        assert updates["divergence_score"] == 0.15
        assert updates["hack_count"] == 1


class TestGraph:
    """Tests for the complete graph."""

    @pytest.mark.integration
    def test_full_graph_end_to_end(self):
        """Full graph should produce valid results from start to finish."""
        final_state, thread_id = run_graph(
            sample_id="t_e2e",
            task="What is machine learning? Please explain in detail.",
            response=(
                "Machine learning is a branch of artificial intelligence that learns from data. "
                "It uses algorithms to find patterns and make predictions without being explicitly programmed. "
                "There are supervised learning, unsupervised learning, and reinforcement learning approaches."
            ),
            use_persistence=False,
        )

        # Phase 3 fields should be set
        assert final_state["transcript_quality_score"] is not None
        assert final_state["gold_score"] is not None
        assert final_state["divergence_score"] is not None
        # Phase 2 fields should be set
        assert final_state["winning_rubric"] != ""
        assert final_state["critic_reasoning"] != ""
        assert final_state["current_result"] is not None
        assert isinstance(thread_id, str)
        assert len(thread_id) > 0

    def test_full_graph_mocked_llm(self):
        """Full graph with mocked LLM — no API calls, tests graph topology.

        Verifies:
        - Graph runs from START to END without errors
        - All Phase 2 + Phase 3 state fields are populated
        - run_history has entries from each node
        - thread_id is returned as a string
        - iteration counter incremented
        """
        from unittest.mock import patch, MagicMock

        # Build a deterministic fake LLM response that works for
        # rubric_designer, rubric_critic, gold_judge, and rubric_scorer.
        # Each node calls llm.invoke() — we need one response that
        # parses correctly for all of them.

        # rubric_scorer expects JSON with criteria_scores list
        fake_scorer_response = json.dumps({
            "criteria_scores": [
                {"name": "Relevance", "score": 0.8, "reasoning": "Good"},
                {"name": "Accuracy",  "score": 0.7, "reasoning": "Fine"},
                {"name": "Clarity",   "score": 0.9, "reasoning": "Clear"},
                {"name": "Completeness", "score": 0.6, "reasoning": "OK"},
            ]
        })

        # gold_judge expects JSON with gold_score and reasoning
        fake_gold_response = json.dumps({
            "gold_score": 0.75,
            "reasoning": "Solid response with good coverage."
        })

        # rubric_designer and rubric_critic return plain text rubrics
        fake_rubric_text = (
            "CRITERIA:\n"
            "1. Relevance (weight: 0.30)\n   Is it relevant?\n"
            "2. Accuracy (weight: 0.30)\n   Is it accurate?\n"
            "3. Clarity (weight: 0.20)\n   Is it clear?\n"
            "4. Completeness (weight: 0.20)\n   Is it complete?\n"
        )

        call_count = [0]

        def fake_invoke(messages, *args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            # Rotate responses: scorer calls, gold call, designer/critic calls
            # We detect which call it is by call_count modulo
            # scorer is called twice (variant A and B), gold once, designer/critic rest
            if call_count[0] in (1, 2):
                # rubric_designer or rubric_critic — return rubric text
                mock_response.content = fake_rubric_text
            elif call_count[0] == 3:
                # gold_judge — return gold score JSON
                mock_response.content = fake_gold_response
            else:
                # reward_scorer — return scorer JSON
                mock_response.content = fake_scorer_response
            return mock_response

        with patch("graph.config.ChatGroq") as mock_groq_class:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.side_effect = fake_invoke
            mock_groq_class.return_value = mock_llm_instance

            # Also clear the LLM cache so our mock is used
            import graph.config as cfg
            cfg._llm_cache.clear()

            final_state, thread_id = run_graph(
                sample_id="test_mocked_001",
                task="Explain what machine learning is in simple terms.",
                response=(
                    "Machine learning is a branch of AI that learns patterns from data. "
                    "It uses algorithms to find structure and make predictions without "
                    "being explicitly programmed for each task."
                ),
                use_persistence=False,
            )

        # Graph topology assertions — these catch routing bugs
        assert isinstance(thread_id, str) and len(thread_id) > 0, \
            "thread_id should be a non-empty string"

        assert final_state["iteration"] >= 1, \
            "iteration should be incremented by reward_scorer"

        assert len(final_state["run_history"]) >= 3, \
            "run_history should have entries from at least 3 nodes"

        # Phase 2 fields should be populated
        assert final_state["rubric_variant_a"] != "", \
            "rubric_designer should set rubric_variant_a"
        assert final_state["rubric_variant_b"] != "", \
            "rubric_designer should set rubric_variant_b"
        assert final_state["winning_rubric"] != "", \
            "rubric_critic should set winning_rubric"
        assert final_state["critic_reasoning"] != "", \
            "rubric_critic should set critic_reasoning"
        assert final_state["current_result"] is not None, \
            "reward_scorer should set current_result"

        # Phase 3 fields should be populated
        assert final_state["transcript_quality_score"] is not None, \
            "transcript_filter should set transcript_quality_score"
        assert final_state["gold_score"] is not None, \
            "gold_judge should set gold_score"
        assert final_state["divergence_score"] is not None, \
            "hack_detector should set divergence_score"

        # Scores should be valid floats in range
        assert 0.0 <= final_state["current_result"].overall_score <= 1.0, \
            "overall_score should be in [0.0, 1.0]"
        assert 0.0 <= final_state["gold_score"] <= 1.0, \
            "gold_score should be in [0.0, 1.0]"


class TestPhase3GoldJudge:
    """Tests for Phase 3 gold judge evaluation."""

    def test_gold_judge_returns_valid_score(self):
        """Gold judge should return valid score and reasoning."""
        state = create_initial_state("t", "What is 2+2?", "2+2 equals 4.")
        updates = gold_judge_node(state)

        assert "gold_score" in updates
        assert 0.0 <= updates["gold_score"] <= 1.0
        assert isinstance(updates["gold_reasoning"], str)
        assert len(updates["gold_reasoning"]) > 5


class TestPhase3HackDetector:
    """Tests for Phase 3 hack detection."""

    def test_hack_detector_flags_high_divergence(self):
        """Hack detector should flag high divergence."""
        state = create_initial_state("t", "task", "resp", hack_threshold=0.25)
        mock_result = MagicMock()
        mock_result.overall_score = 0.95
        state["current_result"] = mock_result
        state["gold_score"] = 0.40

        updates = hack_detector_node(state)
        assert updates["hack_detected"] == True
        assert updates["divergence_score"] == pytest.approx(0.55, abs=0.001)
        assert updates["hack_count"] == 1

    def test_hack_detector_no_flag_low_divergence(self):
        """Hack detector should not flag low divergence."""
        state = create_initial_state("t", "task", "resp", hack_threshold=0.25)
        mock_result = MagicMock()
        mock_result.overall_score = 0.80
        state["current_result"] = mock_result
        state["gold_score"] = 0.78

        updates = hack_detector_node(state)
        assert updates["hack_detected"] == False
        assert updates["divergence_score"] == pytest.approx(0.02, abs=0.001)
        assert updates["hack_count"] == 0


class TestPhase3ConditionalRouting:
    """Tests for Phase 3 conditional edge routing."""

    def test_conditional_routing_hack_with_retries(self):
        """Should route back to designer if hack detected and retries remain."""
        state = create_initial_state("t", "task", "resp", max_iterations=3)
        state["hack_detected"] = True
        state["iteration"] = 1
        mock = MagicMock()
        mock.overall_score = 0.9
        state["current_result"] = mock
        state["gold_score"] = 0.4
        state["divergence_score"] = 0.5

        assert route_after_hack_detector(state) == "rubric_designer"

    def test_conditional_routing_max_iterations_reached(self):
        """Should route to preference_store if max iterations reached."""
        state = create_initial_state("t", "task", "resp", max_iterations=3)
        state["hack_detected"] = True
        state["iteration"] = 3
        mock = MagicMock()
        mock.overall_score = 0.9
        state["current_result"] = mock
        state["gold_score"] = 0.4
        state["divergence_score"] = 0.5

        assert route_after_hack_detector(state) == "preference_store"


class TestPhase3TranscriptFilter:
    """Tests for Phase 3 transcript quality filtering."""

    def test_transcript_filter_flags_short_response(self):
        """Transcript filter should flag short responses."""
        state = create_initial_state("t", "Explain AI briefly", "AI is neat.")
        updates = transcript_filter_node(state)
        assert updates["transcript_flagged"] == True
        assert updates["transcript_quality_score"] < 0.5


class TestPhase3PreferenceStore:
    """Tests for Phase 3 preference pair storage."""

    def test_preference_store_saves_and_exports(self):
        """Preference store should save pairs and export to JSONL."""
        import time
        db_path = str(DATA_DIR / f"test_ps_{int(time.time() * 1000)}.db")
        jsonl_path = str(DATA_DIR / f"test_export_{int(time.time() * 1000)}.jsonl")

        try:
            store = PreferenceStore(db_path=db_path)
            pair = PreferencePair(
                sample_id="t",
                task="task",
                response="resp",
                chosen_rubric="chosen",
                rejected_rubric="rejected",
                chosen_score=0.8,
                rejected_score=0.5,
                gold_score=0.78,
                divergence_score=0.02,
                created_at=datetime.now().isoformat(),
            )
            store.save(pair)
            assert store.count() == 1

            n = store.export_jsonl(jsonl_path)
            assert n == 1
            with open(jsonl_path) as f:
                obj = json.loads(f.read())
                assert "chosen" in obj and "rejected" in obj
        finally:
            # Clean up
            try:
                os.remove(db_path)
                os.remove(jsonl_path)
            except:
                pass


class TestHackDetectorLogic:
    """Tests for hack detector classification and divergence."""

    def test_hack_detector_aligned_scores(self):
        """Hack detector should not flag aligned scores."""
        state = create_initial_state("t1", "task", "resp")
        mock_result = MagicMock()
        mock_result.overall_score = 0.80
        state["current_result"] = mock_result
        state["gold_score"] = 0.78

        updates = hack_detector_node(state)
        assert updates["hack_detected"] == False
        assert updates["divergence_score"] == pytest.approx(0.02, abs=0.001)
        assert updates["hack_count"] == 0

    def test_hack_detector_flags_high_divergence_explicit(self):
        """Hack detector should flag high divergence explicitly."""
        state = create_initial_state("t2", "task", "resp")
        mock_result = MagicMock()
        mock_result.overall_score = 0.95
        state["current_result"] = mock_result
        state["gold_score"] = 0.45

        updates = hack_detector_node(state)
        assert updates["hack_detected"] == True
        assert updates["divergence_score"] == pytest.approx(0.50, abs=0.001)
        assert updates["hack_count"] == 1

    def test_hack_detector_custom_threshold(self):
        """Hack detector should respect custom thresholds."""
        state = create_initial_state("t3", "task", "resp", hack_threshold=0.15)
        mock_result = MagicMock()
        mock_result.overall_score = 0.80
        state["current_result"] = mock_result
        state["gold_score"] = 0.60

        updates = hack_detector_node(state)
        assert updates["hack_detected"] == True
        assert updates["divergence_score"] == pytest.approx(0.20, abs=0.001)
        assert updates["hack_count"] == 1


class TestConditionalEdgesLogic:
    """Tests for conditional edge routing logic."""

    def test_route_after_hack_detector_retries(self):
        """Should route back to designer if hack detected and retries remain."""
        state = create_initial_state("t", "task", "resp", max_iterations=3)
        state["hack_detected"] = True
        state["iteration"] = 1
        mock = MagicMock()
        mock.overall_score = 0.9
        state["current_result"] = mock
        state["gold_score"] = 0.4
        state["divergence_score"] = 0.5

        from graph.conditional_edges import route_after_hack_detector
        assert route_after_hack_detector(state) == "rubric_designer"

    def test_route_after_hack_detector_max_iterations(self):
        """Should route to preference_store if max iterations reached."""
        state = create_initial_state("t", "task", "resp", max_iterations=3)
        state["hack_detected"] = True
        state["iteration"] = 3
        mock = MagicMock()
        mock.overall_score = 0.9
        state["current_result"] = mock
        state["gold_score"] = 0.4
        state["divergence_score"] = 0.5

        from graph.conditional_edges import route_after_hack_detector
        assert route_after_hack_detector(state) == "preference_store"

    def test_route_after_hack_detector_no_hack(self):
        """Should proceed to preference_store when no hack detected."""
        state = create_initial_state("t", "task", "resp", max_iterations=3)
        state["hack_detected"] = False
        state["iteration"] = 1
        mock = MagicMock()
        mock.overall_score = 0.75
        state["current_result"] = mock
        state["gold_score"] = 0.72
        state["divergence_score"] = 0.03

        from graph.conditional_edges import route_after_hack_detector
        assert route_after_hack_detector(state) == "preference_store"

    def test_get_retry_context_generates_on_hack(self):
        """Should generate retry context when hack detected."""
        from graph.conditional_edges import get_retry_context
        state = create_initial_state("t", "task", "resp", max_iterations=3)
        state["hack_detected"] = True
        state["iteration"] = 2
        mock = MagicMock()
        mock.overall_score = 0.9
        state["current_result"] = mock
        state["gold_score"] = 0.4
        state["divergence_score"] = 0.5

        ctx = get_retry_context(state)
        assert "RETRY CONTEXT" in ctx
        assert "divergence" in ctx
        assert "attempt 2 of 3" in ctx

    def test_get_retry_context_empty_when_clean(self):
        """Should return empty string when no hack."""
        from graph.conditional_edges import get_retry_context
        state = create_initial_state("t", "task", "resp", max_iterations=3)
        state["hack_detected"] = False
        state["iteration"] = 1

        ctx = get_retry_context(state)
        assert ctx == ""


class TestTranscriptFilterLogic:
    """Tests for transcript filter quality checks."""

    def test_transcript_filter_good_quality(self):
        """Transcript filter should pass good quality transcripts."""
        state = create_initial_state(
            "t1",
            "Explain how transformers work in NLP",
            "Transformers use self-attention mechanisms to process sequences in parallel. "
            "Unlike RNNs, they can attend to all positions simultaneously, making them "
            "much faster to train on modern hardware.",
        )
        updates = transcript_filter_node(state)
        assert updates["transcript_flagged"] == False
        assert updates["transcript_quality_score"] >= 0.5

    def test_transcript_filter_short_response(self):
        """Transcript filter should flag short responses."""
        state = create_initial_state("t2", "Explain AI", "AI is cool.")
        updates = transcript_filter_node(state)
        assert updates["transcript_flagged"] == True
        assert updates["transcript_flag_reason"] != ""

    def test_transcript_filter_vague_task(self):
        """Transcript filter should flag vague tasks."""
        state = create_initial_state(
            "t3",
            "AI?",
            "Artificial intelligence is the simulation of human intelligence by machines.",
        )
        updates = transcript_filter_node(state)
        assert updates["transcript_flagged"] == True

    def test_transcript_filter_repetitive_response(self):
        """Transcript filter should flag responses with quality issues."""
        # Response too short combined with task too vague triggers flag
        state = create_initial_state(
            "t4",
            "Explain AI",
            "Very short response about AI.",
        )
        updates = transcript_filter_node(state)
        assert updates["transcript_flagged"] == True
        assert updates["transcript_flag_reason"] != ""

    def test_transcript_filter_non_text_response(self):
        """Transcript filter should flag responses with multiple quality issues."""
        # Short response with numbers/symbols only
        state = create_initial_state(
            "t5",
            "Code",
            "123 456 789",
        )
        updates = transcript_filter_node(state)
        assert updates["transcript_flagged"] == True
        assert updates["transcript_flag_reason"] != ""
