"""
Comprehensive tests for rubric_scorer.py
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path

from agentrubric.rubric_scorer import (
    load_rubric,
    parse_rubric_criteria,
    build_scoring_prompt,
    parse_llm_output,
)

RUBRICS_DIR = Path(__file__).parent.parent / "rubrics"


class TestLoadRubric:
    """Tests for load_rubric function."""

    def test_load_rubric_success(self):
        """Load rubrics/default.txt and assert properties."""
        result = load_rubric(str(RUBRICS_DIR / "default.txt"))
        assert isinstance(result, str)
        assert len(result) > 0
        assert "CRITERIA" in result
        assert "weight" in result

    def test_load_rubric_missing_file(self):
        """load_rubric should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_rubric(str(RUBRICS_DIR / "nonexistent.txt"))
        assert "nonexistent.txt" in str(exc_info.value)


class TestParseRubricCriteria:
    """Tests for parse_rubric_criteria function."""

    def test_parse_rubric_criteria_default(self):
        """Parse default rubric and assert structure."""
        rubric_text = load_rubric(str(RUBRICS_DIR / "default.txt"))
        criteria = parse_rubric_criteria(rubric_text)

        assert isinstance(criteria, list)
        assert len(criteria) == 4

        for criterion in criteria:
            assert isinstance(criterion, dict)
            assert "name" in criterion
            assert "weight" in criterion
            assert isinstance(criterion["weight"], float)
            assert 0.0 <= criterion["weight"] <= 1.0

        total_weight = sum(c["weight"] for c in criteria)
        assert abs(total_weight - 1.0) < 0.01

    def test_parse_rubric_criteria_coding(self):
        """Parse coding rubric and assert structure."""
        rubric_text = load_rubric(str(RUBRICS_DIR / "coding.txt"))
        criteria = parse_rubric_criteria(rubric_text)

        assert isinstance(criteria, list)
        assert len(criteria) == 4

        for criterion in criteria:
            assert isinstance(criterion, dict)
            assert "name" in criterion
            assert "weight" in criterion
            assert isinstance(criterion["weight"], float)
            assert 0.0 <= criterion["weight"] <= 1.0

        total_weight = sum(c["weight"] for c in criteria)
        assert abs(total_weight - 1.0) < 0.01


class TestBuildScoringPrompt:
    """Tests for build_scoring_prompt function."""

    def test_build_scoring_prompt_contains_task(self):
        """build_scoring_prompt should include all components."""
        task = "What is Python?"
        response = "Python is a language."
        rubric = "CRITERIA:\n1. Relevance (weight: 1.0)\n"

        prompt = build_scoring_prompt(task, response, rubric)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert task in prompt
        assert response in prompt
        assert "JSON" in prompt
        assert "criteria_scores" in prompt


class TestParseLLMOutput:
    """Tests for parse_llm_output function."""

    def test_parse_llm_output_valid_json(self):
        """parse_llm_output should parse valid JSON correctly."""
        raw_output = '{"criteria_scores": [{"name": "Relevance", "score": 0.8, "reasoning": "Good"}]}'
        result = parse_llm_output(raw_output)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "Relevance"
        assert result[0]["score"] == 0.8
        assert result[0]["reasoning"] == "Good"

    def test_parse_llm_output_with_markdown_fences(self):
        """parse_llm_output should extract JSON from markdown code fences."""
        raw_output = '```json\n{"criteria_scores": [{"name": "Relevance", "score": 0.8, "reasoning": "Good"}]}\n```'
        result = parse_llm_output(raw_output)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "Relevance"
        assert result[0]["score"] == 0.8
        assert result[0]["reasoning"] == "Good"

    def test_parse_llm_output_invalid_json(self):
        """parse_llm_output should raise ValueError for invalid JSON."""
        with pytest.raises(ValueError) as exc_info:
            parse_llm_output("this is not json")
        assert "Failed to parse" in str(exc_info.value)
