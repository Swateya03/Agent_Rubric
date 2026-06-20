"""
Phase 4 Training Tests — No GPU, no real model downloads, pure mocking.

Tests for dataset prep, LoRA config, training monitoring, inference, and experiments.
"""

import pytest
import json
import os
import tempfile
import math
from unittest.mock import MagicMock, patch
from datasets import Dataset
import torch


# TEST 1: Dataset validation
def test_dataset_prep_validates_samples():
    """Test that validate_sample correctly filters invalid samples."""
    from agentrubric.training.dataset_prep import validate_sample

    # Valid sample — all rules pass
    valid = {"prompt": "x" * 30, "chosen": "c" * 60, "rejected": "r" * 60}
    result = validate_sample(valid, 0)
    assert result is not None, "Valid sample should not be skipped"
    assert result.prompt == valid["prompt"]

    # Identical chosen and rejected — useless for training
    dup = {
        "prompt": "x" * 30,
        "chosen": "same_rubric " * 10,
        "rejected": "same_rubric " * 10,
    }
    assert validate_sample(dup, 1) is None, "Identical pairs should be skipped"

    # Response too short — rubric below 50 chars threshold
    short = {"prompt": "x" * 30, "chosen": "short", "rejected": "r" * 60}
    assert validate_sample(short, 2) is None, "Short chosen rubric should be skipped"

    # Missing key
    missing_key = {"prompt": "x" * 30, "chosen": "c" * 60}
    assert validate_sample(missing_key, 3) is None, "Missing rejected key should be skipped"


# TEST 2: Dataset building with synthetic data
def test_build_dataset_with_synthetic_data():
    """Test that build_dataset correctly splits synthetic pairs."""
    from agentrubric.training.dataset_prep import generate_synthetic_pairs, build_dataset

    pairs = generate_synthetic_pairs(20)
    assert len(pairs) == 20, f"Expected 20 pairs, got {len(pairs)}"

    for i, p in enumerate(pairs):
        assert "prompt" in p, f"Pair {i} missing 'prompt'"
        assert "chosen" in p, f"Pair {i} missing 'chosen'"
        assert "rejected" in p, f"Pair {i} missing 'rejected'"
        assert p["chosen"] != p["rejected"], f"Pair {i} has identical chosen/rejected"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
        tmp_path = f.name

    try:
        dataset = build_dataset(tmp_path, train_ratio=0.8)
        assert "train" in dataset and "eval" in dataset
        assert len(dataset["train"]) > 0
        assert len(dataset["eval"]) > 0
        total = len(dataset["train"]) + len(dataset["eval"])
        assert total <= 20, "Total should not exceed input size"
    finally:
        os.unlink(tmp_path)


# TEST 3: LoRA config for small model
def test_lora_config_small_model():
    """Test that LoRA config correctly identifies small models."""
    from agentrubric.training.lora_config import ModelConfig, get_lora_config
    from peft import TaskType

    cfg = ModelConfig()
    cfg.model_name = "Qwen/Qwen2-1.5B"

    assert cfg.is_small_model() == True, "Qwen2-1.5B should be classified as small"

    lora = get_lora_config(cfg)

    assert lora.r == 8, f"Small model should have r=8, got {lora.r}"
    assert lora.lora_alpha == 16, f"Expected alpha=16, got {lora.lora_alpha}"
    assert lora.task_type == TaskType.SEQ_CLS, "Must be SEQ_CLS for reward modeling"
    assert "q_proj" in lora.target_modules, "q_proj must be in target modules"
    assert "v_proj" in lora.target_modules, "v_proj must be in target modules"
    assert lora.bias == "none"


# TEST 4: LoRA config for large model
def test_lora_config_large_model():
    """Test that LoRA config scales parameters for larger models."""
    from agentrubric.training.lora_config import ModelConfig, get_lora_config

    cfg = ModelConfig()
    cfg.model_name = "meta-llama/Llama-2-7b"

    assert (
        cfg.is_small_model() == False
    ), "Llama-2-7b should NOT be small model"

    lora = get_lora_config(cfg)
    assert lora.r == 16, f"Large model should have r=16, got {lora.r}"
    assert lora.lora_alpha == 32


# TEST 5: Monitor extracts metrics correctly
def test_monitor_extract_metrics_complete():
    """Test that monitor correctly extracts all metrics from trainer_state."""
    from agentrubric.training.monitor import extract_metrics, detect_issues

    fake_state = {
        "global_step": 10,
        "best_metric": 0.50,
        "log_history": [
            {"step": 1, "loss": 0.7327, "learning_rate": 2e-5},
            {"step": 2, "loss": 0.6061, "learning_rate": 1.9e-5},
            {"step": 3, "loss": 1.041, "learning_rate": 1.8e-5},
            {"step": 5, "eval_loss": 0.65, "eval_reward_accuracy": 0.50},
            {"step": 7, "loss": 0.4563, "learning_rate": 1.5e-5},
            {"step": 9, "loss": 0.4157, "learning_rate": 1.2e-5},
            {"step": 10, "eval_loss": 0.50, "eval_reward_accuracy": 0.625},
        ],
    }

    metrics = extract_metrics(fake_state)

    assert metrics["train_steps"] == [1, 2, 3, 7, 9], \
        f"Wrong train_steps: {metrics['train_steps']}"
    assert metrics["eval_steps"] == [5, 10], \
        f"Wrong eval_steps: {metrics['eval_steps']}"
    assert metrics["total_steps"] == 10
    assert metrics["final_train_loss"] == pytest.approx(0.4157, abs=0.0001)
    assert metrics["final_eval_loss"] == pytest.approx(0.50, abs=0.0001)
    assert len(metrics["eval_reward_acc"]) == 2

    # Based on actual training: loss started 0.73 and ended 0.42 — should be OK
    issues = detect_issues(metrics)
    assert not any("did not decrease" in w for w in issues), \
        "Loss did decrease — should not flag this"


# TEST 6: Monitor detects training issues
def test_monitor_detects_no_improvement():
    """Test that monitor detects when loss doesn't improve."""
    from agentrubric.training.monitor import detect_issues

    # Simulate loss going UP (bad training)
    metrics = {
        "train_loss": [0.4, 0.5, 0.6, 0.7, 0.8],  # increasing!
        "eval_loss": [0.5, 0.6, 0.7],  # also increasing
        "eval_reward_acc": [],
        "total_steps": 5,
        "final_train_loss": 0.8,
        "final_eval_loss": 0.7,
        "best_eval_loss": None,
        "train_steps": [1, 2, 3, 4, 5],
        "eval_steps": [3, 5],
        "learning_rates": [],
    }

    issues = detect_issues(metrics)

    assert len(issues) > 0, "Should have detected at least one issue"
    assert any("did not decrease" in w for w in issues), \
        "Should flag that loss did not decrease"
    assert any("overfitting" in w or "increased" in w for w in issues), \
        "Should flag eval loss increasing"


# TEST 7: RewardScorer with mocked model
def test_reward_scorer_mock_inference():
    """Test that RewardScorer initializes with mocked model components."""
    from agentrubric.training.reward_inference import RewardScorer

    with patch("agentrubric.training.reward_inference.AutoModelForSequenceClassification.from_pretrained") as mock_model_cls, \
         patch("agentrubric.training.reward_inference.PeftModel.from_pretrained") as mock_peft, \
         patch("agentrubric.training.reward_inference.AutoTokenizer.from_pretrained") as mock_tok_cls:

        # Set up mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.__call__ = MagicMock(
            return_value={
                "input_ids": torch.ones((1, 10), dtype=torch.long),
                "attention_mask": torch.ones((1, 10), dtype=torch.long),
            }
        )
        mock_tok_cls.return_value = mock_tokenizer

        # Set up mock base model
        mock_base = MagicMock()
        mock_base.config = MagicMock()
        mock_base.config.eos_token_id = 2
        mock_base.to = MagicMock(return_value=mock_base)
        mock_model_cls.return_value = mock_base

        # Set up mock LoRA model
        mock_peft_model = MagicMock()
        mock_peft_model.eval = MagicMock()
        mock_peft_model.to = MagicMock(return_value=mock_peft_model)
        mock_peft.return_value = mock_peft_model

        scorer = RewardScorer("fake/checkpoint/path")
        scorer.load()

        assert scorer._loaded == True
        assert scorer.model is not None
        assert scorer.tokenizer is not None


# TEST 8: Compute agreement metrics
def test_compute_agreement_metrics_perfect_model():
    """Test that metrics correctly compare model against baseline and gold."""
    from agentrubric.experiments.run_experiment import compute_agreement_metrics

    # Perfect model: scores match gold exactly
    gold_scores = [0.9, 0.3, 0.7, 0.5, 0.8]
    model_scores = [0.9, 0.3, 0.7, 0.5, 0.8]  # identical to gold
    baseline_scores = [0.4, 0.4, 0.4, 0.4, 0.4]  # flat, uninformative (not matching any gold)

    metrics = compute_agreement_metrics(model_scores, gold_scores, baseline_scores)

    assert "error" not in metrics, f"Should not error: {metrics}"
    assert metrics["spearman_model"] == pytest.approx(1.0, abs=0.001), \
        "Perfect model should have Spearman rho = 1.0"
    assert metrics["win_rate"] == pytest.approx(1.0, abs=0.01), \
        "Perfect model should win against flat baseline on all samples"
    assert metrics["mae_model"] == pytest.approx(0.0, abs=0.001), \
        "Perfect model should have 0 MAE vs gold"
    assert metrics["mae_baseline"] > metrics["mae_model"], \
        "Flat baseline should have higher MAE than perfect model"
