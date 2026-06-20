"""
Reward Model Inference — Score (task, response) pairs with the trained checkpoint.

Loads LoRA-tuned model from Phase 4 training and produces scalar reward scores.
Designed for Windows RTX 3050 4GB with float32 throughout.
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from .lora_config import ModelConfig
from agentrubric.logger import get_logger
logger = get_logger(__name__)


class RewardScorer:
    """Load trained checkpoint and score (task, response) pairs."""

    def __init__(self, checkpoint_dir: str, model_cfg: ModelConfig = None):
        """Initialize scorer.

        Args:
            checkpoint_dir: Path to trained checkpoint folder (with adapter_model.bin)
            model_cfg: Optional ModelConfig instance
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def load(self) -> None:
        """Load base model, apply LoRA weights, and prepare for inference."""
        logger.info("Loading base model: %s", self.model_cfg.model_name)
        logger.info("Device: %s", self.device)

        # Step 1: load base model in float32 on CPU
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_cfg.model_name,
            num_labels=1,
            torch_dtype=torch.float32,
            device_map="cpu",  # always load to CPU first
        )
        base_model.config.pad_token_id = base_model.config.eos_token_id

        # Step 2: load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_dir, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: apply LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)

        # Step 4: move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info(
                "Moved to CUDA. VRAM: %.3f GB allocated",
                torch.cuda.memory_allocated()/1e9
            )

        self.model.eval()
        self._loaded = True
        logger.info("RewardScorer ready. Checkpoint: %s", self.checkpoint_dir)

    def _ensure_loaded(self) -> None:
        """Load model on first use."""
        if not self._loaded:
            self.load()

    def score(self, text: str, max_length: int = 512) -> float:
        """Score a single text string.

        Args:
            text: Text to score
            max_length: Truncate to this length

        Returns:
            Normalized reward score (0.0-1.0)
        """
        self._ensure_loaded()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        raw_reward = outputs.logits[0].item()

        # Normalize to 0.0–1.0 via sigmoid
        import torch.nn.functional as F

        normalized = torch.sigmoid(torch.tensor(raw_reward)).item()
        return round(normalized, 4)

    def score_pair(self, task: str, response: str, rubric: str = "") -> float:
        """Score a (task, response) pair.

        Args:
            task: Task/prompt text
            response: Model response text
            rubric: Optional evaluation rubric

        Returns:
            Normalized reward score (0.0-1.0)
        """
        if rubric:
            text = f"Task: {task}\n\nRubric: {rubric}\n\nResponse: {response}"
        else:
            text = f"Task: {task}\n\nResponse: {response}"
        return self.score(text)

    def batch_score(self, pairs: list, show_progress: bool = True) -> list:
        """Score multiple (task, response) pairs.

        Args:
            pairs: List of dicts with keys "task", "response", "rubric" (optional)
            show_progress: Log progress

        Returns:
            List of scores (0.0-1.0)
        """
        scores = []
        n = len(pairs)
        for i, pair in enumerate(pairs):
            if show_progress and (i % 5 == 0 or i == n - 1):
                logger.debug("Scoring %d/%d", i+1, n)
            score = self.score_pair(
                pair.get("task", ""), pair.get("response", ""), pair.get("rubric", "")
            )
            scores.append(score)
        return scores


def compare_with_baseline(scorer, eval_data: list, baseline_scorer_fn) -> dict:
    """Compare reward model scores with baseline.

    Args:
        scorer: RewardScorer instance
        eval_data: List of eval dicts with "task" and "response"
        baseline_scorer_fn: Callable(task, response) -> score

    Returns:
        Dict with comparison results (win_rate, loss_rate, scores, etc.)
    """
    reward_model_scores = []
    baseline_scores = []

    for i, item in enumerate(eval_data):
        task = item.get("task", "")
        response = item.get("response", "")

        rm_score = scorer.score_pair(task, response)
        bl_score = baseline_scorer_fn(task, response)

        reward_model_scores.append(rm_score)
        baseline_scores.append(bl_score)
        logger.info(
            "Sample %d/%d: model=%.4f, baseline=%.4f",
            i+1, len(eval_data), rm_score, bl_score
        )

    n = len(eval_data)
    wins = sum(1 for m, b in zip(reward_model_scores, baseline_scores) if m > b)
    ties = sum(
        1
        for m, b in zip(reward_model_scores, baseline_scores)
        if abs(m - b) <= 0.01
    )
    losses = n - wins - ties

    return {
        "n_samples": n,
        "reward_model_scores": reward_model_scores,
        "baseline_scores": baseline_scores,
        "win_rate": round(wins / n, 4),
        "tie_rate": round(ties / n, 4),
        "loss_rate": round(losses / n, 4),
        "mean_reward_model": round(sum(reward_model_scores) / n, 4),
        "mean_baseline": round(sum(baseline_scores) / n, 4),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score responses with reward model")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/final",
        help="Path to trained checkpoint folder",
    )
    args = parser.parse_args()

    checkpoint_exists = os.path.exists(args.checkpoint)

    if not checkpoint_exists:
        print(f"Checkpoint not found at: {args.checkpoint}")
        print("Running in MOCK mode to verify API shape only.")
        print(
            "To train first: python -m agentrubric.training.reward_trainer --quick-test\n"
        )

        # Mock test — checks class API without real model
        class MockScorer:
            device = "cpu"

            def score_pair(self, task, response, rubric=""):
                # Fake score: longer + more complex response scores higher
                words = len(response.split())
                return min(round(words / 100, 4), 1.0)

            def batch_score(self, pairs, show_progress=True):
                return [
                    self.score_pair(p["task"], p["response"]) for p in pairs
                ]

        scorer = MockScorer()
        print("MockScorer initialized (no real model loaded)")
    else:
        print(f"Loading from checkpoint: {args.checkpoint}")
        scorer = RewardScorer(args.checkpoint)

    test_pairs = [
        {
            "task": "Explain what a neural network is.",
            "response": (
                "A neural network is a computational model inspired by the brain. "
                "It consists of layers of interconnected nodes that learn patterns "
                "from data by adjusting connection weights through backpropagation."
            ),
        },
        {
            "task": "Explain what a neural network is.",
            "response": "neural nets are cool machine learning things",
        },
    ]

    print("\nScoring test pairs:")
    scores = scorer.batch_score(test_pairs)

    for pair, score in zip(test_pairs, scores):
        print(f"  Response preview : {pair['response'][:65]}...")
        print(f"  Reward score     : {score:.4f}")
        print()

    if scores[0] != scores[1]:
        print("[OK] Different responses got different scores — inference working.")
    else:
        print("[!] Both responses got same score — check model loading.")
