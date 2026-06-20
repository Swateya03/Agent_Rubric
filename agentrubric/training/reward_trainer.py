"""
Phase 4 QLoRA Reward Model Training.

Full training script using TRL RewardTrainer on preference pairs.
Optimized for RTX 3050 4GB VRAM.
"""

# Fix Windows TRL encoding issue before any TRL imports
import pathlib
_original_read_text = pathlib.Path.read_text
def _patched_read_text(self, encoding=None, errors=None, newline=None):
    if encoding is None:
        encoding = 'utf-8'
    return _original_read_text(self, encoding=encoding, errors=errors, newline=newline)
pathlib.Path.read_text = _patched_read_text

import os
import json
import torch
from dataclasses import dataclass, field
from datasets import Dataset
from trl import RewardTrainer, RewardConfig
from .model_loader import load_base_model, apply_lora
from .lora_config import ModelConfig
from agentrubric.logger import get_logger
logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for reward model training."""

    output_dir: str = field(
        default_factory=lambda: os.getenv(
            "TRAINING_OUTPUT_DIR", "checkpoints"
        )
    )

    # Batch settings — CRITICAL for 4GB VRAM
    per_device_train_batch_size: int = 1  # batch=1 is mandatory for 4GB VRAM
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    # effective batch size = 1 * 8 = 8
    # increase to 16 on Colab T4 for faster convergence

    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = use num_train_epochs; set to 50 for quick test

    # Learning rate
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # Memory optimizations for 4GB VRAM
    gradient_checkpointing: bool = True
    # trades compute for memory — essential for 4GB
    fp16: bool = False
    # mixed precision disabled due to Windows bfloat16 issues
    bf16: bool = False
    optim: str = "adamw_torch"
    # Use standard AdamW (8-bit requires GPU which may not be available)

    # Logging + saving
    logging_steps: int = 5
    eval_steps: int = 20
    save_steps: int = 20
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    eval_strategy: str = "steps"
    save_strategy: str = "steps"

    # Sequence length — must match ModelConfig.max_seq_length
    max_length: int = 512

    # Reproducibility
    seed: int = 42

    # Quick test mode
    quick_test: bool = False
    # If True, sets max_steps=10 and disables most logging


def build_reward_config(train_cfg: TrainingConfig) -> RewardConfig:
    """Build RewardConfig from TrainingConfig.

    Args:
        train_cfg: TrainingConfig instance

    Returns:
        RewardConfig for TRL RewardTrainer
    """
    if train_cfg.quick_test:
        max_steps = 10
        logging_steps = 1
        eval_steps = 5
    else:
        max_steps = train_cfg.max_steps
        logging_steps = train_cfg.logging_steps
        eval_steps = train_cfg.eval_steps

    return RewardConfig(
        output_dir=train_cfg.output_dir,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=max_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        optim=train_cfg.optim,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=False,
        max_length=train_cfg.max_length,
        seed=train_cfg.seed,
        report_to="none",  # no W&B or MLflow
        remove_unused_columns=False,
    )


def train(
    data_path: str = "data",
    model_cfg: ModelConfig = None,
    train_cfg: TrainingConfig = None,
) -> str:
    """Train reward model using TRL RewardTrainer.

    Args:
        data_path: Path to folder containing train.jsonl and eval.jsonl
        model_cfg: ModelConfig instance
        train_cfg: TrainingConfig instance

    Returns:
        Path to final checkpoint directory
    """
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainingConfig()

    logger.info("=" * 70)
    logger.info("AgentRubric — Phase 4 QLoRA Reward Model Training")
    logger.info("=" * 70)
    logger.info("Model     : %s", model_cfg.model_name)
    logger.info("Output dir: %s", train_cfg.output_dir)
    logger.info("Epochs    : %d", train_cfg.num_train_epochs)
    logger.info(
        "Batch size: %d (effective: %d)",
        train_cfg.per_device_train_batch_size,
        train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
    )
    logger.info("=" * 70)

    # Step 1: Load dataset
    logger.info("Step 1/4: Loading dataset...")
    train_jsonl = os.path.join(data_path, "train.jsonl")
    eval_jsonl = os.path.join(data_path, "eval.jsonl")

    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(
            f"train.jsonl not found at {train_jsonl}. "
            "Run: python training/dataset_prep.py first."
        )

    # Load JSONL
    def load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    train_data = load_jsonl(train_jsonl)
    eval_data = load_jsonl(eval_jsonl)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    logger.info("  Train: %d samples", len(train_dataset))
    logger.info("  Eval:  %d samples", len(eval_dataset))

    # Step 2: Load model + apply LoRA
    logger.info("Step 2/4: Loading model and applying LoRA...")
    model, tokenizer = load_base_model(model_cfg)
    model = apply_lora(model, model_cfg)

    # Step 3: Build trainer
    logger.info("Step 3/4: Configuring trainer...")
    reward_config = build_reward_config(train_cfg)
    os.makedirs(train_cfg.output_dir, exist_ok=True)

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Step 4: Train
    logger.info("Step 4/4: Training...")
    logger.info("Watch loss decrease. Ctrl+C to stop early — checkpoint is saved.")
    trainer.train()

    # Save final model
    final_dir = os.path.join(train_cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Final model saved to: %s", final_dir)

    # Save training config for reproducibility
    config_path = os.path.join(final_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_name": model_cfg.model_name,
                "lora_r": 8 if model_cfg.is_small_model() else 16,
                "epochs": train_cfg.num_train_epochs,
                "learning_rate": train_cfg.learning_rate,
                "batch_size": train_cfg.per_device_train_batch_size,
                "gradient_accumulation": train_cfg.gradient_accumulation_steps,
                "max_length": train_cfg.max_length,
            },
            f,
            indent=2,
        )

    logger.info("Training config saved to: %s", config_path)
    return final_dir


if __name__ == "__main__":
    import argparse
    import sys

    # Resolve data path relative to agentrubric/ folder
    training_dir = os.path.dirname(os.path.abspath(__file__))
    agentrubric_dir = os.path.dirname(training_dir)
    default_data_path = os.path.join(agentrubric_dir, "data")

    parser = argparse.ArgumentParser(
        description="Train AgentRubric reward model with QLoRA"
    )
    parser.add_argument(
        "--data",
        default=default_data_path,
        help="Folder with train.jsonl and eval.jsonl",
    )
    parser.add_argument(
        "--output", default=None, help="Override output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run 10 steps only (for testing)",
    )
    parser.add_argument(
        "--model", default=None, help="Override MODEL_NAME env var"
    )
    args = parser.parse_args()

    model_cfg = ModelConfig()
    if args.model:
        model_cfg.model_name = args.model

    train_cfg = TrainingConfig(
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        quick_test=args.quick_test,
    )
    if args.output:
        train_cfg.output_dir = args.output

    final_dir = train(
        data_path=args.data,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    print(f"\nDone. Checkpoint at: {final_dir}")
