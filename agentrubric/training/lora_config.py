"""
LoRA Configuration for Phase 4 Reward Model Fine-Tuning.

Sized specifically for RTX 3050 4GB VRAM (Qwen2-1.5B).
"""

from peft import LoraConfig, TaskType
from dataclasses import dataclass, field
import os


@dataclass
class ModelConfig:
    """Model configuration for reward model training."""

    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen2-1.5B")
    )
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    max_seq_length: int = 512  # safe for 4GB VRAM; increase to 1024 on Colab T4
    dtype: str = "float16"  # float16 on CUDA; bfloat16 if on Ampere+ (RTX 30xx supports bfloat16)
    output_dir: str = field(
        default_factory=lambda: os.getenv(
            "TRAINING_OUTPUT_DIR", "checkpoints"
        )
    )

    def is_small_model(self) -> bool:
        """Check if model is small (1B-2B range)."""
        model_lower = self.model_name.lower()
        return any(
            keyword in model_lower for keyword in ["1.5b", "1b", "mini", "small"]
        )


def get_lora_config(model_cfg: ModelConfig) -> LoraConfig:
    """Get LoRA configuration sized for model and VRAM constraints.

    Args:
        model_cfg: ModelConfig instance

    Returns:
        LoraConfig for sequence classification (reward modeling)
    """
    # Smaller rank for smaller models to save VRAM
    if model_cfg.is_small_model():
        r = 8
        lora_alpha = 16
        lora_dropout = 0.05
    else:
        r = 16
        lora_alpha = 32
        lora_dropout = 0.1

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.SEQ_CLS,  # sequence classification = reward model head
        inference_mode=False,
    )


def print_lora_summary(lora_cfg: LoraConfig) -> None:
    """Print LoRA configuration summary.

    Args:
        lora_cfg: LoraConfig instance
    """
    print("LoRA Configuration:")
    print(f"  Task type     : {lora_cfg.task_type}")
    print(f"  Rank (r)      : {lora_cfg.r}")
    print(f"  Alpha         : {lora_cfg.lora_alpha}")
    print(f"  Dropout       : {lora_cfg.lora_dropout}")
    print(f"  Target modules: {lora_cfg.target_modules}")
    print(f"  Bias          : {lora_cfg.bias}")
