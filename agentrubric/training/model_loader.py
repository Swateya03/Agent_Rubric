"""
Model Loader for Phase 4 Reward Model Training.

Loads base model with 4-bit quantization, applies LoRA, and provides inference utilities.
Optimized for RTX 3050 4GB VRAM.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import get_peft_model, PeftModel
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentrubric.training.lora_config import ModelConfig, get_lora_config
from agentrubric.logger import get_logger
logger = get_logger(__name__)


def get_vram_usage() -> dict:
    """Get current VRAM usage.

    Returns:
        Dict with allocated_gb, reserved_gb, total_gb
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 3),
    }


def load_base_model(model_cfg: ModelConfig):
    """Load base model with dtype consistency (avoids 4-bit quantization dtype issues on Windows).

    Args:
        model_cfg: ModelConfig instance

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading %s in float32", model_cfg.model_name)
    vram_before = get_vram_usage()
    logger.info("VRAM before: %.3f GB allocated", vram_before["allocated_gb"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-padding for sequence classification

    # Load model with sequence classification head in float32
    # Avoids 4-bit quantization dtype conflicts on Windows with PEFT
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.model_name,
        num_labels=1,  # single scalar reward output
        torch_dtype=torch.float32,
        device_map="cpu",  # Load to CPU first to avoid device_map dtype issues
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Model moved to GPU")

    vram_after = get_vram_usage()
    logger.info("VRAM after:  %.3f GB allocated", vram_after["allocated_gb"])
    logger.info("Model loaded successfully")

    return model, tokenizer


def apply_lora(model, model_cfg: ModelConfig):
    """Apply LoRA to model.

    Args:
        model: Base model
        model_cfg: ModelConfig instance

    Returns:
        Model with LoRA applied
    """
    lora_cfg = get_lora_config(model_cfg)
    model = get_peft_model(model, lora_cfg)

    # Log trainable parameter summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    pct = 100 * trainable_params / total_params

    logger.info("Trainable params: %d / %d (%.2f%%)", trainable_params, total_params, pct)
    vram_after_lora = get_vram_usage()
    logger.info("VRAM after LoRA: %.3f GB", vram_after_lora["allocated_gb"])

    return model


def load_for_inference(checkpoint_dir: str, model_cfg: ModelConfig = None):
    """Load model from checkpoint for inference.

    Args:
        checkpoint_dir: Path to LoRA checkpoint directory
        model_cfg: Optional ModelConfig instance

    Returns:
        Tuple of (model, tokenizer)
    """
    model_cfg = model_cfg or ModelConfig()
    model, tokenizer = load_base_model(model_cfg)
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_dir)
    return model, tokenizer


if __name__ == "__main__":
    from training.lora_config import print_lora_summary

    print("=" * 70)
    print("Model Loader Smoke Test")
    print("=" * 70)
    print()

    # Load config
    model_cfg = ModelConfig()
    print(f"Model: {model_cfg.model_name}")
    print(f"Is small model: {model_cfg.is_small_model()}")
    print()

    # Show LoRA config
    lora_cfg = get_lora_config(model_cfg)
    print_lora_summary(lora_cfg)
    print()

    # Load model
    print("Loading model (this will download ~1-3GB on first run)...")
    print()
    model, tokenizer = load_base_model(model_cfg)
    print()

    # Apply LoRA
    model = apply_lora(model, model_cfg)
    print()

    # Test forward pass
    print("Testing forward pass with dummy input...")
    test_text = "This is a test rubric for scoring response quality."
    inputs = tokenizer(
        test_text, return_tensors="pt", truncation=True, max_length=64
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Forward pass output shape: {outputs.logits.shape}")
    reward_scalar = outputs.logits[0].item()
    print(f"Reward scalar: {reward_scalar:.4f}")

    assert (
        outputs.logits.shape == (1, 1)
    ), f"Expected (1,1), got {outputs.logits.shape}"

    print()
    print("=" * 70)
    print("Model loader smoke test PASSED")
    print("=" * 70)
