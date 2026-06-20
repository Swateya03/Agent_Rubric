"""
Dataset Preparation for Phase 4 QLoRA Training.

Loads Phase 3 preference pairs, validates them, cleans them,
and converts to HuggingFace Dataset format for TRL RewardTrainer.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from datasets import Dataset, DatasetDict

from agentrubric.logger import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class PreferenceSample:
    """A single preference pair for reward model training."""

    prompt: str  # "{task}\n\nResponse: {response}"
    chosen: str  # rubric that produced higher quality score
    rejected: str  # rubric that produced lower or hacked score
    source_id: str  # sample_id from Phase 3, for traceability


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file line by line.

    Args:
        path: Path to JSONL file

    Returns:
        List of dicts, one per line

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON parsing fails (includes line number)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON parse error at line {idx}: {str(e)}\n"
                    f"First 80 chars: {line[:80]}"
                )
    return data


def validate_sample(raw: dict, idx: int) -> PreferenceSample | None:
    """Validate a raw preference pair dict.

    Args:
        raw: Raw dict from JSONL
        idx: Line index (for error messages)

    Returns:
        PreferenceSample if valid, None otherwise
    """
    # Check required keys
    if "prompt" not in raw:
        logger.warning("validate_sample item %d: missing 'prompt' key", idx)
        return None
    if "chosen" not in raw:
        logger.warning("validate_sample item %d: missing 'chosen' key", idx)
        return None
    if "rejected" not in raw:
        logger.warning("validate_sample item %d: missing 'rejected' key", idx)
        return None

    prompt = raw["prompt"]
    chosen = raw["chosen"]
    rejected = raw["rejected"]

    # Type checks
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("validate_sample item %d: prompt is not a non-empty string", idx)
        return None
    if not isinstance(chosen, str) or not chosen.strip():
        logger.warning("validate_sample item %d: chosen is not a non-empty string", idx)
        return None
    if not isinstance(rejected, str) or not rejected.strip():
        logger.warning("validate_sample item %d: rejected is not a non-empty string", idx)
        return None

    # Chosen != Rejected
    if chosen == rejected:
        logger.warning("validate_sample item %d: chosen and rejected rubrics are identical", idx)
        return None

    # Length checks
    if len(chosen) < 50:
        logger.warning("validate_sample item %d: chosen rubric too short (<50 chars)", idx)
        return None
    if len(rejected) < 50:
        logger.warning("validate_sample item %d: rejected rubric too short (<50 chars)", idx)
        return None
    if len(prompt) < 20:
        logger.warning("validate_sample item %d: prompt too short (<20 chars)", idx)
        return None

    source_id = raw.get("sample_id", f"item_{idx}")
    return PreferenceSample(
        prompt=prompt.strip(),
        chosen=chosen.strip(),
        rejected=rejected.strip(),
        source_id=source_id,
    )


def build_dataset(
    jsonl_path: str, train_ratio: float = 0.8, seed: int = 42
) -> DatasetDict:
    """Build a HuggingFace Dataset from preference pairs JSONL.

    Args:
        jsonl_path: Path to preference_pairs.jsonl
        train_ratio: Fraction to use for training (rest for eval)
        seed: Random seed for shuffling

    Returns:
        DatasetDict with 'train' and 'eval' splits
    """
    # Load raw data
    raw_data = load_jsonl(jsonl_path)
    logger.info("Loaded %d raw samples from %s", len(raw_data), jsonl_path)

    # Validate
    valid_samples = []
    for idx, raw in enumerate(raw_data, 1):
        sample = validate_sample(raw, idx)
        if sample is not None:
            valid_samples.append(sample)

    n_raw = len(raw_data)
    n_valid = len(valid_samples)
    logger.info("Total loaded   : %d", n_raw)
    logger.info("Valid samples  : %d", n_valid)
    logger.info("Skipped        : %d", n_raw - n_valid)

    # Warning if too few
    if n_valid < 10:
        logger.warning(
            "Only %d valid samples found. "
            "Phase 4 training needs at least 50 pairs for meaningful results. "
            "Run run_pipeline.py on more data first, or use the synthetic fallback.",
            n_valid
        )

    # Convert to HF Dataset
    data_for_hf = [
        {
            "prompt": s.prompt,
            "chosen": s.chosen,
            "rejected": s.rejected,
        }
        for s in valid_samples
    ]
    dataset = Dataset.from_list(data_for_hf)

    # Shuffle
    dataset = dataset.shuffle(seed=seed)

    # Split
    n_train = int(len(dataset) * train_ratio)
    train_split = dataset.select(range(n_train))
    eval_split = dataset.select(range(n_train, len(dataset)))

    return DatasetDict({"train": train_split, "eval": eval_split})


def generate_synthetic_pairs(n: int = 100) -> list[dict]:
    """Generate synthetic preference pairs for testing.

    Args:
        n: Number of pairs to generate

    Returns:
        List of dicts in preference_pairs.jsonl format
    """
    tasks_and_responses = [
        (
            "Explain what artificial intelligence is",
            "AI is the simulation of human intelligence by computer systems. "
            "It involves machine learning, natural language processing, and computer vision. "
            "AI can perform tasks that normally require human intelligence.",
        ),
        (
            "What are the main types of machine learning?",
            "There are three main types: supervised learning (learns from labeled data), "
            "unsupervised learning (finds patterns in unlabeled data), and reinforcement learning "
            "(learns by interacting with an environment). Each has distinct applications.",
        ),
        (
            "Write a function to calculate factorial",
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            "This recursively calculates the factorial of a number.",
        ),
        (
            "Explain photosynthesis in simple terms",
            "Photosynthesis is the process by which plants convert sunlight into chemical energy. "
            "Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. "
            "This is the foundation of most ecosystems.",
        ),
        (
            "What is the difference between DNA and RNA?",
            "DNA (deoxyribonucleic acid) stores genetic information and is double-stranded, "
            "while RNA (ribonucleic acid) transmits genetic information and is usually single-stranded. "
            "RNA is typically temporary, whereas DNA is stable.",
        ),
        (
            "Describe the water cycle",
            "The water cycle consists of evaporation (water turns to vapor), "
            "condensation (vapor cools into clouds), precipitation (water falls as rain/snow), "
            "and collection (water returns to oceans and lakes). This cycle repeats continuously.",
        ),
        (
            "What makes a good software design?",
            "Good software design follows principles like clarity, modularity, "
            "maintainability, and efficiency. Code should be readable, testable, "
            "and easy to extend. Design patterns help achieve these goals.",
        ),
        (
            "Explain blockchain technology",
            "Blockchain is a distributed ledger technology where transactions are grouped into blocks "
            "and linked cryptographically. Each block contains a hash of the previous block, "
            "making tampering difficult. It powers cryptocurrencies and other applications.",
        ),
        (
            "What are the benefits of using version control?",
            "Version control systems track code changes, enable collaboration, "
            "allow reverting to previous versions, and provide a history of modifications. "
            "They are essential for team development and code management.",
        ),
        (
            "Define cloud computing",
            "Cloud computing provides on-demand access to computing resources (servers, storage, databases) "
            "via the internet. Users pay only for what they use. It enables scalability and reduces "
            "the need for physical infrastructure.",
        ),
    ]

    good_rubrics = [
        "# High-Quality Explanation Rubric\n\nCRITERIA:\n1. Accuracy (weight: 0.40)\n"
        "   Is the explanation factually correct and technically sound?\n2. Clarity (weight: 0.30)\n"
        "   Is it easy to understand for the target audience?\n3. Completeness (weight: 0.20)\n"
        "   Does it cover the essential points?\n4. Engagement (weight: 0.10)\n"
        "   Does it hold the reader's interest?",
        "# Technical Excellence Rubric\n\nCRITERIA:\n1. Correctness (weight: 0.35)\n"
        "   Is the solution technically correct?\n2. Code Quality (weight: 0.30)\n"
        "   Is the code clean and well-structured?\n3. Efficiency (weight: 0.20)\n"
        "   Is the solution optimal for time and space?\n4. Documentation (weight: 0.15)\n"
        "   Is the code adequately explained?",
        "# Comprehensive Analysis Rubric\n\nCRITERIA:\n1. Depth of Understanding (weight: 0.40)\n"
        "   Shows thorough comprehension of the topic\n2. Logical Structure (weight: 0.30)\n"
        "   Ideas are presented in a clear, logical order\n3. Evidence (weight: 0.20)\n"
        "   Supports claims with facts or examples\n4. Originality (weight: 0.10)\n"
        "   Provides fresh insights or perspectives",
    ]

    weak_rubrics = [
        "# Basic Rubric\n\nCRITERIA:\n1. Good (weight: 0.5)\n"
        "   Is it good?\n2. Clear (weight: 0.25)\n"
        "   Is it clear?\n3. Complete (weight: 0.25)\n"
        "   Is it complete?",
        "# Simple Rubric\n\nCRITERIA:\n1. Quality (weight: 1.0)\n"
        "   Is the quality good or bad?\n",
        "# Minimal Rubric\n\nCRITERIA:\n1. Acceptable (weight: 1.0)\n"
        "   Does it seem acceptable?",
    ]

    pairs = []
    for i in range(n):
        task, response = tasks_and_responses[i % len(tasks_and_responses)]
        good_rubric = good_rubrics[i % len(good_rubrics)]
        weak_rubric = weak_rubrics[i % len(weak_rubrics)]

        pair = {
            "prompt": f"{task}\n\nResponse: {response}",
            "chosen": good_rubric,
            "rejected": weak_rubric,
        }
        pairs.append(pair)

    return pairs


def save_splits(dataset_dict: DatasetDict, output_dir: str = "data") -> None:
    """Save train and eval splits to JSONL files.

    Args:
        dataset_dict: DatasetDict with 'train' and 'eval' splits
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in dataset_dict["train"]:
            f.write(json.dumps(item) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in dataset_dict["eval"]:
            f.write(json.dumps(item) + "\n")

    logger.info("Train split: %d samples -> %s", len(dataset_dict['train']), train_path)
    logger.info("Eval split:  %d samples -> %s", len(dataset_dict['eval']), eval_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare preference pairs for Phase 4 training"
    )
    parser.add_argument(
        "--input", default=str(DATA_DIR / "preference_pairs.jsonl"), help="Input JSONL path"
    )
    parser.add_argument(
        "--output", default=str(DATA_DIR), help="Output directory for splits"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data instead of loading real pairs",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=100, help="Number of synthetic pairs"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4 Dataset Preparation")
    print("=" * 70)
    print()

    if args.synthetic:
        print(f"Generating {args.n_synthetic} synthetic pairs...")
        pairs = generate_synthetic_pairs(args.n_synthetic)
        tmp_path = DATA_DIR / "synthetic_pairs.jsonl"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Saved to {tmp_path}")
        print()
        dataset = build_dataset(tmp_path)
    else:
        dataset = build_dataset(args.input)

    save_splits(dataset, args.output)

    print()
    print("Dataset summary:")
    print(dataset)
    print()
    print("First training example:")
    print(f"  prompt (first 100 chars): {dataset['train'][0]['prompt'][:100]}")
    print(f"  chosen (first 80 chars):  {dataset['train'][0]['chosen'][:80]}")
    print(
        f"  rejected (first 80 chars): {dataset['train'][0]['rejected'][:80]}"
    )
    print()
    print("=" * 70)
