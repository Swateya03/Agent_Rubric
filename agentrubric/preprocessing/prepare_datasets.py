"""Preprocess CodeFeedback and Code-Preference-Pairs datasets for AgentRubric.

Usage:
    python -m agentrubric.preprocessing.prepare_datasets

This downloads:
  1. CodeFeedback-Filtered-Instruction (120 samples) → sample_responses.json for Phase 3
  2. Code-Preference-Pairs (200 samples) → code_preference_eval.jsonl for Phase 4 evaluation
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def prepare_codefeedback(output_path: str = None, num_samples: int = 120):
    """Download and prepare CodeFeedback dataset for Phase 3 pipeline input.

    Maps CodeFeedback items to AgentRubric format:
      instruction → task
      output → response

    Args:
        output_path: Path to save sample_responses.json. Defaults to data/sample_responses.json
        num_samples: How many items to sample (100-150 recommended, default 120)

    Returns:
        Number of samples created
    """
    if output_path is None:
        output_path = str(DATA_DIR / "sample_responses.json")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        return 0

    print(f"\n{'='*60}")
    print(f"PHASE 3: Loading CodeFeedback-Filtered-Instruction")
    print(f"{'='*60}")
    print(f"Downloading {num_samples} samples...")

    try:
        dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")
    except Exception as e:
        print(f"ERROR: Failed to load CodeFeedback dataset: {e}")
        return 0

    if "train" not in dataset:
        print("ERROR: Dataset does not have 'train' split")
        return 0

    train_data = dataset["train"]
    if len(train_data) < num_samples:
        print(f"WARNING: Dataset has only {len(train_data)} items, sampling all available")
        num_samples = len(train_data)

    # Inspect first item to find correct field names
    if len(train_data) > 0:
        first_item = train_data[0]
        available_fields = list(first_item.keys())
        print(f"Available fields in dataset: {available_fields}")

    samples = []
    for i, item in enumerate(train_data.select(range(num_samples))):
        # CodeFeedback fields: query, answer, resource, lang
        task = item.get("query", "")
        response = item.get("answer", "")

        if not task or not response:
            continue

        samples.append({
            "id": f"codefeedback_{i:04d}",
            "task": task,
            "response": response,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(samples, indent=2), encoding="utf-8")

    print(f"[OK] Saved {len(samples)} items to {output_path}")
    print(f"     This is your Phase 3 input data (pipeline will generate preference pairs from this)")
    return len(samples)


def prepare_code_preference_eval(output_path: str = None, num_samples: int = 200):
    """Download and prepare Code-Preference-Pairs evaluation dataset for Phase 4 validation.

    This dataset has unambiguous ground truth:
      chosen = corrected code with comments
      rejected = same code with deliberately inserted bugs

    Filters out items with <im_start> token (spurious correlation artifact identified by RM-R1 researchers).

    Args:
        output_path: Path to save evaluation JSONL. Defaults to data/code_preference_eval.jsonl
        num_samples: How many items to sample (200 recommended, default 200)

    Returns:
        Number of pairs saved (after filtering)
    """
    if output_path is None:
        output_path = str(DATA_DIR / "code_preference_eval.jsonl")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        return 0

    print(f"\n{'='*60}")
    print(f"PHASE 4: Loading Code-Preference-Pairs (benchmark dataset)")
    print(f"{'='*60}")
    print(f"Downloading {num_samples} samples (will filter <im_start> artifacts)...")

    try:
        dataset = load_dataset("Vezora/Code-Preference-Pairs")
    except Exception as e:
        print(f"ERROR: Failed to load Code-Preference-Pairs dataset: {e}")
        return 0

    if "train" not in dataset:
        print("ERROR: Dataset does not have 'train' split")
        return 0

    train_data = dataset["train"]
    if len(train_data) < num_samples:
        print(f"WARNING: Dataset has only {len(train_data)} items, sampling all available")
        num_samples = len(train_data)

    # Inspect first item to find correct field names
    if len(train_data) > 0:
        first_item = train_data[0]
        available_fields = list(first_item.keys())
        print(f"Available fields in dataset: {available_fields}")

    pairs = []
    filtered_count = 0

    for item in train_data.select(range(num_samples)):
        # Code-Preference-Pairs fields: instruction, input, accepted, rejected, ID
        chosen = item.get("accepted", "")
        rejected = item.get("rejected", "")
        prompt = item.get("instruction", "") or item.get("input", "")

        if not chosen or not rejected:
            continue

        # Filter: skip if rejected has <im_start> token (RM-R1 identified artifact)
        if "<im_start>" in rejected:
            filtered_count += 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"[OK] Saved {len(pairs)} pairs to {output_path}")
    print(f"     (Filtered out {filtered_count} items with <im_start> token)")
    print(f"     This is your Phase 4 evaluation benchmark (ground truth: chosen > rejected always)")
    return len(pairs)


def main():
    """Download and prepare both datasets."""
    print("\n" + "="*60)
    print("AgentRubric Dataset Preparation")
    print("="*60)

    # Phase 3: CodeFeedback input (reduced to 40 for Groq free tier)
    phase3_count = prepare_codefeedback(num_samples=40)

    # Phase 4: Code-Preference-Pairs evaluation
    phase4_count = prepare_code_preference_eval(num_samples=200)

    print(f"\n{'='*60}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Phase 3 (input):       {phase3_count} coding task-response pairs")
    print(f"Phase 4 (evaluation):  {phase4_count} preference pairs")
    print(f"\nNext steps:")
    print(f"  1. Run Phase 3 pipeline:")
    print(f"     python agentrubric/run_pipeline.py --data data/sample_responses.json --export-pairs")
    print(f"  2. Train Phase 4 model on generated preference pairs:")
    print(f"     python agentrubric/training/reward_trainer.py")
    print(f"  3. Evaluate on benchmark:")
    print(f"     python agentrubric/experiments/evaluate_on_benchmark.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
