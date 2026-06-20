# AgentRubric Outputs

This directory contains all outputs from the AgentRubric pipeline phases, organized by stage.

## Directory Structure

```
outputs/
├── phase3/                          # Phase 3: Preference Pair Generation
│   ├── results/
│   │   ├── phase3_results.json      # Summary of all 40 processed samples
│   │   └── preference_pairs.db      # SQLite database with preference pairs
│   └── logs/
│       └── phase3_summary.txt       # Summary statistics
│
├── phase4/                          # Phase 4: Reward Model Training
│   ├── training_logs/
│   │   └── training.log             # Full training output with loss curves
│   ├── checkpoints/
│   │   ├── final/                   # Best trained model checkpoint
│   │   │   ├── adapter_model.safetensors   (8.35 MB - LoRA weights)
│   │   │   ├── adapter_config.json
│   │   │   ├── tokenizer.json
│   │   │   └── ...
│   │   └── checkpoint-3/            # Last epoch checkpoint
│   └── metrics.json                 # Training metrics summary
│
├── evaluation/                      # Phase 4: Evaluation on Code-Preference-Pairs
│   ├── experiment_summary.json      # Final metrics (MAE, accuracy, win-rate)
│   ├── per_sample_results.csv       # Detailed per-pair analysis
│   ├── winrate.json                 # Comparison results
│   └── gold_scores_cache.json       # Cached gold judge scores
│
├── datasets/                        # All datasets used in pipeline
│   ├── phase3/
│   │   ├── input/
│   │   │   └── sample_responses.json      # 40 coding tasks from CodeFeedback
│   │   └── output/
│   │       └── preference_pairs.jsonl     # 39 preference pairs generated
│   ├── phase4/
│   │   ├── train.jsonl              # 31 training pairs for QLoRA
│   │   └── eval.jsonl               # 8 eval pairs for training validation
│   └── benchmarks/
│       └── code_preference_eval.jsonl    # 200-pair benchmark (ground truth)
│
└── README.md                        # This file

```

## Key Files

### Phase 3 Results

**phase3_results.json** (33 KB)
- Contains all 40 processed samples with detailed scores
- Fields: sample_id, task, final_score, gold_score, divergence_score, hack_detected, preference_saved
- Key metric: 97.5% success rate (39/40 preference pairs saved)

**preference_pairs.db** (SQLite database)
- Stores preference pairs as structured records
- Fields: sample_id, task, response, chosen_rubric, rejected_rubric, scores, timestamps
- Query: `SELECT COUNT(*) FROM preference_pairs` → 39 pairs

### Phase 4 Results

**training.log** (full training output)
- Wall-clock time: 347 seconds
- Final loss: 0.4177
- Accuracy: 66.67%
- Mean reward: 2.656

**Trained Model Checkpoint** (outputs/phase4/checkpoints/final/)
- **adapter_model.safetensors** (8.35 MB) - QLoRA weights
- **tokenizer.json** (10.89 MB) - Qwen2 tokenizer
- Ready for inference via PEFT + Transformers

### Evaluation Results

**experiment_summary.json**
```json
{
  "mae_model": 0.1994,
  "mae_baseline": 0.3000,
  "win_rate": 1.0,
  "mean_model_score": 0.9994,
  "mean_gold_score": 0.8000,
  "conclusion": "33.5% MAE improvement"
}
```

**Key Metric: Win-rate 100%**
- Trained model scores all 200 benchmark pairs closer to ground truth
- 33.5% reduction in error vs. static baseline rubric

## Datasets

### Input Datasets

1. **CodeFeedback-Filtered-Instruction** (40 samples)
   - Real coding questions with LLM responses
   - Downloaded from HuggingFace: m-a-p/CodeFeedback-Filtered-Instruction
   - Saved to: `outputs/datasets/phase3/input/sample_responses.json`

2. **Code-Preference-Pairs** (200 samples)
   - Benchmark with unambiguous ground truth
   - chosen = correct code, rejected = code with bugs
   - Downloaded from HuggingFace: Vezora/Code-Preference-Pairs
   - Saved to: `outputs/datasets/benchmarks/code_preference_eval.jsonl`

### Generated Datasets

1. **Preference Pairs** (Phase 3 output)
   - 39 pairs from CodeFeedback processing
   - Format: prompt, chosen, rejected
   - Saved to: `outputs/datasets/phase3/output/preference_pairs.jsonl`

2. **Training Split** (Phase 4 input)
   - 31 training + 8 eval pairs (split from 39 total)
   - Format: prompt, chosen, rejected (TRL RewardTrainer format)
   - Saved to: `outputs/datasets/phase4/{train,eval}.jsonl`

## Usage

### Load Trained Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2-1.5B",
    torch_dtype=torch.float32,
    device_map="cpu"
)

model = PeftModel.from_pretrained(
    base_model,
    "outputs/phase4/checkpoints/final"
)
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/phase4/checkpoints/final"
)

# Score a code snippet
inputs = tokenizer("Your code here", return_tensors="pt")
outputs = model(**inputs)
reward_score = outputs.logits[0].item()
print(f"Reward: {reward_score}")
```

### Access Phase 3 Results

```python
import json

# Load phase 3 summary
with open("outputs/phase3/results/phase3_results.json") as f:
    results = json.load(f)

# Get statistics
total_samples = len(results)
passed = sum(1 for r in results if r["passed"])
hacks_detected = sum(1 for r in results if r["hack_detected"])

print(f"Processed: {total_samples}")
print(f"Passed: {passed} ({100*passed/total_samples:.1f}%)")
print(f"Hacks detected: {hacks_detected}")
```

### Access Evaluation Results

```python
import json

with open("outputs/evaluation/experiment_summary.json") as f:
    metrics = json.load(f)

print(f"MAE (trained): {metrics['mae_model']:.4f}")
print(f"MAE (baseline): {metrics['mae_baseline']:.4f}")
print(f"Win-rate: {metrics['win_rate']:.1%}")
print(f"Improvement: {(1 - metrics['mae_model']/metrics['mae_baseline'])*100:.1f}%")
```

## Statistics Summary

| Metric | Value |
|--------|-------|
| **Phase 3 Input** | 40 samples |
| **Phase 3 Output** | 39 preference pairs |
| **Success Rate** | 97.5% |
| **Hacks Detected** | 0% |
| **Training Time** | 347 seconds |
| **Final Train Loss** | 0.4177 |
| **Accuracy (on pairs)** | 66.67% |
| **Eval MAE (trained)** | 0.1994 |
| **Eval MAE (baseline)** | 0.3000 |
| **MAE Improvement** | 33.5% ↓ |
| **Win-rate** | 100.0% |

## Paper Figures

These outputs support figures for your AAAI submission:

1. **Phase 3 Convergence**
   - Divergence histogram (all samples aligned, none gamed)
   - Iteration count distribution
   - Source: `outputs/phase3/results/phase3_results.json`

2. **Phase 4 Training Curves**
   - Loss decreasing (0.7 → 0.4)
   - Accuracy increasing (50% → 67%)
   - Margin improving (good separation)
   - Source: `outputs/phase4/training_logs/training.log`

3. **Phase 4 Evaluation**
   - MAE comparison (0.1994 vs 0.3000)
   - Win-rate bar chart (100% vs 0%)
   - Per-sample error distribution
   - Source: `outputs/evaluation/experiment_summary.json`

## For Research Publication

All files in this directory are anonymized and ready for:
- ✅ Supplementary materials (appendix figures, tables)
- ✅ Reproducibility appendix (checkpoint + datasets)
- ✅ Code release (with model checkpoints)
- ✅ Results tables (metrics in tables 1–3)

**Citation-ready artifacts:**
- Trained model checkpoint (outputs/phase4/checkpoints/final/)
- All datasets (outputs/datasets/)
- Raw metrics (outputs/evaluation/)
- Training logs (outputs/phase4/training_logs/)

---

Generated: 2026-06-21 | AgentRubric Phase 3–4 Complete
