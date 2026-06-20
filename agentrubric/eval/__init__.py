"""
Evaluation — Independent LLM-based ground-truth scoring.

Provides an independent evaluator (gold judge) that scores response quality
without seeing the rubric, creating a ground-truth signal separate from the
rubric-based proxy score. The divergence between proxy and gold scores
detects when rubrics are producing gamed/hacked scores.

Core components:
  gold_judge_node — Independent LLM evaluator scoring without rubric
  parse_gold_output — Robust JSON parsing with error fallbacks
"""
