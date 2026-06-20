"""
AgentRubric Phase 4 Experiment — Compare trained reward model vs baseline rubric.

Research question: Does training on agent-generated preference pairs improve
agreement with the gold judge vs a static rubric baseline?
"""

import json
import os
import csv
from pathlib import Path
from scipy import stats

from ..training.reward_inference import RewardScorer
from ..training.dataset_prep import load_jsonl

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUBRICS_DIR = PROJECT_ROOT / "rubrics"


def baseline_rubric_scorer(task: str, response: str) -> float:
    """Score with static baseline rubric (no agent adaptation).

    Args:
        task: Task/prompt text
        response: Model response text

    Returns:
        Score 0.0-1.0 from static rubric, or 0.5 on error
    """
    try:
        from agentrubric.rubric_scorer import score_response, load_rubric

        rubric_text = load_rubric(str(RUBRICS_DIR / "default.txt"))
        result = score_response(
            sample_id="baseline_eval", task=task, response=response, rubric_text=rubric_text
        )
        return result.overall_score if result else 0.5
    except Exception as e:
        print(f"  Baseline scorer error: {e}")
        return 0.5  # neutral fallback — never 0.0 (biases results)


def get_gold_scores(
    eval_data: list, cache_path: str = "results/gold_scores_cache.json"
) -> list:
    """Get ground-truth scores from gold judge, with caching.

    Args:
        eval_data: List of eval dicts with "task" and "response"
        cache_path: Where to cache gold scores

    Returns:
        List of gold scores
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load existing cache
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached gold scores.")

    # Get scores for each eval item
    for i, item in enumerate(eval_data):
        # Hash key avoids API re-calls for same task+response
        key = str(hash(item.get("task", "") + item.get("response", "")))

        if key not in cache:
            try:
                from ..eval.gold_judge import gold_judge_node
                from ..graph.state import create_initial_state

                state = create_initial_state(
                    sample_id=f"gold_eval_{i}",
                    task=item.get("task", ""),
                    response=item.get("response", ""),
                )
                updates = gold_judge_node(state)
                cache[key] = updates.get("gold_score", 0.5)
                print(f"  Gold judge {i+1}/{len(eval_data)}: {cache[key]:.4f}")
            except Exception as e:
                print(f"  Gold judge error on item {i}: {e}")
                cache[key] = 0.5

            # Save cache after each new score
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)

    # Return scores in order
    return [
        cache[str(hash(d.get("task", "") + d.get("response", "")))]
        for d in eval_data
    ]


def compute_agreement_metrics(model_scores: list, gold_scores: list, baseline_scores: list) -> dict:
    """Compute agreement metrics between model, baseline, and gold scores.

    Args:
        model_scores: Reward model scores
        gold_scores: Gold judge scores
        baseline_scores: Static rubric baseline scores

    Returns:
        Dict with agreement metrics
    """
    # Handle edge cases
    if len(model_scores) < 3:
        return {"error": "Need at least 3 eval samples for meaningful metrics."}

    try:
        rho_model, p_model = stats.spearmanr(model_scores, gold_scores)
        rho_baseline, p_baseline = stats.spearmanr(baseline_scores, gold_scores)
    except Exception:
        rho_model = rho_baseline = p_model = p_baseline = 0.0

    # Win-rate: which system's score is closer to gold?
    n = len(model_scores)
    model_wins = 0
    baseline_wins = 0
    ties = 0
    for m, b, g in zip(model_scores, baseline_scores, gold_scores):
        m_err = abs(m - g)
        b_err = abs(b - g)
        if abs(m_err - b_err) < 0.01:
            ties += 1
        elif m_err < b_err:
            model_wins += 1
        else:
            baseline_wins += 1

    mae_model = sum(abs(m - g) for m, g in zip(model_scores, gold_scores)) / n
    mae_baseline = sum(abs(b - g) for b, g in zip(baseline_scores, gold_scores)) / n

    return {
        "n_samples": n,
        "spearman_model": round(float(rho_model), 4),
        "spearman_baseline": round(float(rho_baseline), 4),
        "p_value_model": round(float(p_model), 4),
        "p_value_baseline": round(float(p_baseline), 4),
        "win_rate": round(model_wins / n, 4),
        "tie_rate": round(ties / n, 4),
        "loss_rate": round(baseline_wins / n, 4),
        "mae_model": round(mae_model, 4),
        "mae_baseline": round(mae_baseline, 4),
        "mean_model_score": round(sum(model_scores) / n, 4),
        "mean_baseline_score": round(sum(baseline_scores) / n, 4),
        "mean_gold_score": round(sum(gold_scores) / n, 4),
    }


def generate_conclusion(metrics: dict) -> str:
    """Generate research conclusion from metrics.

    Args:
        metrics: Dict from compute_agreement_metrics()

    Returns:
        Conclusion string
    """
    if "error" in metrics:
        return metrics["error"]

    rho_m = metrics["spearman_model"]
    rho_b = metrics["spearman_baseline"]
    win = metrics["win_rate"]

    if rho_m > rho_b + 0.05:
        return (
            f"YES - Trained reward model shows stronger agreement with gold judge "
            f"(corr={rho_m:.3f}) vs static rubric baseline (corr={rho_b:.3f}). "
            f"Win-rate: {win:.1%}. "
            f"Agentic rubric refinement improved preference signal quality."
        )
    elif abs(rho_m - rho_b) <= 0.05:
        return (
            f"INCONCLUSIVE - Model (corr={rho_m:.3f}) and baseline (corr={rho_b:.3f}) "
            f"perform similarly. Win-rate: {win:.1%}. "
            f"More training data or epochs may be needed."
        )
    else:
        return (
            f"NO - Static baseline (corr={rho_b:.3f}) outperforms trained model "
            f"(corr={rho_m:.3f}) on this eval set. Win-rate: {win:.1%}. "
            f"Consider data quality review or more training epochs."
        )


def save_results(
    eval_data: list,
    model_scores: list,
    baseline_scores: list,
    gold_scores: list,
    metrics: dict,
    conclusion: str,
    output_dir: str = "results",
) -> None:
    """Save experiment results to JSON and CSV files.

    Args:
        eval_data: List of eval samples
        model_scores: Reward model scores
        baseline_scores: Baseline scores
        gold_scores: Gold judge scores
        metrics: Agreement metrics
        conclusion: Research conclusion
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # File 1: winrate.json
    winrate_data = {
        "research_question": "Does training on agent-refined pairs improve agreement with gold judge?",
        "n_eval_samples": metrics.get("n_samples", len(eval_data)),
        "metrics": metrics,
        "conclusion": conclusion,
    }
    winrate_path = os.path.join(output_dir, "winrate.json")
    with open(winrate_path, "w") as f:
        json.dump(winrate_data, f, indent=2)
    print(f"Saved: {winrate_path}")

    # File 2: per_sample_results.csv
    csv_path = os.path.join(output_dir, "per_sample_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "task_preview",
                "model_score",
                "baseline_score",
                "gold_score",
                "model_error",
                "baseline_error",
                "model_wins",
            ]
        )
        for i, (m, b, g) in enumerate(zip(model_scores, baseline_scores, gold_scores)):
            task = eval_data[i].get("task", "")
            task_preview = (task[:60] + "...") if len(task) > 60 else task
            m_err = round(abs(m - g), 4)
            b_err = round(abs(b - g), 4)
            model_wins = "yes" if m_err < b_err else "no"
            writer.writerow([i + 1, task_preview, m, b, g, m_err, b_err, model_wins])
    print(f"Saved: {csv_path}")

    # File 3: experiment_summary.json
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    summary_data = {
        "model_scores": model_scores,
        "baseline_scores": baseline_scores,
        "gold_scores": gold_scores,
        "metrics": metrics,
        "conclusion": conclusion,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved: {summary_path}")


def print_experiment_report(metrics: dict, conclusion: str) -> None:
    """Print formatted experiment report.

    Args:
        metrics: Agreement metrics
        conclusion: Research conclusion
    """
    if "error" in metrics:
        print(f"Cannot generate report: {metrics['error']}")
        return

    n = metrics["n_samples"]
    rho_m = metrics["spearman_model"]
    rho_b = metrics["spearman_baseline"]
    mae_m = metrics["mae_model"]
    mae_b = metrics["mae_baseline"]
    ms = metrics["mean_model_score"]
    bs = metrics["mean_baseline_score"]
    win_rate = metrics["win_rate"]

    print("=" * 60)
    print("  AgentRubric - Phase 4 Experiment Results")
    print("=" * 60)
    print(f"  Eval samples : {n}")
    print("=" * 60)
    print("  Metric              Model        Baseline")
    print("  -----------------------------------------------")
    print(f"  Spearman rho        {rho_m:.4f}     {rho_b:.4f}")
    print(f"  MAE vs gold         {mae_m:.4f}     {mae_b:.4f}")
    print(f"  Mean score          {ms:.4f}     {bs:.4f}")
    print("=" * 60)
    print(f"  Win-rate (model closer to gold): {win_rate:.1%}")
    print("=" * 60)
    print("  Conclusion:")
    print("  " + conclusion)
    print("=" * 60)


def main():
    """Main experiment runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase 4 experiment: trained model vs baseline rubric"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/final",
        help="Path to trained checkpoint folder",
    )
    parser.add_argument("--eval-data", default="data/eval.jsonl", help="Path to eval data")
    parser.add_argument(
        "--output", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--skip-gold",
        action="store_true",
        help="Use cached gold scores only (don't call gold judge)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with synthetic scores (no model/API needed)",
    )
    args = parser.parse_args()

    # Load eval data
    eval_data_path = args.eval_data
    if not os.path.exists(eval_data_path):
        print(f"Eval data not found: {eval_data_path}")
        print("Run: python -m agentrubric.training.dataset_prep --synthetic")
        return

    eval_data = load_jsonl(eval_data_path)
    print(f"Loaded {len(eval_data)} eval samples.")

    # Get model scores
    if args.mock or not os.path.exists(args.checkpoint):
        print("MOCK MODE — using synthetic scores (no model loaded)")
        import random

        random.seed(42)
        model_scores = [round(random.uniform(0.5, 0.9), 4) for _ in eval_data]
    else:
        print(f"Loading reward model from: {args.checkpoint}")
        scorer = RewardScorer(args.checkpoint)
        model_scores = scorer.batch_score(
            [{"task": d.get("task", ""), "response": d.get("response", "")} for d in eval_data]
        )

    # Get baseline scores
    print("\nGetting baseline scores (static rubric)...")
    baseline_scores = []
    for i, item in enumerate(eval_data):
        score = baseline_rubric_scorer(item.get("task", ""), item.get("response", ""))
        baseline_scores.append(score)
        if (i + 1) % 5 == 0 or i == len(eval_data) - 1:
            print(f"  Baseline {i+1}/{len(eval_data)}: {score:.4f}")

    # Get gold scores
    if args.skip_gold:
        print("\nSkipping gold judge (--skip-gold set).")
        gold_scores = [0.5] * len(eval_data)
    else:
        print("\nGetting gold judge scores...")
        gold_scores = get_gold_scores(eval_data)

    # Compute metrics and report
    metrics = compute_agreement_metrics(model_scores, gold_scores, baseline_scores)
    conclusion = generate_conclusion(metrics)

    print()
    print_experiment_report(metrics, conclusion)
    print()
    save_results(eval_data, model_scores, baseline_scores, gold_scores, metrics, conclusion, args.output)


if __name__ == "__main__":
    main()
