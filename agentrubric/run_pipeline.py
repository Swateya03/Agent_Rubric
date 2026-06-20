"""
AgentRubric Phase 2 + Phase 3 Pipeline — Main entry point.

Runs all samples through the multi-agent LangGraph system:
  - Phase 2: Rubric evaluation with variant scoring
  - Phase 3: Quality filtering, gold judge scoring, hack detection, preference collection
"""

import json
import argparse
import time
from pathlib import Path
from agentrubric.graph.graph import run_graph, print_graph_result
from agentrubric.flywheel.preference_store import PreferenceStore
from agentrubric.constants import DEFAULT_HACK_THRESHOLD, DEFAULT_MAX_ITERATIONS
from agentrubric.utils import truncate_task
from agentrubric.logger import get_logger, configure_log_level

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _categorise_error(e: Exception) -> str:
    """Categorise an exception for structured error reporting.

    Returns a short string label for the error type so failures
    can be analysed in batch results without reading raw tracebacks.

    Args:
        e: The exception to categorise.

    Returns:
        A short lowercase label string.
    """
    try:
        import groq
        if isinstance(e, groq.RateLimitError):
            return "rate_limit"
        if isinstance(e, groq.AuthenticationError):
            return "auth_error"
    except ImportError:
        pass

    if isinstance(e, FileNotFoundError):
        return "missing_file"
    if isinstance(e, ValueError):
        return "validation_error"
    if isinstance(e, TimeoutError):
        return "timeout"
    if isinstance(e, json.JSONDecodeError):
        return "json_parse_error"
    return "unknown_error"


def load_samples(path: str) -> list[dict]:
    """Load samples from a JSON file.

    Args:
        path: Path to JSON file containing array of samples

    Returns:
        List of sample dicts with id, task, response

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If content is not a list
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    content = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(content, list):
        raise ValueError(f"Expected JSON array, got {type(content).__name__}")

    return content


def _process_single_sample(
    sample: dict,
    hack_threshold: float,
    max_iterations: int,
) -> tuple:
    """Run one sample through the full graph pipeline.

    Separated from run_all_samples() so it can be unit tested
    and called independently without the batch loop overhead.

    Args:
        sample: Dict with keys: id, task, response.
        hack_threshold: Passed through to run_graph().
        max_iterations: Passed through to run_graph().

    Returns:
        Tuple of (final_state, thread_id, elapsed_seconds).

    Raises:
        Any exception from run_graph() — caller handles errors.
    """
    start_time = time.time()
    final_state, thread_id = run_graph(
        sample_id=sample["id"],
        task=sample["task"],
        response=sample["response"],
        hack_threshold=hack_threshold,
        max_iterations=max_iterations,
    )
    elapsed = time.time() - start_time
    return final_state, thread_id, elapsed


def _format_result(
    sample: dict,
    final_state,
    thread_id: str,
    elapsed: float,
) -> dict:
    """Build the result dict from a completed graph state.

    Separated from run_all_samples() so result formatting can be
    tested independently from graph execution.

    Args:
        sample: The original sample dict.
        final_state: The AgentState returned by run_graph().
        thread_id: The thread ID used for this run.
        elapsed: Wall-clock seconds the run took.

    Returns:
        Result dict suitable for JSON serialisation and summary reporting.
    """
    score_a = final_state["score_variant_a"] or 0.0
    score_b = final_state["score_variant_b"] or 0.0
    current_result = final_state["current_result"]

    return {
        "sample_id": sample["id"],
        "task": truncate_task(sample["task"]),
        "score_variant_a": round(score_a, 4),
        "score_variant_b": round(score_b, 4),
        "final_score": round(current_result.overall_score, 4)
            if current_result else 0.0,
        "passed": current_result.passed if current_result else False,
        "winning_variant": "A" if score_a >= score_b else "B",
        "critic_reasoning": final_state["critic_reasoning"],
        "elapsed_seconds": round(elapsed, 2),
        "thread_id": thread_id,
        "gold_score": round(final_state["gold_score"] or 0.0, 4),
        "divergence_score": round(final_state["divergence_score"] or 0.0, 4),
        "hack_detected": final_state["hack_detected"],
        "hack_count": final_state["hack_count"],
        "transcript_quality": round(
            final_state["transcript_quality_score"] or 0.0, 2
        ),
        "transcript_flagged": final_state["transcript_flagged"],
        "preference_saved": final_state["preference_pair_saved"],
    }


def _format_error_result(sample: dict, e: Exception) -> dict:
    """Build a result dict for a failed sample.

    Args:
        sample: The original sample dict.
        e: The exception that caused the failure.

    Returns:
        Result dict with error metadata and zero scores.
    """
    logger.error(
        "Sample %s failed [%s]: %s",
        sample["id"], _categorise_error(e), e
    )
    return {
        "sample_id": sample["id"],
        "task": truncate_task(sample["task"]),
        "score_variant_a": 0.0,
        "score_variant_b": 0.0,
        "final_score": 0.0,
        "passed": False,
        "winning_variant": None,
        "critic_reasoning": "",
        "elapsed_seconds": 0.0,
        "thread_id": "",
        "gold_score": 0.0,
        "divergence_score": 0.0,
        "hack_detected": False,
        "hack_count": 0,
        "transcript_quality": 0.0,
        "transcript_flagged": False,
        "preference_saved": False,
        "error_type": _categorise_error(e),
        "error_message": str(e),
    }


def run_all_samples(
    samples: list[dict],
    verbose: bool = False,
    hack_threshold: float = DEFAULT_HACK_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> list[dict]:
    """Run all samples through the graph pipeline (Phase 2 + Phase 3).

    Args:
        samples: List of sample dicts
        verbose: If True, print detailed results for each sample
        hack_threshold: Divergence threshold above which a hack is flagged
        max_iterations: Maximum hack detection loops before giving up

    Returns:
        List of result dicts with scores and metadata
    """
    results = []
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        sample_id = sample.get("id", f"sample_{i}")
        logger.info("Processing %s (%d/%d)...", sample_id, i, total)

        # Rate limit: wait 15 seconds between samples (Groq free tier: 6000 TPM)
        if i > 1:
            logger.debug("Waiting 15s to respect Groq rate limits...")
            time.sleep(15)

        try:
            final_state, thread_id, elapsed = _process_single_sample(
                sample, hack_threshold, max_iterations
            )
            result = _format_result(sample, final_state, thread_id, elapsed)
            results.append(result)

            if verbose:
                print_graph_result(final_state)
            else:
                logger.info(
                    "  %s: score=%.4f, gold=%.4f, hack=%s, saved=%s",
                    sample_id,
                    result["final_score"],
                    result["gold_score"],
                    result["hack_detected"],
                    result["preference_saved"],
                )

        except Exception as e:
            results.append(_format_error_result(sample, e))

    return results


def print_summary_report(results: list[dict]) -> None:
    """Print a formatted summary report of all results (Phase 2 + Phase 3).

    Args:
        results: List of result dicts from run_all_samples
    """
    total = len(results)
    pass_count = sum(1 for r in results if r["passed"])
    fail_count = total - pass_count
    pass_pct = (pass_count / total * 100) if total > 0 else 0
    avg_score = sum(r["final_score"] for r in results) / total if total > 0 else 0
    avg_time = sum(r["elapsed_seconds"] for r in results) / total if total > 0 else 0

    # Phase 3 stats
    pairs_saved = sum(1 for r in results if r.get("preference_saved", False))
    hacks_detected = sum(1 for r in results if r.get("hack_detected", False))
    avg_divergence = (
        sum(r.get("divergence_score", 0.0) for r in results) / total
        if total > 0
        else 0.0
    )
    flagged_count = sum(1 for r in results if r.get("transcript_flagged", False))

    print()
    print("+" + "=" * 50 + "+")
    print("|  AgentRubric Phase 2 + Phase 3 Summary         |")
    print("+" + "=" * 50 + "+")
    print(f"|  Samples run    : {total:<37}|")
    print(f"|  Passed         : {pass_count} ({pass_pct:.0f}%)".ljust(51) + "|")
    print(f"|  Failed         : {fail_count:<37}|")
    print(f"|  Avg score      : {avg_score:.4f}".ljust(51) + "|")
    print(f"|  Avg time/sample: {avg_time:.1f}s".ljust(51) + "|")
    print("+" + "=" * 50 + "+")
    print("|  Phase 3 Metrics:                              |")
    print(f"|  Preference pairs saved : {pairs_saved:<26}|")
    print(f"|  Hacks detected         : {hacks_detected:<26}|")
    print(f"|  Avg divergence         : {avg_divergence:.4f}".ljust(51) + "|")
    print(f"|  Transcripts flagged    : {flagged_count:<26}|")
    print("+" + "=" * 50 + "+")
    print("| Per-sample breakdown (sorted by score):         |")
    print("+" + "=" * 50 + "+")

    sorted_results = sorted(results, key=lambda r: r["final_score"], reverse=True)

    for result in sorted_results:
        sample_id = result["sample_id"]
        score = result["final_score"]
        status = "[PASS]" if result["passed"] else "[FAIL]"
        variant = result["winning_variant"] or "N/A"
        line = f"| {sample_id:<13} {score:.4f}  {status}  variant {variant} won".ljust(51) + "|"
        print(line)

    print("+" + "=" * 50 + "+")


def main():
    """Parse arguments and run the pipeline (Phase 2 + Phase 3)."""
    parser = argparse.ArgumentParser(
        description="Run AgentRubric Phase 2 + Phase 3 pipeline on sample responses"
    )
    parser.add_argument(
        "--data",
        default=str(DATA_DIR / "sample_responses.json"),
        help="Path to samples JSON file",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "phase3_results.json"),
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--hack-threshold",
        type=float,
        default=DEFAULT_HACK_THRESHOLD,
        help=f"Divergence threshold above which a hack is flagged (default {DEFAULT_HACK_THRESHOLD})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum hack detection loops before giving up (default {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--export-pairs",
        action="store_true",
        help="If set, export preference pairs to JSONL for Phase 4 training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each sample",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except warnings and errors",
    )

    args = parser.parse_args()

    configure_log_level(verbose=args.verbose, quiet=args.quiet)

    logger.info("Loading samples...")
    samples = load_samples(args.data)
    logger.info("Loaded %d samples from %s", len(samples), args.data)

    logger.info("Running pipeline...")
    results = run_all_samples(
        samples,
        verbose=args.verbose,
        hack_threshold=args.hack_threshold,
        max_iterations=args.max_iterations,
    )

    print_summary_report(results)
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")
    print()

    # Export preference pairs if requested
    if args.export_pairs:
        store = PreferenceStore()
        export_path = DATA_DIR / "preference_pairs.jsonl"
        n = store.export_jsonl(str(export_path))
        print(f"Exported {n} preference pairs to {export_path}")
        print("This file is ready for Phase 4 QLoRA training.")
        print()


if __name__ == "__main__":
    main()
