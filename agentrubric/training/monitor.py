"""
Training Monitor — Parse trainer_state.json and generate diagnostic plots.

Designed for Phase 4 reward model training on Windows RTX 3050 4GB.
Analyzes loss curves, learning rate, and reward accuracy metrics.
"""

import json
import os
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_trainer_state(checkpoint_dir: str) -> dict:
    """Load trainer_state.json from checkpoint directory.

    Searches in order:
      1. {checkpoint_dir}/trainer_state.json
      2. {checkpoint_dir}/final/trainer_state.json
      3. Any subdirectory {checkpoint_dir}/checkpoint-*/trainer_state.json

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Parsed trainer_state.json as dict

    Raises:
        FileNotFoundError: If trainer_state.json not found
    """
    candidates = [
        os.path.join(checkpoint_dir, "trainer_state.json"),
        os.path.join(checkpoint_dir, "final", "trainer_state.json"),
    ]

    # Check direct paths first
    for path in candidates:
        if os.path.exists(path):
            print(f"Found trainer_state.json at: {path}")
            with open(path, "r") as f:
                return json.load(f)

    # Search in checkpoint-* subdirectories
    if os.path.isdir(checkpoint_dir):
        for entry in os.listdir(checkpoint_dir):
            if entry.startswith("checkpoint-"):
                state_path = os.path.join(checkpoint_dir, entry, "trainer_state.json")
                if os.path.exists(state_path):
                    print(f"Found trainer_state.json at: {state_path}")
                    with open(state_path, "r") as f:
                        return json.load(f)

    # Not found
    raise FileNotFoundError(
        f"trainer_state.json not found in {checkpoint_dir}. "
        "Make sure training completed. "
        f"Expected locations: {checkpoint_dir}/trainer_state.json "
        f"or {checkpoint_dir}/final/trainer_state.json"
    )


def extract_metrics(trainer_state: dict) -> dict:
    """Extract metrics from trainer_state log_history.

    Args:
        trainer_state: Parsed trainer_state.json dict

    Returns:
        Dict with extracted metrics:
          - train_steps, train_loss: training loss at each step
          - eval_steps, eval_loss: evaluation loss at each step
          - eval_reward_acc: reward accuracy (if present)
          - learning_rates: learning rate schedule
          - final_train_loss, final_eval_loss: last recorded losses
          - total_steps: global_step count
          - best_eval_loss: best metric recorded
    """
    log_history = trainer_state.get("log_history", [])

    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    eval_reward_acc = []
    learning_rates = []

    for entry in log_history:
        step = entry.get("step")

        # Training step
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])

            lr = entry.get("learning_rate")
            if lr is not None:
                learning_rates.append(lr)

        # Evaluation step
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

            acc = entry.get("eval_reward_accuracy")
            if acc is not None:
                eval_reward_acc.append(acc)

    final_train_loss = train_loss[-1] if train_loss else None
    final_eval_loss = eval_loss[-1] if eval_loss else None

    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "eval_reward_acc": eval_reward_acc,
        "learning_rates": learning_rates,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "total_steps": trainer_state.get("global_step", 0),
        "best_eval_loss": trainer_state.get("best_metric", None),
    }


def detect_issues(metrics: dict) -> list:
    """Detect training issues and warnings.

    Args:
        metrics: Dict from extract_metrics()

    Returns:
        List of warning strings (empty if no issues)
    """
    warnings = []

    # Check 1: loss not decreasing
    train_loss = metrics.get("train_loss", [])
    if len(train_loss) >= 5 and train_loss[-1] >= train_loss[0]:
        warnings.append(
            "WARNING: Training loss did not decrease overall. "
            "Check learning rate (try 1e-5) and data quality."
        )

    # Check 2: loss exploded
    if any(math.isnan(l) or l > 10 for l in train_loss):
        warnings.append(
            "WARNING: Loss exploded (NaN or >10). Reduce learning rate to 5e-6."
        )

    # Check 3: overfitting
    eval_loss = metrics.get("eval_loss", [])
    if len(eval_loss) >= 3:
        if eval_loss[-1] > eval_loss[0] * 1.2:
            warnings.append(
                "WARNING: Eval loss increased (possible overfitting). "
                "Consider fewer epochs or more training data."
            )

    # Check 4: very few steps
    total = metrics.get("total_steps", 0)
    if total < 20:
        warnings.append(
            f"NOTE: Only {total} training steps completed. "
            "Run full training with --epochs 3 for meaningful results. "
            "Quick-test results are for pipeline verification only."
        )

    return warnings


def plot_training_curves(metrics: dict, output_dir: str) -> list:
    """Generate training diagnostic plots.

    Args:
        metrics: Dict from extract_metrics()
        output_dir: Directory to save PNG files

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # Plot 1: Loss curves
    train_steps = metrics.get("train_steps", [])
    train_loss = metrics.get("train_loss", [])
    eval_steps = metrics.get("eval_steps", [])
    eval_loss = metrics.get("eval_loss", [])

    if train_loss or eval_loss:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        plt.style.use("seaborn-v0_8-whitegrid")

        if train_loss:
            ax.plot(
                train_steps,
                train_loss,
                color="#E05C3A",
                linewidth=2,
                label="Train loss",
                marker="o",
                markersize=4,
            )

        if eval_loss:
            ax.plot(
                eval_steps,
                eval_loss,
                color="#1D9E75",
                linewidth=2,
                linestyle="--",
                label="Eval loss",
                marker="s",
                markersize=5,
            )
            ax.axhline(
                y=eval_loss[-1],
                color="gray",
                linestyle=":",
                alpha=0.5,
                label=f"Final eval loss: {eval_loss[-1]:.4f}",
            )

        ax.set_title("Training and evaluation loss", fontsize=13, fontweight="normal")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()

        loss_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()
        saved_paths.append(loss_path)

    # Plot 2: Learning rate schedule
    learning_rates = metrics.get("learning_rates", [])
    if learning_rates:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        plt.style.use("seaborn-v0_8-whitegrid")

        ax.plot(learning_rates, color="#534AB7", linewidth=2, marker="o", markersize=3)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax.set_title("Learning rate schedule", fontsize=13, fontweight="normal")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning rate")
        plt.tight_layout()

        lr_path = os.path.join(output_dir, "lr_schedule.png")
        plt.savefig(lr_path)
        plt.close()
        saved_paths.append(lr_path)

    # Plot 3: Reward accuracy
    eval_reward_acc = metrics.get("eval_reward_acc", [])
    if eval_reward_acc:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        plt.style.use("seaborn-v0_8-whitegrid")

        x_pos = range(len(eval_reward_acc))
        ax.bar(x_pos, eval_reward_acc, color="#534AB7", alpha=0.75)
        ax.axhline(
            y=0.5,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Random baseline (50%)",
        )
        ax.set_ylim(0, 1.05)

        ax.set_title(
            "Reward model accuracy on eval set", fontsize=13, fontweight="normal"
        )
        ax.set_xlabel("Evaluation step")
        ax.set_ylabel("Accuracy")
        ax.legend()
        plt.tight_layout()

        acc_path = os.path.join(output_dir, "reward_accuracy.png")
        plt.savefig(acc_path)
        plt.close()
        saved_paths.append(acc_path)

    return saved_paths


def print_training_summary(metrics: dict, issues: list) -> None:
    """Print formatted training summary.

    Args:
        metrics: Dict from extract_metrics()
        issues: List from detect_issues()
    """
    final_train = metrics.get("final_train_loss")
    final_eval = metrics.get("final_eval_loss")
    best_eval = metrics.get("best_eval_loss")

    train_str = f"{final_train:.4f}" if final_train is not None else "N/A"
    eval_str = f"{final_eval:.4f}" if final_eval is not None else "N/A"
    best_str = f"{best_eval:.4f}" if best_eval is not None else "N/A"

    print("=" * 45)
    print("  Training Summary")
    print("=" * 45)
    print(f"  Total steps       : {metrics.get('total_steps', 0):<24}")
    print(f"  Final train loss  : {train_str:<24}")
    print(f"  Final eval loss   : {eval_str:<24}")
    print(f"  Best eval loss    : {best_str:<24}")
    print("=" * 45)

    if issues:
        print("\nIssues detected:")
        for w in issues:
            print(f"  ! {w}")
    else:
        print("\n[OK] No training issues detected.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor reward model training — analyze loss, accuracy, learning rate"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Where to save plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    try:
        state = load_trainer_state(args.checkpoint)
    except FileNotFoundError as e:
        print(e)
        print("\nTo generate a trainer_state.json, run:")
        print("  python -m agentrubric.training.reward_trainer --quick-test")
        exit(1)

    metrics = extract_metrics(state)
    issues = detect_issues(metrics)
    print_training_summary(metrics, issues)

    saved = plot_training_curves(metrics, args.output)

    if saved:
        print(f"\nPlots saved ({len(saved)} files):")
        for p in saved:
            print(f"  {p}")
    else:
        print("\nNo plots generated — training log may be empty.")
        print("Run full training first:")
        print("  python -m agentrubric.training.reward_trainer --epochs 3")
        print("Then run monitor again:")
        print("  python -m agentrubric.training.monitor")
