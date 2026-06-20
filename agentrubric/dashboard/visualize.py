"""
AgentRubric Dashboard — Generate self-contained HTML report.

Creates a single HTML file with embedded charts (no server needed).
Opens directly in any browser.
"""

import json
import os
import base64
import io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def safe_load_json(path: str) -> dict | None:
    """Safely load JSON file. Returns None if missing or parse error."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return None


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def make_loss_chart(trainer_state_path: str) -> str | None:
    """Generate loss curve chart from trainer_state.json."""
    if not trainer_state_path or not os.path.exists(trainer_state_path):
        return None

    state = safe_load_json(trainer_state_path)
    if not state:
        return None

    try:
        from ..training.monitor import extract_metrics

        metrics = extract_metrics(state)
    except Exception:
        return None

    train_steps = metrics.get("train_steps", [])
    train_loss = metrics.get("train_loss", [])
    eval_steps = metrics.get("eval_steps", [])
    eval_loss = metrics.get("eval_loss", [])

    if not train_loss and not eval_loss:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    if train_loss:
        ax.plot(
            train_steps,
            train_loss,
            color="#E05C3A",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Train loss",
        )
    if eval_loss:
        ax.plot(
            eval_steps,
            eval_loss,
            color="#1D9E75",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=5,
            label="Eval loss",
        )
        ax.axhline(
            y=eval_loss[-1],
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Final eval: {eval_loss[-1]:.4f}",
        )

    ax.set_title("Training and evaluation loss", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)


def make_winrate_chart(winrate_path: str) -> str | None:
    """Generate Spearman correlation comparison chart."""
    if not os.path.exists(winrate_path):
        return None

    data = safe_load_json(winrate_path)
    if not data or "metrics" not in data:
        return None

    metrics = data["metrics"]
    if "error" in metrics:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = ["Reward model", "Static baseline"]
    raw_values = [metrics.get("spearman_model", 0), metrics.get("spearman_baseline", 0)]

    # Handle NaN values
    import math
    values = [v if not math.isnan(v) and v is not None else 0.0 for v in raw_values]
    colors = ["#E05C3A", "#888780"]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.4, edgecolor="white")

    # Label above each bar
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    max_val = max(values) if any(v > 0 for v in values) else 0.1
    ax.set_ylim(0, max(max_val * 1.3, 0.1))
    ax.set_ylabel("Spearman rho with gold judge")
    ax.set_title("Agreement with gold judge (Spearman rho, higher is better)")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_score_distribution_chart(summary_path: str) -> str | None:
    """Generate score distribution histogram."""
    if not os.path.exists(summary_path):
        return None

    data = safe_load_json(summary_path)
    if not data:
        return None

    model_s = data.get("model_scores", [])
    baseline_s = data.get("baseline_scores", [])
    gold_s = data.get("gold_scores", [])

    if not any([model_s, baseline_s, gold_s]):
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = 20

    if model_s:
        ax.hist(
            model_s,
            bins=bins,
            range=(0, 1),
            alpha=0.6,
            color="#E05C3A",
            label="Reward model",
        )
    if baseline_s:
        ax.hist(
            baseline_s,
            bins=bins,
            range=(0, 1),
            alpha=0.6,
            color="#534AB7",
            label="Static baseline",
        )
    if gold_s:
        ax.hist(
            gold_s,
            bins=bins,
            range=(0, 1),
            alpha=0.6,
            color="#1D9E75",
            label="Gold judge",
        )

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score distributions - model vs baseline vs gold")
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)


def build_sample_table_html(csv_path: str) -> str:
    """Generate HTML table from per_sample_results.csv."""
    if not os.path.exists(csv_path):
        return "<p><em>Per-sample results not available. Run experiments/run_experiment.py first.</em></p>"

    import csv as csv_mod

    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
    except Exception:
        return "<p><em>Error reading CSV file.</em></p>"

    if not rows:
        return "<p><em>No rows in results CSV.</em></p>"

    display_rows = rows[:20]
    extra = len(rows) - 20 if len(rows) > 20 else 0

    thead = """<thead><tr>
    <th>#</th><th>Task preview</th>
    <th>Model</th><th>Baseline</th><th>Gold</th><th>Winner</th>
  </tr></thead>"""

    tbody_rows = []
    for i, row in enumerate(display_rows):
        wins = row.get("model_wins", "no") == "yes"
        bg = "#f0f9f5" if wins else "#fdf0ed"
        winner = "[+] model" if wins else "[-] baseline"
        tbody_rows.append(
            f'<tr style="background:{bg}">'
            f'<td>{row.get("idx","")}</td>'
            f'<td style="font-size:12px">{row.get("task_preview","")}</td>'
            f'<td>{row.get("model_score","")}</td>'
            f'<td>{row.get("baseline_score","")}</td>'
            f'<td>{row.get("gold_score","")}</td>'
            f'<td>{winner}</td>'
            f'</tr>'
        )

    extra_row = (
        f'<tr><td colspan="6" style="text-align:center;color:#888;font-size:12px">...and {extra} more rows</td></tr>'
        if extra > 0
        else ""
    )

    return f"<table>{thead}<tbody>{''.join(tbody_rows)}{extra_row}</tbody></table>"


def generate_report(
    checkpoint_dir: str = "checkpoints/final",
    results_dir: str = "results",
    output_path: str = "results/report.html",
) -> str:
    """Generate complete HTML report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Locate trainer_state.json
    trainer_state_candidates = [
        os.path.join(checkpoint_dir, "trainer_state.json"),
        os.path.join(os.path.dirname(checkpoint_dir), "trainer_state.json"),
    ]
    trainer_state_path = next((p for p in trainer_state_candidates if os.path.exists(p)), None)

    # Other file paths
    winrate_path = os.path.join(results_dir, "winrate.json")
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    csv_path = os.path.join(results_dir, "per_sample_results.csv")
    train_cfg_path = os.path.join(checkpoint_dir, "training_config.json")

    # Generate charts (None if data missing)
    loss_chart = make_loss_chart(trainer_state_path) if trainer_state_path else None
    winrate_chart = make_winrate_chart(winrate_path)
    dist_chart = make_score_distribution_chart(summary_path)
    table_html = build_sample_table_html(csv_path)

    # Load summary data
    winrate_data = safe_load_json(winrate_path) or {}
    metrics = winrate_data.get("metrics", {})
    conclusion = winrate_data.get("conclusion", "Experiment not yet run.")
    train_cfg = safe_load_json(train_cfg_path) or {}

    # Helper to render a chart or placeholder
    def chart_html(b64, title):
        if b64:
            return f'<img src="data:image/png;base64,{b64}" alt="{title}">'
        return f'<div class="unavailable">Chart: {title} - run training first</div>'

    # Metric card helper
    def metric_card(label, model_val, baseline_val):
        mv = f"{model_val:.4f}" if isinstance(model_val, float) else "N/A"
        bv = f"{baseline_val:.4f}" if isinstance(baseline_val, float) else "N/A"
        return f"""<div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-row">
            <span class="metric-model">{mv}</span>
            <span class="metric-sep">vs</span>
            <span class="metric-base">{bv}</span>
          </div>
        </div>"""

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AgentRubric - Phase 4 Results</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 960px;
          margin: 40px auto; padding: 0 24px; color: #1a1a1a; line-height: 1.6; }}
  h1 {{ font-size: 22px; font-weight: 500; border-bottom: 1px solid #e8e8e8;
        padding-bottom: 10px; margin-bottom: 6px; }}
  h2 {{ font-size: 15px; font-weight: 500; margin-top: 2rem; color: #222; }}
  .subtitle {{ font-size: 13px; color: #666; margin-bottom: 1.5rem; }}
  .research-q {{ background: #f5f5f5; border-left: 3px solid #534AB7;
                  padding: 12px 16px; border-radius: 0 8px 8px 0;
                  font-size: 13px; font-style: italic; margin: 1rem 0; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
                   gap: 12px; margin: 1rem 0; }}
  .metric-card {{ background: #f8f8f8; border-radius: 8px; padding: 12px 16px; }}
  .metric-label {{ font-size: 11px; color: #666; margin-bottom: 6px; }}
  .metric-row {{ display: flex; align-items: center; gap: 8px; }}
  .metric-model {{ font-size: 18px; font-weight: 500; color: #E05C3A; }}
  .metric-base {{ font-size: 18px; font-weight: 500; color: #888; }}
  .metric-sep {{ font-size: 11px; color: #aaa; }}
  .conclusion {{ background: #f0f9f5; border-left: 3px solid #1D9E75;
                  padding: 12px 16px; border-radius: 0 8px 8px 0;
                  font-size: 13px; margin: 1rem 0; }}
  .winrate-badge {{ display: inline-block; background: #E05C3A; color: white;
                     font-size: 13px; padding: 4px 12px; border-radius: 20px;
                     font-weight: 500; margin: 0.5rem 0; }}
  img {{ width: 100%; border-radius: 8px; margin: 8px 0;
          border: 1px solid #f0f0f0; }}
  .unavailable {{ background: #f8f8f8; border: 1px dashed #ddd;
                   border-radius: 8px; padding: 24px; text-align: center;
                   color: #999; font-size: 13px; margin: 8px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th {{ background: #f0f0f0; padding: 8px 10px; text-align: left;
        font-weight: 500; font-size: 12px; }}
  td {{ padding: 7px 10px; border-top: 1px solid #f5f5f5; }}
  pre {{ background: #f5f5f5; border-radius: 8px; padding: 14px;
          font-size: 12px; overflow-x: auto; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
             font-size: 12px; color: #999; }}
</style>
</head>
<body>

<h1>AgentRubric - Phase 4 Results</h1>
<p class="subtitle">QLoRA reward model trained on agent-generated preference pairs</p>

<h2>Research question</h2>
<div class="research-q">
  Does training on agent-refined preference pairs improve agreement with gold judge vs static rubric baseline?
</div>

<h2>Key metrics</h2>
<div class="metric-grid">
  {metric_card("Spearman rho (up better)",
               metrics.get("spearman_model"),
               metrics.get("spearman_baseline"))}
  {metric_card("MAE vs gold (down better)",
               metrics.get("mae_model"),
               metrics.get("mae_baseline"))}
  {metric_card("Mean score",
               metrics.get("mean_model_score"),
               metrics.get("mean_baseline_score"))}
</div>
<div>
  <span class="winrate-badge">
    Win-rate: {f"{metrics['win_rate']:.1%}" if 'win_rate' in metrics else 'N/A'}
  </span>
</div>

<h2>Conclusion</h2>
<div class="conclusion">{conclusion}</div>

<h2>Training loss</h2>
{chart_html(loss_chart, "Loss curve")}

<h2>Agreement with gold judge</h2>
{chart_html(winrate_chart, "Win-rate chart")}

<h2>Score distributions</h2>
{chart_html(dist_chart, "Score distributions")}

<h2>Per-sample results</h2>
{table_html}

<h2>Training configuration</h2>
<pre>{json.dumps(train_cfg, indent=2) if train_cfg else "training_config.json not found"}</pre>

<footer>
  Generated by AgentRubric - dashboard/visualize.py<br>
  Hardware: Windows - RTX 3050 4GB - float32 - adamw_torch
</footer>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = round(os.path.getsize(output_path) / 1024, 1)
    print(f"Report saved: {output_path}")
    print(f"File size: {size_kb} KB")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate AgentRubric HTML report")
    parser.add_argument(
        "--checkpoint", default="checkpoints/final", help="Checkpoint dir"
    )
    parser.add_argument(
        "--results", default="results", help="Results dir"
    )
    parser.add_argument(
        "--output",
        default="results/report.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    generate_report(args.checkpoint, args.results, args.output)
