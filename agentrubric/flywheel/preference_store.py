"""
Preference Store — SQLite storage for preference pairs (chosen vs rejected rubrics).

These pairs are the training data for Phase 4 reward model fine-tuning.
Each pair records which rubric produced a better, non-hacked score for a
given (task, response) combination.

Pair structure:
  chosen_rubric  — rubric that produced the higher-quality, non-hacked score
  rejected_rubric — rubric that scored lower or produced a hacked score
  prompt         — the task + response being evaluated

The export format (prompt / chosen / rejected JSONL) is compatible with
TRL RewardTrainer, which trains a scalar reward head on top of a base LLM
using pairwise preference loss — NOT DPO. DPO fine-tunes a policy model
directly; RewardTrainer trains a separate reward model. These are different
algorithms that happen to share the same data format.

The flywheel loop:
  agent pipeline generates pairs → stored here → exported to JSONL
  → Phase 4 QLoRA training → trained reward model
"""

import sqlite3
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from agentrubric.graph.state import AgentState
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass
class PreferencePair:
    """A single preference pair: chosen rubric vs rejected rubric for a response."""

    sample_id: str
    """Sample identifier."""

    task: str
    """The task/question given to the model."""

    response: str
    """The model's response."""

    chosen_rubric: str
    """Rubric that scored higher or produced non-hacked score."""

    rejected_rubric: str
    """Rubric that scored lower or produced hacked score."""

    chosen_score: float
    """Score from the chosen rubric."""

    rejected_score: float
    """Score from the rejected rubric."""

    gold_score: float
    """Score from the gold judge (ground truth)."""

    divergence_score: float
    """Divergence between proxy and gold score."""

    created_at: str
    """ISO timestamp when pair was created."""


class PreferenceStore:
    """SQLite storage for preference pairs."""

    DB_PATH = OUTPUTS_DIR / "preference_pairs.db"

    def __init__(self, db_path: str | Path = None):
        """Initialize preference store with SQLite database.

        Args:
            db_path: Path to SQLite database file. Defaults to DB_PATH.
        """
        if db_path is None:
            db_path = self.DB_PATH
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create preference_pairs table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preference_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    response TEXT NOT NULL,
                    chosen_rubric TEXT NOT NULL,
                    rejected_rubric TEXT NOT NULL,
                    chosen_score REAL NOT NULL,
                    rejected_score REAL NOT NULL,
                    gold_score REAL,
                    divergence_score REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, pair: PreferencePair) -> int:
        """Save a preference pair to the database.

        Args:
            pair: PreferencePair object to save

        Returns:
            The row id of the inserted pair
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO preference_pairs
                (sample_id, task, response, chosen_rubric, rejected_rubric,
                 chosen_score, rejected_score, gold_score, divergence_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair.sample_id,
                    pair.task,
                    pair.response,
                    pair.chosen_rubric,
                    pair.rejected_rubric,
                    pair.chosen_score,
                    pair.rejected_score,
                    pair.gold_score,
                    pair.divergence_score,
                    pair.created_at,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def load_all(self) -> list[PreferencePair]:
        """Load all preference pairs from the database.

        Returns:
            List of PreferencePair objects
        """
        pairs = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM preference_pairs")
            for row in cursor.fetchall():
                pair = PreferencePair(
                    sample_id=row[1],
                    task=row[2],
                    response=row[3],
                    chosen_rubric=row[4],
                    rejected_rubric=row[5],
                    chosen_score=row[6],
                    rejected_score=row[7],
                    gold_score=row[8],
                    divergence_score=row[9],
                    created_at=row[10],
                )
                pairs.append(pair)
        return pairs

    def count(self) -> int:
        """Get total number of preference pairs in the database.

        Returns:
            Number of pairs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM preference_pairs")
            return cursor.fetchone()[0]

    def export_jsonl(self, output_path: str) -> int:
        """Export all preference pairs to JSONL format.

        Format is compatible with TRL RewardTrainer (reward modeling format):
        {
          "prompt": "{task}\\n\\nResponse: {response}",
          "chosen": "{chosen_rubric}",
          "rejected": "{rejected_rubric}"
        }

        Args:
            output_path: Path to write JSONL file

        Returns:
            Number of lines written
        """
        pairs = self.load_all()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                line = {
                    "prompt": f"{pair.task}\n\nResponse: {pair.response}",
                    "chosen": pair.chosen_rubric,
                    "rejected": pair.rejected_rubric,
                }
                f.write(json.dumps(line) + "\n")

        return len(pairs)


def preference_store_node(state: AgentState) -> dict:
    """Save a preference pair from the current state.

    This is a LangGraph node function. It reads scores and rubrics from state
    and saves them to the preference store if valid.

    Args:
        state: The shared AgentState

    Returns:
        Dict with updated state keys
    """
    # Determine chosen vs rejected rubrics
    score_a = state["score_variant_a"] or 0.0
    score_b = state["score_variant_b"] or 0.0

    if score_a >= score_b:
        chosen_rubric = state["rubric_variant_a"]
        rejected_rubric = state["rubric_variant_b"]
        chosen_score = score_a
        rejected_score = score_b
    else:
        chosen_rubric = state["rubric_variant_b"]
        rejected_rubric = state["rubric_variant_a"]
        chosen_score = score_b
        rejected_score = score_a

    # Only save if we have sufficient data
    should_save = (
        state["current_result"] is not None
        and state["gold_score"] is not None
        and chosen_rubric != rejected_rubric
    )

    preference_pair_saved = False
    saved_row_id = None

    if should_save:
        store = PreferenceStore()
        pair = PreferencePair(
            sample_id=state["sample_id"],
            task=state["task"],
            response=state["response"],
            chosen_rubric=chosen_rubric,
            rejected_rubric=rejected_rubric,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            gold_score=state["gold_score"],
            divergence_score=state["divergence_score"] or 0.0,
            created_at=datetime.now().isoformat(),
        )
        saved_row_id = store.save(pair)
        preference_pair_saved = True

    msg = (
        f"preference_store: saved pair #{store.count() if preference_pair_saved else 'N/A'}, "
        f"chosen_score={chosen_score:.4f}, rejected_score={rejected_score:.4f}"
    ) if preference_pair_saved else "preference_store: pair not saved (missing data)"

    return {
        "preference_pair_saved": preference_pair_saved,
        "chosen_rubric": chosen_rubric if preference_pair_saved else "",
        "rejected_rubric": rejected_rubric if preference_pair_saved else "",
        "run_history": [HumanMessage(content=msg)],
    }
