"""
Persistence layer for LangGraph using memory checkpointing.

Enables run history to be stored in memory during graph invocations.
Thread IDs provide unique identifiers for each run, allowing state retrieval.

Note: This uses MemorySaver instead of SqliteSaver since SqliteSaver is not
available in langgraph 0.2.x. For persistent storage across process exits,
consider upgrading langgraph or implementing a custom checkpointer.
"""

from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

_checkpointer = None


def get_checkpointer() -> MemorySaver:
    """Get or create a memory checkpointer for the graph.

    Returns:
        MemorySaver configured for in-memory storage
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer
