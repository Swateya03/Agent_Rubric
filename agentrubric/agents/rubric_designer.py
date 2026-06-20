"""
Rubric Designer agent — retrieves and adapts rubric templates for specific tasks.

This is the first node in the LangGraph graph. It:
  1. Takes the task from state
  2. Uses BM25 to retrieve the best rubric template
  3. Uses the LLM to lightly adapt the template for the specific task
  4. Writes two variants to state (original and adapted)
"""

from agentrubric.graph.state import AgentState
from agentrubric.graph.conditional_edges import get_retry_context
from agentrubric.config import get_llm
from agentrubric.retrieval.bm25_retriever import RubricRetriever
from langchain_core.messages import HumanMessage
from agentrubric.logger import get_logger

logger = get_logger(__name__)

_retriever = None


def get_retriever() -> RubricRetriever:
    """Get or create the rubric retriever (cached at module level).

    Returns:
        RubricRetriever instance, initialized once and reused across all runs
    """
    global _retriever
    if _retriever is None:
        _retriever = RubricRetriever()
    return _retriever


def build_adaptation_prompt(
    task: str, rubric_template: str, retry_context: str = ""
) -> str:
    """Build a prompt asking the LLM to adapt a rubric template to a task.

    Args:
        task: The original task/question given to the model
        rubric_template: The retrieved rubric template
        retry_context: Optional context from a previous hack detection

    Returns:
        A prompt string for the LLM
    """
    context_section = ""
    if retry_context:
        context_section = f"{retry_context}\n\n"

    return f"""{context_section}You are a rubric expert. Adapt the following rubric template to be more
specific to this task domain, while keeping the same structure.

TASK:
{task}

RUBRIC TEMPLATE:
{rubric_template}

Your job:
1. Keep the same criteria names, weights, and structure
2. Do NOT add or remove criteria
3. Adapt the descriptions to be more specific to "{task}"
4. If the rubric already fits well, return it unchanged

Return ONLY the adapted rubric text, no commentary or explanation."""


def rubric_designer_node(state: AgentState) -> dict:
    """Retrieve and adapt a rubric template for the task.

    This is a LangGraph node function. It reads from state and returns
    a dict of only the keys that changed.

    Args:
        state: The shared AgentState

    Returns:
        Dict with updated state keys
    """
    task = state["task"]

    retriever = get_retriever()

    retrieved_doc = retriever.retrieve_top(task)

    # Get retry context if this is a retry loop
    retry_context = get_retry_context(state)

    prompt = build_adaptation_prompt(task, retrieved_doc.text, retry_context)

    llm = get_llm(temperature=0.2)

    response = llm.invoke([HumanMessage(content=prompt)])
    adapted_rubric_text = response.content

    # Mark in run history if this is a retry
    is_retry = retry_context != ""
    msg = (
        f"rubric_designer: retrieved='{retrieved_doc.name}', "
        f"retry={is_retry}, created 2 variants"
    )
    logger.info(msg)

    return {
        "retrieved_rubric_template": retrieved_doc.text,
        "active_rubric": adapted_rubric_text,
        "rubric_variant_a": retrieved_doc.text,
        "rubric_variant_b": adapted_rubric_text,
        "run_history": [HumanMessage(content=msg)],
    }
