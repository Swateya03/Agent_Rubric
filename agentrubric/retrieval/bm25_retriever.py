"""
Vectorless RAG using BM25 keyword matching to retrieve relevant rubrics.

BM25Okapi efficiently finds the best rubric template for a task
without requiring vector embeddings or semantic similarity models.
"""

import re
import string
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

PACKAGE_ROOT = Path(__file__).parent.parent
RUBRICS_DIR = PACKAGE_ROOT / "rubrics"


@dataclass
class RubricDocument:
    """A single rubric document with metadata and tokens for BM25 indexing."""

    name: str
    """Filename without extension e.g. "default"."""

    path: str
    """Full file path."""

    text: str
    """Full rubric text."""

    tokens: list[str]
    """Lowercased word tokens for BM25 indexing."""


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Args:
        text: Raw text to tokenize

    Returns:
        List of lowercase tokens (punctuation removed except hyphens, min 2 chars)
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation.replace('-', ''))}]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2]


class RubricRetriever:
    """BM25-based retrieval system for finding relevant rubric templates."""

    def __init__(self, rubrics_dir: str | Path = None):
        """Initialize the retriever by loading and indexing all rubrics.

        Args:
            rubrics_dir: Directory containing .txt rubric files. Defaults to RUBRICS_DIR.
        """
        self.documents: list[RubricDocument] = []
        if rubrics_dir is None:
            rubrics_dir = RUBRICS_DIR
        rubrics_path = Path(rubrics_dir)

        for rubric_file in sorted(rubrics_path.glob("*.txt")):
            name = rubric_file.stem
            text = rubric_file.read_text(encoding="utf-8")
            tokens = tokenize(text)

            doc = RubricDocument(
                name=name,
                path=str(rubric_file),
                text=text,
                tokens=tokens,
            )
            self.documents.append(doc)

        corpus = [doc.tokens for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)

        print(f"Initialized BM25Okapi with {len(self.documents)} rubrics:")
        for doc in self.documents:
            print(f"  - {doc.name} ({len(doc.tokens)} tokens)")

    def retrieve(self, query: str, top_k: int = 1) -> list[RubricDocument]:
        """Retrieve the top-k rubrics matching the query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of RubricDocument sorted by BM25 score (descending)
        """
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        scored_docs = list(zip(scores, self.documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        if all(s == 0.0 for s in scores):
            return [self.documents[0]]  # fallback: no query match, return first doc
        return [doc for _, doc in scored_docs[:top_k]]

    def retrieve_top(self, query: str) -> RubricDocument:
        """Retrieve the single best matching rubric.

        Args:
            query: Search query text

        Returns:
            The highest-scoring RubricDocument
        """
        results = self.retrieve(query, top_k=1)
        return results[0]
