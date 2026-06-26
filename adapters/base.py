"""
Base Adapter Interface
======================
Any RAG system plugs in by implementing this interface.
You tell rag-eval HOW to call your retriever, generator, reranker —
then rag-eval handles all the scoring automatically.

Example:
    class MyRAG(EvalAdapter):
        def retrieve(self, query):
            return my_vector_db.search(query, top_k=5)

        def generate(self, query, context):
            return my_llm.chat(query, context=context)

    evaluator = RagEvaluator(adapter=MyRAG())
    results = evaluator.evaluate(dataset)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """What your retriever returns."""
    chunks: list[str]                   # The retrieved text chunks
    scores: list[float] = field(default_factory=list)  # Similarity scores (optional)
    metadata: list[dict[str, Any]] = field(default_factory=list)  # Chunk metadata (optional)


@dataclass
class GenerationResult:
    """What your generator returns."""
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)  # Token usage, latency, etc.


class EvalAdapter(ABC):
    """
    Abstract base class for connecting rag-eval to ANY RAG system.

    Implement the methods that match your pipeline's components.
    Only `retrieve` and `generate` are required — everything else is optional.
    """

    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's question.

        Returns:
            RetrievalResult with chunks and optional scores/metadata.
        """
        ...

    @abstractmethod
    def generate(self, query: str, context: list[str]) -> GenerationResult:
        """
        Generate an answer given a query and retrieved context.

        Args:
            query: The user's question.
            context: Retrieved chunks from the retriever.

        Returns:
            GenerationResult with the answer string.
        """
        ...

    def rerank(self, query: str, chunks: list[str]) -> list[str]:
        """
        (Optional) Rerank chunks. Override if your pipeline has a reranker.

        Default: returns chunks as-is (no reranking).
        """
        return chunks

    def rewrite_query(self, query: str) -> str:
        """
        (Optional) Rewrite/expand a query. Override if your pipeline does query rewriting.

        Default: returns query as-is.
        """
        return query

    def get_tools(self) -> list[str]:
        """
        (Optional) Return the list of available tools for agent evaluation.

        Default: returns empty list (no agent evaluation).
        """
        return []

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        (Optional) Call a tool by name. Override for agent/MCP evaluation.

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(f"Tool calling not implemented for this adapter.")
