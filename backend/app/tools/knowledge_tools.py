"""Mocked RAG tool: search_knowledge_base."""

from __future__ import annotations

from app.rag import retrieval
from app.tools.models import KnowledgeHit


def search_knowledge_base(query: str, top_k: int = 3) -> list[KnowledgeHit]:
    """Search the policy knowledge base and return the top matches."""
    return retrieval.search(query, top_k=top_k)
