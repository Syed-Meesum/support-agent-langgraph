"""Tests for the vector retrieval layer behind search_knowledge_base.

These are the tests that would fail if someone swapped real embeddings back out
for keyword matching, which is the point of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag import _chunking, retrieval
from app.tools import knowledge_tools


def _sources(hits) -> set[str]:
    return {hit.source for hit in hits}


class TestSemanticRetrieval:
    def test_paraphrase_with_little_keyword_overlap_still_matches(self):
        # No rare content words in common with the doc: "money back", "twice",
        # "this month". A lexical scorer does not get this one.
        hits = knowledge_tools.search_knowledge_base(
            "I want my money back, I was charged twice this month"
        )
        assert "duplicate-charge-policy.md" in _sources(hits)

    def test_synonym_query_matches_the_threshold_policy(self):
        hits = knowledge_tools.search_knowledge_base(
            "what is the biggest amount you can pay out without a person checking"
        )
        assert "refund-policy.md" in _sources(hits)

    def test_unrelated_query_falls_below_the_floor(self):
        query = "the weather in lisbon next tuesday"
        hits = knowledge_tools.search_knowledge_base(query)
        assert hits == []

    def test_threshold_is_configurable(self, monkeypatch):
        query = "the weather in lisbon next tuesday"
        monkeypatch.setenv(retrieval.MIN_SIMILARITY_ENV, "0.0")
        assert knowledge_tools.search_knowledge_base(query) != []

    def test_scores_are_bounded_similarities(self):
        hits = knowledge_tools.search_knowledge_base("refund policy", top_k=5)
        assert hits
        assert all(0.0 <= hit.score <= 1.0 for hit in hits)

    def test_the_same_query_is_deterministic(self):
        first = knowledge_tools.search_knowledge_base("duplicate charge", top_k=3)
        second = knowledge_tools.search_knowledge_base("duplicate charge", top_k=3)
        assert first == second


class TestIndexSync:
    def test_every_chunk_of_every_doc_is_indexed(self):
        expected = sum(
            len(_chunking.split_chunks(path))
            for path in sorted(retrieval.KNOWLEDGE_DIR.glob("*.md"))
        )
        assert expected > 0
        assert retrieval.indexed_chunk_count() == expected

    def test_resyncing_does_not_duplicate_chunks(self):
        before = retrieval.indexed_chunk_count()
        retrieval.reload_index()
        assert retrieval.indexed_chunk_count() == before

    def test_index_persists_across_a_client_rebuild(self):
        before = knowledge_tools.search_knowledge_base("cancellation", top_k=3)
        retrieval.reload_index()
        after = knowledge_tools.search_knowledge_base("cancellation", top_k=3)
        assert [(h.source, h.score) for h in before] == [(h.source, h.score) for h in after]


class TestChunking:
    def test_splits_on_second_level_headings(self, tmp_path: Path):
        doc = tmp_path / "sample.md"
        doc.write_text(
            "# Title\n\nintro\n\n## Alpha\n\nalpha body\n\n## Beta\n\nbeta body\n",
            encoding="utf-8",
        )
        chunks = _chunking.split_chunks(doc)
        assert [c.position for c in chunks] == [0, 1, 2]
        assert all(c.source == "sample.md" for c in chunks)
        assert "alpha body" in chunks[1].content
        assert "beta body" in chunks[2].content

    def test_empty_document_yields_no_chunks(self, tmp_path: Path):
        doc = tmp_path / "empty.md"
        doc.write_text("   \n\n", encoding="utf-8")
        assert _chunking.split_chunks(doc) == []

    def test_real_knowledge_docs_all_produce_chunks(self):
        docs = sorted(retrieval.KNOWLEDGE_DIR.glob("*.md"))
        assert len(docs) >= 5
        for path in docs:
            assert _chunking.split_chunks(path), f"{path.name} produced no chunks"
