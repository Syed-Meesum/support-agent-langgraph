"""Shared test setup."""

from __future__ import annotations

import os

import pytest

from app.rag import retrieval


@pytest.fixture(scope="session", autouse=True)
def _isolated_chroma_index(tmp_path_factory):
    """Build the vector index in a temp dir, not in the repo.

    Session-scoped so the docs are embedded once, not once per test.
    """
    os.environ[retrieval.CHROMA_PATH_ENV] = str(tmp_path_factory.mktemp("chroma"))
    retrieval.reload_index()
    yield
    os.environ.pop(retrieval.CHROMA_PATH_ENV, None)
    retrieval.reload_index()
