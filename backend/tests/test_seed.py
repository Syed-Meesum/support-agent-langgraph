"""Seeder tests: the marker check, the row counts, and idempotence.

Runs against a temp SQLite file, so no Postgres and no network.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from app.db.seed import (
    TABLES,
    SeedReport,
    create_tables,
    is_seeded,
    open_seed_connection,
    seed,
    status,
)
from app.db.settings import CASE_DB_PATH_ENV, DATABASE_URL_ENV
from app.rag.retrieval import chunk_id, knowledge_chunks
from app.tools.fixtures import DEFAULT_FIXTURES_PATH


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv(CASE_DB_PATH_ENV, str(tmp_path / "seed.sqlite"))
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)


def fixtures() -> dict:
    return json.loads(DEFAULT_FIXTURES_PATH.read_text(encoding="utf-8"))


class TestSeeding:
    def test_a_fresh_database_is_empty(self):
        assert status() == {table: 0 for table in TABLES}

    def test_it_seeds_every_fixture_row(self):
        report = seed()
        raw = fixtures()
        assert report.skipped is False
        assert report.inserted["customers"] == len(raw["customers"])
        assert report.inserted["charges"] == len(raw["charges"])
        assert report.inserted["subscriptions"] == len(raw["subscriptions"])
        assert report.total == sum(report.inserted.values())

    def test_the_counts_land_in_the_database(self):
        expected = seed().inserted
        assert status() == expected

    def test_a_second_run_is_a_no_op(self):
        seed()
        again = seed()
        assert again.skipped is True
        assert again.inserted == {}
        assert "already seeded" in again.describe()

    def test_force_replaces_rather_than_duplicates(self):
        first = seed().inserted
        forced = seed(force=True)
        assert forced.skipped is False
        # same rows, not twice as many
        assert status() == first

    def test_the_marker_row_is_the_customer_count(self):
        with open_seed_connection() as db:
            create_tables(db)
            assert is_seeded(db) is False
        seed()
        with open_seed_connection() as db:
            assert is_seeded(db) is True

    def test_the_knowledge_chunks_match_the_retrieval_index(self):
        report = seed()
        chunks = knowledge_chunks()
        assert report.inserted["knowledge_chunks"] == len(chunks)
        assert len(chunks) > 0

    def test_knowledge_chunks_are_content_addressed_the_same_way_as_the_index(self):
        # The table and the vector index must agree on what a chunk is, or the
        # documented pgvector swap would silently duplicate or lose chunks.
        seed()
        rows = _chunk_ids()
        expected = {
            chunk_id(chunk.source, chunk.position, chunk.content) for chunk in knowledge_chunks()
        }
        assert rows == expected

    def test_knowledge_can_be_skipped(self):
        report = seed(include_knowledge=False)
        assert report.inserted["knowledge_chunks"] == 0
        assert status()["knowledge_chunks"] == 0
        assert status()["customers"] > 0

    def test_charges_keep_their_amounts_and_ids(self):
        seed()
        rows = _query("select id, amount, status from charges order by id")
        raw = {row["id"]: row for row in fixtures()["charges"]}
        assert len(rows) == len(raw)
        for row in rows:
            # The value round-trips exactly. The *text* form does not: SQLite's
            # NUMERIC affinity stores 15.00 as 15. Comparing as Decimal is the
            # assertion that matters - no float drift, no cents lost.
            assert Decimal(str(row["amount"])) == Decimal(raw[row["id"]]["amount"])
            assert row["status"] == raw[row["id"]]["status"]


class TestSeedReport:
    def test_describe_counts_the_rows(self):
        report = SeedReport(skipped=False, inserted={"customers": 10, "charges": 13})
        assert report.total == 23
        assert "customers=10" in report.describe()

    def test_a_skip_says_so(self):
        assert "nothing to do" in SeedReport(skipped=True, inserted={}).describe()


def test_create_tables_is_safe_to_repeat():
    with open_seed_connection() as db:
        create_tables(db)
        create_tables(db)
        assert db.count("customers") == 0


def _query(statement: str) -> list[dict]:
    with open_seed_connection() as db:
        return db.query(statement)


def _chunk_ids() -> set[str]:
    return {row["id"] for row in _query("select id from knowledge_chunks")}
