"""Tests for root .env loading."""

from __future__ import annotations

import os

from app import config


class TestEnvLoading:
    def test_missing_env_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent.env")
        config.reload_env()
        assert config.load_env() is False

    def test_values_are_loaded_when_absent_from_the_process(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
        monkeypatch.setattr(config, "ENV_FILE", env_file)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config.reload_env()

        assert config.load_env() is True
        assert os.environ["ANTHROPIC_API_KEY"] == "from-file"

    def test_process_environment_wins_over_the_file(self, tmp_path, monkeypatch):
        # docker compose and an explicit export must not be clobbered by the file.
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
        monkeypatch.setattr(config, "ENV_FILE", env_file)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-process")
        config.reload_env()

        assert config.load_env() is True
        assert os.environ["ANTHROPIC_API_KEY"] == "from-process"

    def test_loading_is_idempotent(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
        monkeypatch.setattr(config, "ENV_FILE", env_file)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config.reload_env()

        assert config.load_env() is True
        os.environ["ANTHROPIC_API_KEY"] = "changed-after-load"
        assert config.load_env() is True
        # second call is a cache hit and does not re-read the file
        assert os.environ["ANTHROPIC_API_KEY"] == "changed-after-load"

    def test_default_env_file_is_the_repo_root(self):
        assert config.ENV_FILE.name == ".env"
        assert (config.REPO_ROOT / "README.md").is_file()
