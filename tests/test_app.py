from __future__ import annotations

from unittest.mock import patch

from bot.app import _configure_logging


class TestConfigureLogging:
    def test_enables_openai_agents_verbose_logging_when_env_var_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_AGENTS_VERBOSE_LOGGING", "1")

        with (
            patch("bot.app.logging.basicConfig") as mock_basic_config,
            patch("bot.app.enable_verbose_stdout_logging") as mock_enable_verbose_logging,
        ):
            _configure_logging()

        mock_basic_config.assert_called_once()
        mock_enable_verbose_logging.assert_called_once_with()

    def test_does_not_enable_openai_agents_verbose_logging_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_AGENTS_VERBOSE_LOGGING", raising=False)

        with (
            patch("bot.app.logging.basicConfig") as mock_basic_config,
            patch("bot.app.enable_verbose_stdout_logging") as mock_enable_verbose_logging,
        ):
            _configure_logging()

        mock_basic_config.assert_called_once()
        mock_enable_verbose_logging.assert_not_called()
