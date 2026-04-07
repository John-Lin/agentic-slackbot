from __future__ import annotations

import json

import pytest

from bot.config import Configuration
from bot.config import env_flag


@pytest.fixture()
def config_env(monkeypatch):
    """Set required environment variables for Configuration."""
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")
    monkeypatch.setattr("bot.config.find_dotenv", lambda: "")


class TestConfiguration:
    def test_loads_slack_bot_token(self, config_env):
        cfg = Configuration()
        assert cfg.slack_bot_token == "xoxb-test"

    def test_loads_slack_app_token(self, config_env):
        cfg = Configuration()
        assert cfg.slack_app_token == "xapp-test"

    def test_loads_openai_key(self, config_env):
        cfg = Configuration()
        assert cfg.openai_api_key == "sk-test"

    def test_loads_http_proxy(self, config_env):
        cfg = Configuration()
        assert cfg.http_proxy == "http://proxy:8080"

    def test_missing_values_are_none(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.setattr("bot.config.find_dotenv", lambda: "")
        cfg = Configuration()
        assert cfg.slack_bot_token is None
        assert cfg.slack_app_token is None
        assert cfg.openai_api_key is None
        assert cfg.http_proxy is None


class TestLoadConfig:
    def test_loads_valid_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {"mcpServers": {"test": {"command": "echo", "args": []}}}
        config_file.write_text(json.dumps(config_data))
        result = Configuration.load_config(str(config_file))
        assert result == config_data

    def test_returns_default_on_missing_file(self):
        result = Configuration.load_config("/nonexistent/config.json")
        assert result == {"mcpServers": {}}

    def test_raises_on_invalid_json(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            Configuration.load_config(str(config_file))


class TestEnvFlag:
    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("MY_FLAG", raising=False)
        assert env_flag("MY_FLAG") is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_common_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("MY_FLAG", value)
        assert env_flag("MY_FLAG") is False

    @pytest.mark.parametrize("value", ["FALSE", "No", "Off", "  0  "])
    def test_case_insensitive_and_whitespace(self, monkeypatch, value):
        monkeypatch.setenv("MY_FLAG", value)
        assert env_flag("MY_FLAG") is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled", "anything"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("MY_FLAG", value)
        assert env_flag("MY_FLAG") is True
