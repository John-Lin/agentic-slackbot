import json
import logging
import os
from typing import Any

from agent_core.env import env_flag
from dotenv import find_dotenv
from dotenv import load_dotenv

__all__ = ["Configuration", "env_flag"]

logger = logging.getLogger(__name__)


class Configuration:
    """Manages configuration and environment variables for the MCP Slackbot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.http_proxy = os.getenv("HTTP_PROXY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv(find_dotenv())

    _DEFAULT_CONFIG: dict[str, Any] = {"mcpServers": {}}

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Returns a default empty config when the file does not exist,
        allowing the application to start without a config file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            JSONDecodeError: If configuration file is invalid JSON.
        """
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file %s not found, using default empty config", file_path)
            return dict(Configuration._DEFAULT_CONFIG)
