"""Tests for Slack bot event handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bot.slack import SlackMCPBot


@pytest.fixture
def bot():
    """Create a SlackMCPBot with mocked dependencies."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value="hello")
    with (
        patch("bot.slack.AsyncApp"),
        patch("bot.slack.AsyncSocketModeHandler"),
        patch("bot.slack.AsyncWebClient") as mock_web_client,
    ):
        web_client = MagicMock()
        web_client.users_info = AsyncMock(
            return_value={
                "user": {
                    "profile": {
                        "display_name": "",
                        "real_name": "Alice",
                    }
                }
            }
        )
        mock_web_client.return_value = web_client
        b = SlackMCPBot(
            slack_bot_token="xoxb-fake",
            slack_app_token="xapp-fake",
            proxy=None,
            openai_agent=agent,
        )
    b.bot_id = "U_BOT"
    return b


class TestHandleMention:
    @pytest.mark.anyio
    async def test_mention_calls_agent_with_channel_key(self, bot):
        event = {
            "channel": "C001",
            "user": "U123",
            "text": "<@U_BOT> what is the weather?",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_mention(event, say, ack)

        ack.assert_called_once()
        # Conversation key should be ts (no thread_ts means new conversation)
        bot.agent.run.assert_called_once_with("1234567890.123456", "[Alice] what is the weather?")

    @pytest.mark.anyio
    async def test_mention_in_thread_uses_thread_ts(self, bot):
        event = {
            "channel": "C001",
            "user": "U123",
            "text": "<@U_BOT> follow up",
            "thread_ts": "1111111111.111111",
            "ts": "2222222222.222222",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_mention(event, say, ack)

        bot.agent.run.assert_called_once_with("1111111111.111111", "[Alice] follow up")

    @pytest.mark.anyio
    async def test_mention_replies_in_thread(self, bot):
        event = {
            "channel": "C001",
            "user": "U123",
            "text": "<@U_BOT> hi",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_mention(event, say, ack)

        say.assert_called_once_with(text="hello", channel="C001", thread_ts="1234567890.123456")

    @pytest.mark.anyio
    async def test_mention_strips_bot_mention(self, bot):
        event = {
            "channel": "C001",
            "user": "U123",
            "text": "<@U_BOT> help me",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_mention(event, say, ack)

        bot.agent.run.assert_called_once_with("1234567890.123456", "[Alice] help me")


class TestHandleMessage:
    @pytest.mark.anyio
    async def test_dm_calls_agent(self, bot):
        message = {
            "channel": "D001",
            "channel_type": "im",
            "user": "U123",
            "text": "hello",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_message(message, say, ack)

        bot.agent.run.assert_called_once_with("1234567890.123456", "[Alice] hello")

    @pytest.mark.anyio
    async def test_non_dm_is_ignored(self, bot):
        message = {
            "channel": "C001",
            "channel_type": "channel",
            "user": "U123",
            "text": "hello",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_message(message, say, ack)

        bot.agent.run.assert_not_called()

    @pytest.mark.anyio
    async def test_subtype_messages_are_ignored(self, bot):
        message = {
            "channel": "D001",
            "channel_type": "im",
            "user": "U123",
            "text": "hello",
            "ts": "1234567890.123456",
            "subtype": "message_changed",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_message(message, say, ack)

        bot.agent.run.assert_not_called()

    @pytest.mark.anyio
    async def test_bot_own_messages_are_ignored(self, bot):
        message = {
            "channel": "D001",
            "channel_type": "im",
            "user": "U_BOT",
            "text": "hello",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_message(message, say, ack)

        bot.agent.run.assert_not_called()


class TestErrorHandling:
    @pytest.mark.anyio
    async def test_agent_error_sends_error_message(self, bot):
        bot.agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        event = {
            "channel": "C001",
            "user": "U123",
            "text": "<@U_BOT> hi",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        ack = AsyncMock()

        await bot.handle_mention(event, say, ack)

        say.assert_called_once()
        call_kwargs = say.call_args[1]
        assert "error" in call_kwargs["text"].lower()

    @pytest.mark.anyio
    async def test_say_failure_falls_back_to_plain_text(self, bot):
        """When say() fails with mrkdwn text, retry with raw agent output."""
        calls: list[dict] = []

        async def failing_say(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise Exception("invalid_blocks")

        bot.agent.run = AsyncMock(return_value="**bold heading**")
        event = {
            "channel": "C001",
            "user": "U123",
            "text": "<@U_BOT> hi",
            "ts": "1234567890.123456",
        }
        ack = AsyncMock()

        await bot.handle_mention(event, failing_say, ack)

        assert len(calls) == 2
        # First call: mrkdwn-converted text
        assert calls[0]["text"] == "*bold heading*"
        # Second call: raw plain text fallback
        assert calls[1]["text"] == "**bold heading**"
