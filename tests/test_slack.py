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

    @pytest.mark.anyio
    async def test_thread_followup_without_mention_is_ignored(self, bot):
        """A plain (non-@mention) message in a thread is ignored, even if the
        same user previously @mentioned the bot in that thread. Every turn
        requires an explicit mention.
        """
        # First turn: user @mentions in thread.
        mention_event = {
            "channel": "C001",
            "user": "U123",
            "text": "<@U_BOT> first question",
            "thread_ts": "1111111111.111111",
            "ts": "1111111111.111111",
        }
        await bot.handle_mention(mention_event, AsyncMock(), AsyncMock())
        assert bot.agent.run.call_count == 1

        # Second turn: same user, same thread, no @mention — should be ignored.
        followup = {
            "channel": "C001",
            "channel_type": "channel",
            "user": "U123",
            "text": "follow up without mention",
            "thread_ts": "1111111111.111111",
            "ts": "2222222222.222222",
        }
        await bot.handle_message(followup, AsyncMock(), AsyncMock())

        # Still only the original mention call.
        assert bot.agent.run.call_count == 1


class TestMultiUserThread:
    @pytest.mark.anyio
    async def test_second_user_mention_in_same_thread_shares_history(self, bot):
        """When a second user joins a thread by @mentioning the bot, they engage
        the same conversation key (thread_ts) so the bot has shared history.
        """
        thread_ts = "1111111111.111111"

        alice_mention = {
            "channel": "C001",
            "user": "U_ALICE",
            "text": "<@U_BOT> what is the capital of France?",
            "thread_ts": thread_ts,
            "ts": thread_ts,
        }
        bob_mention = {
            "channel": "C001",
            "user": "U_BOB",
            "text": "<@U_BOT> and Germany?",
            "thread_ts": thread_ts,
            "ts": "2222222222.222222",
        }

        await bot.handle_mention(alice_mention, AsyncMock(), AsyncMock())
        await bot.handle_mention(bob_mention, AsyncMock(), AsyncMock())

        # Both calls used the same thread_ts as the conversation key.
        assert bot.agent.run.call_count == 2
        assert bot.agent.run.call_args_list[0].args[0] == thread_ts
        assert bot.agent.run.call_args_list[1].args[0] == thread_ts

    @pytest.mark.anyio
    async def test_second_user_plain_message_in_thread_is_ignored(self, bot):
        """A second user typing a plain (non-@mention) message in an active
        thread is ignored — they must @mention to participate.
        """
        thread_ts = "1111111111.111111"

        # Alice opens the thread with a mention.
        alice_mention = {
            "channel": "C001",
            "user": "U_ALICE",
            "text": "<@U_BOT> hi",
            "thread_ts": thread_ts,
            "ts": thread_ts,
        }
        await bot.handle_mention(alice_mention, AsyncMock(), AsyncMock())
        assert bot.agent.run.call_count == 1

        # Bob types in the thread without mentioning the bot.
        bob_plain = {
            "channel": "C001",
            "channel_type": "channel",
            "user": "U_BOB",
            "text": "hey what's going on",
            "thread_ts": thread_ts,
            "ts": "2222222222.222222",
        }
        await bot.handle_message(bob_plain, AsyncMock(), AsyncMock())

        # Bob's plain message did not trigger the bot.
        assert bot.agent.run.call_count == 1


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
