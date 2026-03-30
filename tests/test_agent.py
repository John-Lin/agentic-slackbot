from __future__ import annotations

from unittest.mock import create_autospec
from unittest.mock import patch

import pytest
from agents.models.interface import Model

from bot.agent import DEFAULT_INSTRUCTIONS
from bot.agent import MAX_TURNS
from bot.agent import OpenAIAgent


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch):
    """Prevent tests from constructing a real OpenAI client."""
    monkeypatch.setattr("bot.agent._get_model", lambda: create_autospec(Model))


class TestPerChannelConversations:
    def test_separate_channels_have_independent_history(self):
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id="C001", message="hello from C001")
        agent.append_user_message(chat_id="C002", message="hello from C002")

        msgs_c001 = agent.get_messages(chat_id="C001")
        msgs_c002 = agent.get_messages(chat_id="C002")

        assert len(msgs_c001) == 1
        assert len(msgs_c002) == 1
        assert msgs_c001[0]["content"] == "hello from C001"
        assert msgs_c002[0]["content"] == "hello from C002"

    def test_same_channel_accumulates_messages(self):
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id="C001", message="first")
        agent.append_user_message(chat_id="C001", message="second")

        msgs = agent.get_messages(chat_id="C001")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"

    def test_unknown_channel_returns_empty(self):
        agent = OpenAIAgent(name="test")
        assert agent.get_messages(chat_id="C999") == []

    def test_set_messages_replaces_history(self):
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id="C001", message="old")
        new_msgs = [{"role": "user", "content": "replaced"}]
        agent.set_messages(chat_id="C001", messages=new_msgs)
        assert agent.get_messages(chat_id="C001") == new_msgs

    def test_set_messages_does_not_affect_other_channels(self):
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id="C001", message="channel 1")
        agent.append_user_message(chat_id="C002", message="channel 2")
        agent.set_messages(chat_id="C001", messages=[])
        assert agent.get_messages(chat_id="C001") == []
        assert len(agent.get_messages(chat_id="C002")) == 1


class TestInstructions:
    def test_default_instructions_when_none_provided(self):
        agent = OpenAIAgent(name="test")
        assert agent.agent.instructions == DEFAULT_INSTRUCTIONS

    def test_custom_instructions(self):
        agent = OpenAIAgent(name="test", instructions="Be a Slack helper.")
        assert agent.agent.instructions == "Be a Slack helper."

    def test_from_dict_reads_instructions(self):
        config = {
            "instructions": "Custom prompt here.",
            "mcpServers": {},
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert agent.agent.instructions == "Custom prompt here."

    def test_from_dict_uses_default_without_instructions(self):
        config = {
            "mcpServers": {},
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert agent.agent.instructions == DEFAULT_INSTRUCTIONS


class TestHistoryTruncation:
    def test_default_max_turns(self):
        assert MAX_TURNS == 25

    def test_truncate_keeps_recent_turns(self):
        agent = OpenAIAgent(name="test")
        for i in range(30):
            agent.set_messages(
                chat_id="C001",
                messages=agent.get_messages(chat_id="C001")
                + [
                    {"role": "user", "content": f"user-{i}"},
                    {"role": "assistant", "content": f"assistant-{i}"},
                ],
            )

        agent.truncate_history(chat_id="C001")
        msgs = agent.get_messages(chat_id="C001")

        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == MAX_TURNS
        assert user_msgs[0]["content"] == "user-5"
        assert user_msgs[-1]["content"] == "user-29"

    def test_truncate_preserves_tool_messages_within_turn(self):
        agent = OpenAIAgent(name="test")
        history = []
        for i in range(MAX_TURNS + 2):
            history.append({"role": "user", "content": f"user-{i}"})
            if i == MAX_TURNS + 1:
                history.append({"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]})
                history.append({"role": "tool", "content": "tool-result", "tool_call_id": "tc1"})
            history.append({"role": "assistant", "content": f"assistant-{i}"})

        agent.set_messages(chat_id="C001", messages=history)
        agent.truncate_history(chat_id="C001")
        msgs = agent.get_messages(chat_id="C001")

        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == MAX_TURNS
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    def test_no_truncation_when_under_limit(self):
        agent = OpenAIAgent(name="test")
        for i in range(3):
            agent.set_messages(
                chat_id="C001",
                messages=agent.get_messages(chat_id="C001")
                + [
                    {"role": "user", "content": f"user-{i}"},
                    {"role": "assistant", "content": f"assistant-{i}"},
                ],
            )

        agent.truncate_history(chat_id="C001")
        msgs = agent.get_messages(chat_id="C001")
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 3


class TestFromDict:
    def test_creates_mcp_servers_from_config(self):
        config = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["hello"],
                    "env": {"KEY": "val"},
                }
            }
        }
        with patch("bot.agent.MCPServerStdio") as mock_mcp:
            OpenAIAgent.from_dict("test", config)
            mock_mcp.assert_called_once()
            call_kwargs = mock_mcp.call_args[1]
            assert call_kwargs["params"]["command"] == "echo"
            assert call_kwargs["params"]["args"] == ["hello"]
            assert call_kwargs["params"]["env"] == {"KEY": "val"}

    def test_empty_mcp_servers(self):
        config = {"mcpServers": {}}
        agent = OpenAIAgent.from_dict("test", config)
        assert agent.agent.mcp_servers == []
