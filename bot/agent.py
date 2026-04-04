from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from agents import Agent
from agents import Runner
from agents import TResponseInputItem
from agents.mcp import MCPServerStdio
from agents.mcp import MCPServerStreamableHttp
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI

DEFAULT_INSTRUCTIONS = (
    "You are agentic Slack bot, a helpful assistant.\n"
    "When responding, you must strictly use Slack's `mrkdwn` formatting syntax only.\n"
    "Do not generate headings (`#`), tables, or any other Markdown features not supported by Slack.\n"
    "Ensure that all output strictly complies with Slack's `mrkdwn` specifications.\n"
    "Your answer must be precise, of high-quality, and written by an expert using an unbiased and journalistic tone.\n"
    "Use the language specified by user in messages as the working language when explicitly provided.\n"
    "If you need to use Mandarin, you MUST use Traditional Chinese (台灣繁體中文)"
)

MAX_TURNS = 25
MCP_SESSION_TIMEOUT_SECONDS = 30.0


def _get_model() -> OpenAIChatCompletionsModel:
    """Create an OpenAI model from environment variables.

    Client selection priority:
    1. Azure OpenAI — when both AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set.
    2. OpenAI-compatible proxy — when OPENAI_PROXY_BASE_URL is set.
    3. Default OpenAI — falls back to standard AsyncOpenAI (reads OPENAI_API_KEY).
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.4")

    client: AsyncOpenAI
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("OPENAI_API_VERSION", "2025-04-01-preview"),
        )
    elif os.getenv("OPENAI_PROXY_BASE_URL"):
        client = AsyncOpenAI(
            base_url=os.environ["OPENAI_PROXY_BASE_URL"],
            api_key=os.getenv("OPENAI_PROXY_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        )
    else:
        client = AsyncOpenAI()

    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


class OpenAIAgent:
    """A wrapper for OpenAI Agent with MCP server support."""

    def __init__(
        self,
        name: str,
        mcp_servers: list | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ) -> None:
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=_get_model(),
            mcp_servers=(mcp_servers if mcp_servers is not None else []),
        )
        self.name = name
        self._conversations: dict[str, list[TResponseInputItem]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_messages(self, chat_id: str) -> list[TResponseInputItem]:
        return self._conversations.get(chat_id, [])

    def set_messages(self, chat_id: str, messages: list[TResponseInputItem]) -> None:
        self._conversations[chat_id] = messages

    def append_user_message(self, chat_id: str, message: str) -> None:
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []
        self._conversations[chat_id].append({"role": "user", "content": message})

    def truncate_history(self, chat_id: str) -> None:
        """Keep only the last MAX_TURNS turns of conversation history.

        A turn starts at each user message. All messages between two user
        messages (assistant replies, tool calls, tool results) belong to
        the preceding turn.
        """
        msgs = self.get_messages(chat_id)
        user_indices = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        if len(user_indices) <= MAX_TURNS:
            return
        cut = user_indices[-MAX_TURNS]
        self._conversations[chat_id] = msgs[cut:]

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> OpenAIAgent:
        mcp_servers: list[MCPServerStreamableHttp | MCPServerStdio] = []
        for mcp_srv in config.get("mcpServers", {}).values():
            if "httpUrl" in mcp_srv:
                mcp_servers.append(
                    MCPServerStreamableHttp(
                        client_session_timeout_seconds=MCP_SESSION_TIMEOUT_SECONDS,
                        params={
                            "url": mcp_srv["httpUrl"],
                            "headers": mcp_srv.get("headers", {}),
                        },
                    )
                )
            else:
                mcp_servers.append(
                    MCPServerStdio(
                        client_session_timeout_seconds=MCP_SESSION_TIMEOUT_SECONDS,
                        params={
                            "command": mcp_srv["command"],
                            "args": mcp_srv.get("args", []),
                            "env": mcp_srv.get("env"),
                        },
                    )
                )
        instructions = config.get("instructions", DEFAULT_INSTRUCTIONS)
        return cls(name, mcp_servers, instructions=instructions)

    async def connect(self) -> None:
        for mcp_server in self.agent.mcp_servers:
            try:
                await mcp_server.connect()
                logging.info(f"Server {mcp_server.name} connected")
            except Exception:
                logging.warning(
                    f"MCP server {mcp_server.name} failed to connect — bot will run without its tools",
                    exc_info=True,
                )

    async def run(self, chat_id: str, message: str) -> str:
        """Run a workflow starting at the given agent.

        A per-chat async lock ensures that when two messages arrive for the
        same conversation in quick succession, they are processed sequentially.
        Without the lock, the second ``await Runner.run`` could start before
        the first one finishes, and whichever completes last would overwrite
        the conversation history — silently dropping the other message and
        its reply.  Different conversations are unaffected and still run in
        parallel.
        """
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            history = self.get_messages(chat_id) + [{"role": "user", "content": message}]
            result = await Runner.run(self.agent, input=history)
            self.set_messages(chat_id, result.to_input_list())
            self.truncate_history(chat_id)
            return str(result.final_output)

    async def cleanup(self) -> None:
        """Clean up resources."""
        for mcp_server in self.agent.mcp_servers:
            try:
                await mcp_server.cleanup()
                logging.info(f"Server {mcp_server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {mcp_server.name}: {e}")
