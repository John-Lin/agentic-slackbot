from __future__ import annotations

import logging
from typing import Any

from agentize.model import get_openai_model
from agentize.model import get_openai_model_settings
from agentize.prompts.summary import INSTRUCTIONS as SUMMARIZE_PROMPT
from agentize.prompts.summary import Summary
from agentize.tools.duckduckgo import duckduckgo_search
from agentize.tools.firecrawl import map_tool
from agentize.tools.markitdown import markitdown_scrape_tool
from agents import Agent
from agents import Runner
from agents import TResponseInputItem
from agents.mcp import MCPServerStdio

INSTRUCTIONS = """
You are agentic Slack bot, a helpful assistant.
When responding, you must strictly use Slack’s `mrkdwn` formatting syntax only.
Do not generate headings (`#`), tables, or any other Markdown features not supported by Slack.
Ensure that all output strictly complies with Slack’s `mrkdwn` specifications.
Your answer must be precise, of high-quality, and written by an expert using an unbiased and journalistic tone.
Use the language specified by user in messages as the working language when explicitly provided.
If you need to use Mandarin, you MUST use Traditional Chinese (台灣繁體中文)
You MUST handoff to the summary agent when you need to summarize.
"""  # noqa


class OpenAIAgent:
    """A wrapper for OpenAI Agent"""

    def __init__(self, name: str, mcp_servers: list | None = None) -> None:
        self.language_preference = "Traditional Chinese (台灣繁體中文)"
        self.summary_agent = Agent(
            name="summary_agent",
            model=get_openai_model(model="o3-mini", api_type="chat_completions"),
            instructions=SUMMARIZE_PROMPT.format(lang=self.language_preference, length=1_000),
            output_type=Summary,
        )
        self.main_agent = Agent(
            name=name,
            instructions=INSTRUCTIONS.format(lang=self.language_preference),
            model=get_openai_model(model="gpt-4.1", api_type="chat_completions"),
            model_settings=get_openai_model_settings(),
            tools=[markitdown_scrape_tool, map_tool, duckduckgo_search],
            handoffs=[self.summary_agent],
            mcp_servers=(mcp_servers if mcp_servers is not None else []),
        )
        self.name = name
        self.messages: list[TResponseInputItem] = []

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> OpenAIAgent:
        mcp_servers = [
            MCPServerStdio(
                client_session_timeout_seconds=60.0,
                params={
                    "command": mcp_srv["command"],
                    "args": mcp_srv["args"],
                    "env": mcp_srv.get("env", {}),
                },
            )
            for mcp_srv in config.values()
        ]
        return cls(name, mcp_servers)

    async def connect(self) -> None:
        for mcp_server in self.main_agent.mcp_servers:
            try:
                await mcp_server.connect()
                logging.info(f"Server {mcp_server.name} connecting")
            except Exception as e:
                logging.error(f"Error during connecting of server {mcp_server.name}: {e}")

    async def run(self, message: str) -> str:
        """Run a workflow starting at the given agent."""
        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )
        result = await Runner.run(self.main_agent, input=self.messages)
        self.messages = result.to_input_list()
        # Add conversation history (last 5 messages)
        self.messages = self.messages[-5:]
        return result.final_output

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up servers
        for mcp_server in self.main_agent.mcp_servers:
            try:
                await mcp_server.cleanup()
                logging.info(f"Server {mcp_server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {mcp_server.name}: {e}")
