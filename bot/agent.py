from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from agents import Agent
from agents import Runner
from agents import ShellCommandRequest
from agents import ShellTool
from agents import ShellToolLocalEnvironment
from agents import ShellToolLocalSkill
from agents import TResponseInputItem
from agents.mcp import MCPServerStdio
from agents.mcp import MCPServerStreamableHttp
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from agents.tracing import set_tracing_disabled
from openai import AsyncOpenAI

from .config import env_flag

INSTRUCTIONS_FILE = Path("instructions.md")

MAX_TURNS = 10
MCP_SESSION_TIMEOUT_SECONDS = 30.0
SHELL_TIMEOUT = 30.0

set_tracing_disabled(True)


def _load_instructions() -> str:
    """Load agent instructions from ``instructions.md`` in the working directory.

    Fails fast with a clear error if the file is missing, so misconfiguration
    is caught immediately at startup.
    """
    try:
        return INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Instructions file not found: {INSTRUCTIONS_FILE.resolve()}. "
            "Create or mount instructions.md with the agent system prompt."
        ) from e


def _get_model() -> OpenAIResponsesModel | OpenAIChatCompletionsModel:
    """Create an OpenAI model from environment variables.

    Uses the standard OpenAI client, which works with both OpenAI and
    Azure OpenAI v1 API (via OPENAI_BASE_URL + OPENAI_API_KEY).

    OPENAI_API_TYPE controls which API the model uses:
      - "responses" (default): OpenAI Responses API — recommended by the SDK
      - "chat_completions": Chat Completions API
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.4")
    api_type = os.getenv("OPENAI_API_TYPE", "responses")
    client = AsyncOpenAI()

    if api_type == "chat_completions":
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    return OpenAIResponsesModel(model=model_name, openai_client=client)


def _parse_skill_description(content: str) -> str:
    """Return the description field from a SKILL.md YAML frontmatter, or ""."""
    if not content.startswith("---"):
        return ""
    end = content.find("\n---", 3)
    if end == -1:
        return ""
    for line in content[3:end].splitlines():
        if line.startswith("description:"):
            value = line[len("description:") :].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            return value
    return ""


def _load_shell_skills(skills_dir: Path) -> list[ShellToolLocalSkill]:
    """Discover local shell skills under ``skills_dir``.

    Each immediate subdirectory containing a ``SKILL.md`` file is mounted as a
    ``ShellToolLocalSkill``. The skill name is the directory name; the
    description is read from the ``SKILL.md`` YAML frontmatter.
    """
    if not skills_dir.is_dir():
        return []
    skills: list[ShellToolLocalSkill] = []
    for skill_dir in sorted(skills_dir.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_dir.is_dir() or not skill_md.is_file():
            continue
        try:
            content = skill_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logging.warning("Skipping unreadable skill file: %s", skill_md, exc_info=True)
            continue
        skills.append(
            ShellToolLocalSkill(
                name=skill_dir.name,
                description=_parse_skill_description(content),
                path=str(skill_dir),
            )
        )
    return skills


def _get_shell_environment() -> ShellToolLocalEnvironment | None:
    """Return the configured local shell environment, if enabled.

    Controlled by two independent environment variables:

    * ``SHELL_ENABLED`` — when truthy, the ``ShellTool`` is attached to the agent.
    * ``SHELL_SKILLS_DIR`` — optional path; when set, skills discovered under it
      are mounted alongside the shell. Ignored if ``SHELL_ENABLED`` is not set.
    """
    skills_dir_env = os.getenv("SHELL_SKILLS_DIR")
    if not env_flag("SHELL_ENABLED"):
        if skills_dir_env:
            logging.warning(
                "SHELL_SKILLS_DIR=%r is set but SHELL_ENABLED is not; ignoring skills dir.",
                skills_dir_env,
            )
        return None

    environment: ShellToolLocalEnvironment = {"type": "local"}
    if skills_dir_env:
        skills = _load_shell_skills(Path(skills_dir_env))
        if skills:
            environment["skills"] = skills
        else:
            logging.warning(
                "SHELL_SKILLS_DIR=%r yielded no skills; attaching bare local shell.",
                skills_dir_env,
            )
    return environment


async def _shell_executor(request: ShellCommandRequest) -> str:
    """Run each shell command from the request and return combined output.

    Honours action.timeout_ms when set, otherwise falls back to SHELL_TIMEOUT.
    stderr is merged into stdout for simplicity.
    """
    action = request.data.action
    timeout = (action.timeout_ms / 1000.0) if action.timeout_ms is not None else SHELL_TIMEOUT

    outputs: list[str] = []
    for command in action.commands:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as e:
            outputs.append(f"Failed to run command: {command}: {e}")
            break
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="replace")
            if proc.returncode:
                output += f"\n[exit code: {proc.returncode}]"
            outputs.append(output)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            outputs.append(f"Command timed out after {timeout}s: {command}")
            break
    return "\n".join(outputs)


class OpenAIAgent:
    """A wrapper for OpenAI Agent with MCP server and local shell skill support."""

    def __init__(
        self,
        name: str,
        instructions: str,
        mcp_servers: list | None = None,
        tools: list | None = None,
    ) -> None:
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=_get_model(),
            mcp_servers=(mcp_servers if mcp_servers is not None else []),
            tools=(tools if tools is not None else []),
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
            if "url" in mcp_srv:
                mcp_servers.append(
                    MCPServerStreamableHttp(
                        client_session_timeout_seconds=MCP_SESSION_TIMEOUT_SECONDS,
                        params={
                            "url": mcp_srv["url"],
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
        tools: list[Any] = []
        environment = _get_shell_environment()
        if environment is not None:
            tools.append(ShellTool(executor=_shell_executor, environment=environment))

        instructions = _load_instructions()
        return cls(name, instructions=instructions, mcp_servers=mcp_servers, tools=tools)

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
