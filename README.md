# agentic-slackbot

A simple Slack bot that uses the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) to interact with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers.

See also: [agentic-telegram-bot](https://github.com/John-Lin/agentic-telegram-bot) — a similar demo bot for Telegram.

## Features

- Channel @mention and DM support
- Thread-aware conversations (follow-ups stay in the same thread)
- Connects to any MCP server via `servers_config.json`
- Local shell skills via `ShellTool` (opt-in via `SHELL_SKILLS_ENABLED`)
- Supports OpenAI and OpenAI-compatible endpoints (including Azure OpenAI v1 API)
- Per-conversation history with automatic truncation (last 10 turns)

## Install Dependencies

```bash
uv sync
```

## Slack App Setup

1. Create a new Slack app at [api.slack.com/apps](https://api.slack.com/apps).
2. Enable **Socket Mode** and generate an app-level token (`xapp-...`).
3. Under **OAuth & Permissions**, add the following bot token scopes:
   - `app_mentions:read`
   - `chat:write`
   - `im:history`
   - `users:read`
4. Under **Event Subscriptions**, subscribe to:
   - `app_mention`
   - `message.im`
5. Install the app to your workspace and copy the bot token (`xoxb-...`).

## Environment Variables

Create a `.envrc` or `.env` file in the root directory:

```
export SLACK_BOT_TOKEN=""
export SLACK_APP_TOKEN=""
export OPENAI_API_KEY=""
export OPENAI_MODEL="gpt-5.4"

# Shell skills (disabled by default)
# export SHELL_SKILLS_ENABLED=1
```

If you are using Azure OpenAI (v1 API) or another OpenAI-compatible endpoint:

```
export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://<resource-name>.openai.azure.com/openai/v1/"
export OPENAI_MODEL="gpt-5.4"
```

Optional HTTP proxy for outbound requests:

```
export HTTP_PROXY=""
```

## Agent Instructions

Create an `instructions.md` file in the project root with the agent system prompt:

```markdown
You are a helpful assistant in a Slack workspace.
When responding, you must strictly use Slack's `mrkdwn` formatting syntax only.
Keep responses concise and well-structured.
```

An example is provided in `instructions.md.example`. The bot will fail to start if this file is missing.

## MCP Server Configuration (Optional)

Create a `servers_config.json` file to add your MCP servers. If this file is not provided, the bot starts with no MCP servers configured.

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uvx",
      "args": ["my-mcp-server"]
    }
  }
}
```

For HTTP-based MCP servers (Streamable HTTP), use `url`:

```json
{
  "mcpServers": {
    "my-server": {
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

For local MCP servers, use `uv --directory`:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": ["--directory", "/path/to/my-server", "run", "my-entrypoint"]
    }
  }
}
```

## Running the Bot

```bash
uv run bot
```

## Shell Skills (Optional)

The bot can execute local shell commands via skills defined in a `skills/` directory. Each subdirectory containing a `SKILL.md` file is registered as a skill.

When using the Docker image, mount `skills/` at runtime (the image build excludes this directory by default):

```bash
-v /path/to/skills:/app/skills:ro
```

This feature is **disabled by default**. To enable it, set:

```
SHELL_SKILLS_ENABLED=1
```

Skills are auto-discovered at startup. The `SKILL.md` file should have YAML frontmatter with `name` and `description` fields:

```markdown
---
name: my-skill
description: A brief description of what this skill does
---

Detailed instructions for the agent...
```

## Docker

```bash
docker build -t agentic-slackbot .

docker run -d \
  --name slackbot \
  -e SLACK_BOT_TOKEN="" \
  -e SLACK_APP_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -e OPENAI_MODEL="gpt-5.4" \
  -e SHELL_SKILLS_ENABLED=1 \
  -v /path/to/instructions.md:/app/instructions.md \
  -v /path/to/skills:/app/skills:ro \
  agentic-slackbot
```

To use MCP servers, also mount your config:

```bash
docker run -d \
  --name slackbot \
  -e SLACK_BOT_TOKEN="" \
  -e SLACK_APP_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -e OPENAI_MODEL="gpt-5.4" \
  -e SHELL_SKILLS_ENABLED=1 \
  -v /path/to/instructions.md:/app/instructions.md \
  -v /path/to/skills:/app/skills:ro \
  -v /path/to/servers_config.json:/app/servers_config.json \
  agentic-slackbot
```

## Credit

This project is based on the [sooperset/mcp-client-slackbot](https://github.com/sooperset/mcp-client-slackbot) example.
