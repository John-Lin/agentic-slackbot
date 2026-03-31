# agentic-slackbot

A simple Slack bot that uses the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) to interact with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers.

See also: [agentic-telegram-bot](https://github.com/John-Lin/agentic-telegram-bot) — a similar demo bot for Telegram.

## Features

- Channel @mention and DM support
- Thread-aware conversations (follow-ups stay in the same thread)
- Connects to any MCP server via `servers_config.json`
- Supports OpenAI, Azure OpenAI, and OpenAI-compatible proxy endpoints
- Per-conversation history with automatic truncation

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
export OPENAI_MODEL="gpt-4.1"
```

If you are using Azure OpenAI, set these instead:

```
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_ENDPOINT="https://<myopenai>.azure.com/"
export OPENAI_MODEL="gpt-4.1"
export OPENAI_API_VERSION="2025-03-01-preview"
```

If you are using an OpenAI-compatible proxy:

```
export OPENAI_PROXY_BASE_URL="https://my-proxy.example.com/v1"
export OPENAI_PROXY_API_KEY=""
```

Optional HTTP proxy for outbound requests:

```
export HTTP_PROXY=""
```

## MCP Server Configuration

Edit `servers_config.json` to add your MCP servers:

```json
{
  "instructions": "Your custom system prompt here.",
  "mcpServers": {
    "my-server": {
      "command": "uvx",
      "args": ["my-mcp-server"]
    }
  }
}
```

For local MCP servers, use `uv --directory`:

```json
{
  "instructions": "Your custom system prompt here.",
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

## Docker

```bash
docker build -t agentic-slackbot .

docker run -d \
  --name slackbot \
  -e SLACK_BOT_TOKEN="" \
  -e SLACK_APP_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -e OPENAI_MODEL="gpt-4.1" \
  -v /path/to/servers_config.json:/app/servers_config.json \
  agentic-slackbot
```

## Credit

This project is based on the [sooperset/mcp-client-slackbot](https://github.com/sooperset/mcp-client-slackbot) example.
