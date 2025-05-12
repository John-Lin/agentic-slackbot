# agentic-slackbot
A simple Slack bot that uses the OpenAI Agents SDK to interact with the Model Context Protocol (MCP) server.

## Install Dependencies

```bash
uv sync
```

## Environment Variables

Create a `.envrc` file in the root directory of the project and add the following environment variables:

```
export OPENAI_API_KEY=""
export SLACK_BOT_TOKEN=""
export SLACK_APP_TOKEN=""
export OPENAI_MODEL="gpt-4o"
export HTTP_PROXY=""
```

If you are using Azure OpenAI, you can set the following environment variables instead:
```
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT="https://<myopenai>.azure.com/"
OPENAI_MODEL="gpt-4o"
<<<<<<< HEAD
OPENAI_API_VERSION="2025-03-01-preview"
```

If you are using Langfuse

```
export LANGFUSE_PUBLIC_KEY="xxx"
export LANGFUSE_SECRET_KEY="xxx"
export LANGFUSE_HOST="xxx"
```

## Running the Bot

```bash
uv run bot
``````

Running the bot in docker

```bash
# Build the Docker image
docker build . -t agentic-slackbot

# Run the Docker container
docker run -e SLACK_BOT_TOKEN="" \
    -e SLACK_APP_TOKEN="" \
    -e HTTP_PROXY="" \
    -e OPENAI_PROXY_BASE_URL="" \
    -e OPENAI_PROXY_API_KEY="" \
    -e OPENAI_MODEL=gpt-4o \
    -e FIRECRAWL_API_URL="" slackbot
```

## Credit

This project is based on the [sooperset/mcp-client-slackbot](https://github.com/sooperset/mcp-client-slackbot) example.
