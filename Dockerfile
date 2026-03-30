FROM python:3.14-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app
WORKDIR /app

# Provide empty MCP config as default; mount your own at runtime:
#   docker run -v /path/to/servers_config.json:/app/servers_config.json ...
COPY dummy_servers_config.json /app/servers_config.json

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --locked

VOLUME ["/app/servers_config.json"]

CMD ["uv", "run", "bot"]
