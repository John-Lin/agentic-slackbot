FROM python:3.14-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

COPY . /app
WORKDIR /app

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --locked

CMD ["uv", "run", "bot"]
