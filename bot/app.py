import asyncio
import logging

from .agent import OpenAIAgent
from .config import Configuration
from .slack import SlackMCPBot


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    """Initialize and run the Slack bot."""
    config = Configuration()

    # server_config = config.load_config("servers_config.json")
    server_config = config.load_config("dummy_servers_config.json")

    # Initialize the OpenAI agents with mcp servers
    openai_agent = OpenAIAgent.from_dict("Slack Bot Agent", server_config["mcpServers"])

    # Initialize the OpenAI agents
    # openai_agent = OpenAIAgent("Slack Bot Agent")

    slack_bot = SlackMCPBot(
        config.slack_bot_token,
        config.slack_app_token,
        config.http_proxy,
        openai_agent,
    )

    try:
        await slack_bot.start()
        # Keep the main task alive until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await slack_bot.cleanup()
        await openai_agent.cleanup()


def run():
    asyncio.run(main())
