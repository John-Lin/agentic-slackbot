import asyncio
import logging

from .agent import OpenAIAgent
from .config import Configuration
from .slack import SlackMCPBot


async def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    config = Configuration()
    server_config = config.load_config("servers_config.json")

    openai_agent = OpenAIAgent.from_dict("Slack Bot Agent", server_config)

    slack_bot = SlackMCPBot(
        config.slack_bot_token,
        config.slack_app_token,
        config.http_proxy,
        openai_agent,
    )

    try:
        await slack_bot.start()
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await slack_bot.cleanup()
        await openai_agent.cleanup()


def run():
    asyncio.run(main())
