import asyncio
import logging

from agent_core import OpenAIAgent
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from .formatting import markdown_to_slack_mrkdwn


class SlackMCPBot:
    """Manages the Slack bot integration with agents."""

    def __init__(
        self,
        slack_bot_token: str | None,
        slack_app_token: str | None,
        proxy: str | None,
        openai_agent: OpenAIAgent,
    ) -> None:
        if slack_bot_token is None:
            raise ValueError("SLACK_BOT_TOKEN is not set")
        if slack_app_token is None:
            raise ValueError("SLACK_APP_TOKEN is not set")

        self.app = AsyncApp(
            token=slack_bot_token,
            raise_error_for_unhandled_request=False,
        )
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, slack_app_token)

        self.client = AsyncWebClient(token=slack_bot_token, proxy=proxy)
        self.agent = openai_agent
        self._user_name_cache: dict[str, str] = {}

        # Set up event handlers
        self.app.event("app_mention")(self.handle_mention)
        self.app.event("message")(self.handle_message)

    async def initialize_agent(self) -> None:
        """Initialize all MCP servers and discover tools."""
        await self.agent.connect()
        logging.info(f"Initialized agent {self.agent.name}")

    async def initialize_bot_info(self) -> None:
        """Get the bot's ID and other info."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_id = None

    async def handle_mention(self, event, say, ack):
        """Handle mentions of the bot in channels and threads."""
        await ack()
        await self._process_message(event, say)

    async def handle_message(self, message, say, ack):
        """Handle direct messages. Channel and thread messages must @mention
        the bot and are handled by ``handle_mention`` via the ``app_mention``
        event — every turn in a thread requires an explicit mention, regardless
        of who started the thread.
        """
        await ack()
        if message.get("subtype"):
            return

        if message.get("channel_type") == "im":
            await self._process_message(message, say)

    async def _get_display_name(self, user_id: str) -> str:
        if user_id in self._user_name_cache:
            return self._user_name_cache[user_id]
        try:
            resp = await self.client.users_info(user=user_id)
            profile = resp["user"]["profile"]
            name = profile.get("display_name") or profile.get("real_name") or user_id
        except Exception:
            name = user_id
        self._user_name_cache[user_id] = name
        return name

    async def _process_message(self, event, say):
        """Process incoming messages and generate responses."""
        channel = event["channel"]
        user_id = event.get("user")

        # Skip messages from the bot itself
        if user_id == getattr(self, "bot_id", None):
            return

        # Get text and remove bot mention if present
        user_text = event.get("text", "")
        if hasattr(self, "bot_id") and self.bot_id:
            user_text = user_text.replace(f"<@{self.bot_id}>", "").strip()

        if user_id:
            display_name = await self._get_display_name(user_id)
            user_text = f"[{display_name}] {user_text}"

        thread_ts = event.get("thread_ts", event.get("ts"))

        try:
            asst_text = await self.agent.run(thread_ts, user_text)
            mrkdwn_text = markdown_to_slack_mrkdwn(str(asst_text))
            try:
                await say(text=mrkdwn_text, channel=channel, thread_ts=thread_ts)
            except Exception:
                logging.warning("Failed to send mrkdwn message, falling back to plain text")
                await say(text=str(asst_text), channel=channel, thread_ts=thread_ts)
        except Exception as e:
            logging.error(f"Error processing message: {e}", exc_info=True)
            await say(
                text="I'm sorry, I encountered an error processing your request.", channel=channel, thread_ts=thread_ts
            )

    async def start(self) -> None:
        """Start the Slack bot."""
        await self.initialize_agent()
        await self.initialize_bot_info()
        logging.info("Starting Slack bot...")
        asyncio.create_task(self.socket_mode_handler.start_async())
        logging.info("Slack bot started and waiting for messages")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "socket_mode_handler"):
                await self.socket_mode_handler.close_async()
            logging.info("Slack socket mode handler closed")
        except Exception as e:
            logging.error(f"Error closing socket mode handler: {e}")
