import asyncio
import logging
from typing import Any

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from .agent import OpenAIAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SlackMCPBot:
    """Manages the Slack bot integration with agents."""

    def __init__(
        self,
        slack_bot_token: str | None,
        slack_app_token: str | None,
        proxy: str | None,
        openai_agent: OpenAIAgent,
    ) -> None:
        self.app = AsyncApp(
            token=slack_bot_token,
            raise_error_for_unhandled_request=False,
        )
        # Create a socket mode handler with the app token
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, slack_app_token)

        self.client = AsyncWebClient(token=slack_bot_token, proxy=proxy)
        self.agent = openai_agent
        self.conversations: dict[
            str, dict[str, list[dict[str, str | Any | None]]]
        ] = {}  # Store conversation context per channel

        # Set up event handlers
        self.app.event("app_mention")(self.handle_mention)
        self.app.message()(self.handle_message)

    async def initialize_agent(self) -> None:
        """Initialize all MCP servers and discover tools."""
        try:
            await self.agent.connect()
            logging.info(f"Initialized agent {self.agent.name} with tools")
        except Exception as e:
            logging.error(f"Failed to initialize agent {self.agent.name}: {e}")

    async def initialize_bot_info(self) -> None:
        """Get the bot's ID and other info."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_id = None

    async def handle_mention(self, event, say):
        """Handle mentions of the bot in channels."""
        await self._process_message(event, say)

    async def handle_message(self, message, say):
        """Handle direct messages to the bot."""
        # Only process direct messages
        if message.get("channel_type") == "im" and not message.get("subtype"):
            await self._process_message(message, say)

    async def _process_message(self, event, say):
        """Process incoming messages and generate responses."""
        channel = event["channel"]
        user_id = event.get("user")

        # Skip messages from the bot itself
        if user_id == getattr(self, "bot_id", None):
            return

        # Get text and remove bot mention if present
        text = event.get("text", "")
        if hasattr(self, "bot_id") and self.bot_id:
            text = text.replace(f"<@{self.bot_id}>", "").strip()

        thread_ts = event.get("thread_ts", event.get("ts"))

        # Get or create conversation context
        if channel not in self.conversations:
            self.conversations[channel] = {"messages": []}

        try:
            messages = []

            # Add user message to history
            self.conversations[channel]["messages"].append({"role": "user", "content": text})

            # Add conversation history (last 5 messages)
            if "messages" in self.conversations[channel]:
                messages.extend(self.conversations[channel]["messages"][-5:])

            logging.debug(messages)
            # Get LLM response
            agent_resp = await self.agent.run(messages)

            # Add assistant response to conversation history
            self.conversations[channel]["messages"].append({"role": "assistant", "content": str(agent_resp)})

            # Send the response to the user
            await say(text=str(agent_resp), channel=channel, thread_ts=thread_ts)

        except Exception as e:
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            logging.error(f"Error processing message: {e}", exc_info=True)
            await say(text=error_message, channel=channel, thread_ts=thread_ts)

    async def start(self) -> None:
        """Start the Slack bot."""
        # await self.connect()
        await self.initialize_agent()
        await self.initialize_bot_info()
        # Start the socket mode handler
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
