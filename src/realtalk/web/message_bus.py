"""Message bus for decoupled WebSocket message handling.

This module implements a publish-subscribe pattern for WebSocket messages,
enabling clean separation between message routing and business logic.
"""
import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

from ..logging_config import setup_logger
from ..messages import (
    AnyMessage,
    BaseMessage,
    ClientMessage,
    MessageType,
    deserialize_message,
)

logger = setup_logger("realtalk.message_bus")

T = TypeVar("T", bound=BaseMessage)

# Type alias for message handlers
MessageHandler = Callable[[T], Awaitable[None]]
"""Type alias for async message handler functions."""


class MessageBus:
    """Message bus for routing WebSocket messages to handlers.

    The message bus decouples message reception from message processing,
    allowing handlers to be registered dynamically and tested in isolation.

    Example:
        bus = MessageBus()

        @bus.subscribe(MessageType.TEXT)
        async def handle_text(msg: TextInput) -> None:
            print(f"Received: {msg.text}")

        await bus.publish(TextInput(text="Hello"))
    """

    def __init__(self):
        self._handlers: Dict[MessageType, List[MessageHandler]] = defaultdict(list)
        self._middleware: List[Callable[[BaseMessage, Callable], Awaitable[None]]] = []
        self._metrics: Dict[str, int] = defaultdict(int)

    def subscribe(
        self,
        message_type: MessageType,
        handler: MessageHandler,
    ) -> Callable[[], None]:
        """Subscribe a handler to a specific message type.

        Args:
            message_type: The message type to subscribe to
            handler: Async handler function that receives the message

        Returns:
            Unsubscribe function that removes the handler when called
        """
        self._handlers[message_type].append(handler)
        logger.debug(f"Subscribed handler to {message_type.value}")

        def unsubscribe() -> None:
            if handler in self._handlers[message_type]:
                self._handlers[message_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {message_type.value}")

        return unsubscribe

    def subscribe_all(
        self,
        handler: Callable[[AnyMessage], Awaitable[None]],
    ) -> Callable[[], None]:
        """Subscribe a handler to all message types.

        Useful for logging, metrics, or catch-all handlers.

        Args:
            handler: Async handler function that receives any message

        Returns:
            Unsubscribe function that removes the handler when called
        """
        unsubscribers = []
        for msg_type in MessageType:
            unsubscribers.append(self.subscribe(msg_type, handler))

        def unsubscribe_all() -> None:
            for unsub in unsubscribers:
                unsub()

        return unsubscribe_all

    def add_middleware(
        self,
        middleware: Callable[[BaseMessage, Callable], Awaitable[None]],
    ) -> None:
        """Add middleware to the message processing pipeline.

        Middleware can inspect, modify, or drop messages before they reach handlers.

        Args:
            middleware: Async middleware function with signature:
                async def middleware(message: BaseMessage, next_handler: Callable) -> None

        Example:
            async def logging_middleware(msg: BaseMessage, next_handler: Callable) -> None:
                logger.info(f"Processing {msg.type}")
                await next_handler(msg)

            bus.add_middleware(logging_middleware)
        """
        self._middleware.append(middleware)

    async def publish(self, message: BaseMessage) -> None:
        """Publish a message to all subscribed handlers.

        Args:
            message: The message to publish
        """
        msg_type = MessageType(message.get_type_value())
        self._metrics[f"published.{msg_type.value}"] += 1

        handlers = self._handlers.get(msg_type, [])
        if not handlers:
            logger.warning(f"No handlers registered for message type: {msg_type.value}")
            return

        # Build middleware chain
        async def execute_handlers(msg: BaseMessage) -> None:
            for handler in handlers:
                try:
                    # Cast to Any because handler expects specific type
                    await handler(msg)  # type: ignore
                except Exception as e:
                    logger.exception(f"Handler error for {msg_type.value}: {e}")
                    # Continue with other handlers even if one fails

        # Execute through middleware chain
        await self._execute_middleware(message, execute_handlers)

    async def _execute_middleware(
        self,
        message: BaseMessage,
        final_handler: Callable[[BaseMessage], Awaitable[None]],
    ) -> None:
        """Execute middleware chain ending with the final handler."""

        async def build_chain(index: int) -> Callable[[], Awaitable[None]]:
            if index == len(self._middleware):
                return lambda: final_handler(message)

            middleware = self._middleware[index]
            next_handler = await build_chain(index + 1)
            return lambda: middleware(message, lambda: next_handler())

        handler = await build_chain(0)
        await handler()

    async def handle_raw_message(self, data: Dict[str, Any]) -> Optional[BaseMessage]:
        """Deserialize and publish a raw message dictionary.

        This is a convenience method for handling WebSocket messages.

        Args:
            data: Raw message dictionary (typically from JSON)

        Returns:
            The deserialized message, or None if deserialization failed
        """
        try:
            message = deserialize_message(data)
            await self.publish(message)
            return message
        except ValueError as e:
            logger.error(f"Message deserialization failed: {e}")
            self._metrics["errors.deserialization"] += 1
            return None
        except Exception as e:
            logger.exception(f"Unexpected error handling message: {e}")
            self._metrics["errors.unexpected"] += 1
            return None

    def get_metrics(self) -> Dict[str, int]:
        """Get message processing metrics.

        Returns:
            Dictionary with metrics counters
        """
        return dict(self._metrics)

    def clear_metrics(self) -> None:
        """Reset all metrics counters."""
        self._metrics.clear()


class TypedMessageBus(MessageBus):
    """Type-safe message bus with handler registration by message class.

    This variant allows subscribing using the message class directly
    for better IDE support and type checking.

    Example:
        bus = TypedMessageBus()

        @bus.on(TextInput)
        async def handle_text(msg: TextInput) -> None:
            print(f"Received: {msg.text}")

        await bus.emit(TextInput(text="Hello"))
    """

    def on(
        self,
        message_class: type[T],
        handler: Callable[[T], Awaitable[None]],
    ) -> Callable[[], None]:
        """Subscribe a handler to a specific message class.

        Args:
            message_class: The message class to subscribe to
            handler: Async handler function

        Returns:
            Unsubscribe function
        """
        # Get the message type from the class
        dummy_instance = message_class.model_construct()
        msg_type_value = dummy_instance.get_type_value()
        msg_type = MessageType(msg_type_value)

        # Wrap handler to cast type
        async def wrapper(msg: BaseMessage) -> None:
            await handler(msg)  # type: ignore

        return self.subscribe(msg_type, wrapper)

    async def emit(self, message: T) -> None:
        """Emit a message to subscribers.

        Args:
            message: The message to emit
        """
        await self.publish(message)


def create_message_bus(with_logging: bool = True) -> MessageBus:
    """Create a new message bus with optional default middleware.

    Args:
        with_logging: Whether to add logging middleware

    Returns:
        Configured MessageBus instance
    """
    bus = MessageBus()

    if with_logging:

        async def logging_middleware(
            msg: BaseMessage,
            next_handler: Callable,
        ) -> None:
            logger.debug(f"→ Publishing {msg.type} (api_version={msg.api_version})")
            await next_handler()

        bus.add_middleware(logging_middleware)

    return bus
