"""
Event System for BabyAGI

Lightweight event emitter that enables decoupled communication between
agent internals and output handlers (CLI, UI, logging, etc.).

Events:
    Agent events:
        - agent_response: Agent finished generating a response
        - tool_start: Tool execution starting
        - tool_end: Tool execution completed
        - objective_start: Background objective starting
        - objective_end: Background objective completed

    Scheduler events:
        - task_start: Scheduled task starting
        - task_end: Scheduled task completed

Usage:
    class MyAgent(EventEmitter):
        def process(self):
            self.emit("tool_start", {"name": "memory", "input": {...}})
            result = self.tools["memory"].execute(...)
            self.emit("tool_end", {"name": "memory", "result": result})

    agent = MyAgent()
    agent.on("tool_start", lambda e: print(f"Running {e['name']}..."))
"""

from typing import Callable, Any


class EventEmitter:
    """
    Mixin class that provides event emission and subscription.

    Designed to be lightweight and easy to integrate into existing classes.
    Supports multiple subscribers per event and wildcard subscriptions.
    """

    def __init_events__(self):
        """Initialize event storage. Call this in your __init__ if using as mixin."""
        if not hasattr(self, '_event_handlers'):
            self._event_handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable[[dict[str, Any]], None] = None) -> Callable:
        """
        Subscribe to an event.

        Args:
            event: Event name (e.g., "tool_start") or "*" for all events
            handler: Callback function that receives event data dict (optional for decorator use)

        Returns:
            The handler (for chaining or later removal), or a decorator if handler is None

        Example:
            agent.on("tool_start", lambda e: print(f"Tool: {e['name']}"))
            
            # Or as a decorator:
            @agent.on("tool_start")
            def handle_tool_start(e):
                print(f"Tool: {e['name']}")
        """
        self.__init_events__()
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        
        if handler is None:
            def decorator(fn: Callable[[dict[str, Any]], None]) -> Callable:
                self._event_handlers[event].append(fn)
                return fn
            return decorator
        
        self._event_handlers[event].append(handler)
        return handler

    def off(self, event: str, handler: Callable = None):
        """
        Unsubscribe from an event.

        Args:
            event: Event name
            handler: Specific handler to remove, or None to remove all
        """
        self.__init_events__()
        if event not in self._event_handlers:
            return
        if handler is None:
            self._event_handlers[event] = []
        else:
            self._event_handlers[event] = [h for h in self._event_handlers[event] if h != handler]

    def emit(self, event: str, data: dict[str, Any] = None):
        """
        Emit an event to all subscribers.

        Args:
            event: Event name
            data: Event data dictionary

        Events are delivered synchronously. Handlers should be fast
        to avoid blocking the main loop.
        """
        self.__init_events__()
        data = data or {}
        data['_event'] = event  # Include event name in data

        # Call specific event handlers
        for handler in self._event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                # Don't let handler errors crash the agent
                pass

        # Call wildcard handlers
        for handler in self._event_handlers.get('*', []):
            try:
                handler(data)
            except Exception:
                pass

    def once(self, event: str, handler: Callable[[dict[str, Any]], None]) -> Callable:
        """
        Subscribe to an event for a single emission only.

        Args:
            event: Event name
            handler: Callback function

        Returns:
            Wrapper handler (for removal if needed)
        """
        def wrapper(data):
            self.off(event, wrapper)
            handler(data)

        return self.on(event, wrapper)
