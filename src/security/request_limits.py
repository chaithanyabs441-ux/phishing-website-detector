"""Small in-process scan rate limiter and ASGI request-body cap."""

import json
import time
from collections import defaultdict, deque


class ScanRateLimiter:
    def __init__(self, limit=30, window_seconds=60, clock=time.monotonic):
        self.limit = int(limit)
        self.window_seconds = float(window_seconds)
        self.clock = clock
        self.events = defaultdict(deque)

    def allow(self, key):
        now = self.clock()
        events = self.events[key]
        while events and now - events[0] >= self.window_seconds:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        return True


class MaxBodySizeMiddleware:
    def __init__(self, app, max_bytes, paths=("/scan", "/api/v1/scan")):
        self.app = app
        self.max_bytes = int(max_bytes)
        self.paths = set(paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in self.paths:
            return await self.app(scope, receive, send)
        messages = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    body = json.dumps({"detail": "Request body exceeds the configured limit"}).encode()
                    await send({
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
                    })
                    await send({"type": "http.response.body", "body": body})
                    return
                messages.append(message)
                if not message.get("more_body", False):
                    break
        index = 0

        async def replay_receive():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)
