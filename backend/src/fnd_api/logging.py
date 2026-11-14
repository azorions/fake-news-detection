"""Structured logging and the middleware that gives every request an ID.

Request logs carry the method, path, status and duration only. Submitted text and URLs are never
logged.
"""

from __future__ import annotations

import logging
import re
import time

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fnd_api.api.errors import REQUEST_ID_HEADER, new_request_id, problem_response

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def configure_logging(json_logs: bool = True, level: str = "INFO") -> None:
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    renderer: structlog.typing.Processor
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


class RequestContextMiddleware:
    """Assign a request ID, echo it in ``X-Request-ID``, log the request, and turn an unexpected
    exception into a 500 problem response that does not reveal the cause."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._log = get_logger("fnd_api.request")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status = 500
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            self._log.exception("unhandled_error", method=scope["method"], path=scope["path"])
            if started:  # too late to change the response
                raise
            response = problem_response(500, "internal", "Internal server error.", request_id)
            await response(scope, receive, send_with_id)
        finally:
            self._log.info(
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
