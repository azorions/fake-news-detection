"""Errors as ``application/problem+json`` with a stable ``code`` and the request ID."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from ulid import ULID

PROBLEM_JSON = "application/problem+json"
REQUEST_ID_HEADER = "X-Request-ID"

_CODE_BY_STATUS = {404: "not_found", 413: "payload_too_large", 429: "rate_limited"}


class ApiError(Exception):
    """An expected failure: ``code`` is one of the design's stable codes, ``detail`` is shown to users."""

    def __init__(
        self, code: str, status: int, detail: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.status = status
        self.detail = detail
        self.headers = headers


def new_request_id() -> str:
    return str(ULID())


def request_id_of(request: Request) -> str:
    """The ID the request middleware assigned (a fresh one if it has not run)."""
    request_id = getattr(request.state, "request_id", None)
    return request_id if isinstance(request_id, str) else new_request_id()


def problem_response(
    status: int,
    code: str,
    detail: str,
    request_id: str,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": HTTPStatus(status).phrase,
        "status": status,
        "code": code,
        "detail": detail,
        "request_id": request_id,
        **extra,
    }
    all_headers = {**(headers or {}), REQUEST_ID_HEADER: request_id}
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=all_headers)


def code_for_status(status: int) -> str:
    if status in _CODE_BY_STATUS:
        return _CODE_BY_STATUS[status]
    return "internal" if status >= 500 else "invalid_input"


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return problem_response(exc.status, exc.code, exc.detail, request_id_of(request), exc.headers)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # The rejected input is left out on purpose: it can be a whole article.
    errors = [
        {"loc": list(error["loc"]), "msg": str(error["msg"]), "type": str(error["type"])}
        for error in exc.errors()
    ]
    first = errors[0] if errors else {"loc": [], "msg": "invalid request"}
    where = ".".join(str(part) for part in first["loc"])
    detail = f"{where}: {first['msg']}" if where else str(first["msg"])
    return problem_response(422, "invalid_input", detail, request_id_of(request), errors=errors)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    detail = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
    headers = dict(exc.headers) if exc.headers else None
    code = code_for_status(exc.status_code)
    return problem_response(exc.status_code, code, detail, request_id_of(request), headers)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
