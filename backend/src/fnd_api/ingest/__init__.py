"""Getting from a URL to article text: URL checks, fetching, extraction, language detection.

The errors below carry the API's stable error code and a message that is safe to show to the user
(no internal addresses, no exception text).
"""

from __future__ import annotations


class IngestError(Exception):
    code = "fetch_failed"
    status = 502

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class FetchFailed(IngestError):
    """The page could not be retrieved or is not a web page."""


class FetchBlocked(IngestError):
    """The URL is not allowed: wrong scheme, embedded credentials, or a non-public address."""

    code = "fetch_blocked"
    status = 422


class PayloadTooLarge(IngestError):
    code = "payload_too_large"
    status = 413
