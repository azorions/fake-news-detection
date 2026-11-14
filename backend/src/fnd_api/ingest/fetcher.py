"""Fetch one web page defensively: checked redirects, a size cap, a deadline, text types only."""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin

import anyio
import httpx

from fnd_api.config import Settings
from fnd_api.ingest import FetchBlocked, FetchFailed, PayloadTooLarge
from fnd_api.ingest.url_guard import Resolver, is_public_ip, system_resolver, validate_url

MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ACCEPTED_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8",
    "Accept-Language": "en",
}
_META_CHARSET = re.compile(rb"<meta[^>]{0,200}?charset\s*=\s*[\"']?\s*([\w.:-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class FetchedPage:
    url: str  # as requested
    final_url: str  # after redirects
    status: int
    html: str
    fetched_at: datetime


def _check_peer(response: httpx.Response) -> None:
    """Refuse a connection that ended up at a non-public address.

    The host name is resolved once for the check and again by the HTTP client, and a hostile DNS
    server can answer differently the second time. The transport reports the address it really
    connected to, so the response is dropped before any of its body is read. (Behind an outbound
    proxy the peer is the proxy; such deployments must keep the proxy on a public address or set
    ``allow_private_urls``.)
    """
    stream = response.extensions.get("network_stream")
    address = stream.get_extra_info("server_addr") if stream is not None else None
    if address and not is_public_ip(str(address[0])):
        raise FetchBlocked("This address is not on the public internet, so it cannot be fetched.")


def _decode(body: bytes, content_type: str) -> str:
    charset = ""
    for parameter in content_type.split(";")[1:]:
        name, _, value = parameter.partition("=")
        if name.strip().lower() == "charset":
            charset = value.strip().strip("\"'")
    if not charset:
        declared = _META_CHARSET.search(body[:4096])
        charset = declared.group(1).decode("ascii", "replace") if declared else "utf-8"
    try:
        codecs.lookup(charset)
    except LookupError:
        charset = "utf-8"
    return body.decode(charset, errors="replace")


async def _read_capped(response: httpx.Response, limit: int) -> bytes:
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise PayloadTooLarge("The page is larger than the size limit.")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > limit:
            raise PayloadTooLarge("The page is larger than the size limit.")
    return bytes(body)


async def _fetch(
    url: str, client: httpx.AsyncClient, settings: Settings, resolver: Resolver
) -> FetchedPage:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        await validate_url(current, resolver=resolver, allow_private=settings.allow_private_urls)
        async with client.stream(
            "GET",
            current,
            headers=_REQUEST_HEADERS,
            follow_redirects=False,
            timeout=settings.fetch_timeout_s,
        ) as response:
            if not settings.allow_private_urls:
                _check_peer(response)
            status = response.status_code
            if status in _REDIRECT_STATUSES:
                location = response.headers.get("location")
                if not location:
                    raise FetchFailed("The site sent a redirect without a destination.")
                current = urljoin(current, location)
                continue
            if status >= 400 or status < 200:
                raise FetchFailed(f"The site answered with an error (HTTP {status}).")
            content_type = response.headers.get("content-type", "")
            if content_type.split(";")[0].strip().lower() not in _ACCEPTED_TYPES:
                raise FetchFailed("The URL does not point to a web page.")
            body = await _read_capped(response, settings.max_fetch_bytes)
            return FetchedPage(
                url=url,
                final_url=current,
                status=status,
                html=_decode(body, content_type),
                fetched_at=datetime.now(UTC),
            )
    raise FetchFailed("The site redirected too many times.")


async def fetch_html(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    resolver: Resolver = system_resolver,
) -> FetchedPage:
    """Fetch ``url``. Every hop is validated; the whole fetch shares one deadline.

    Raises ``FetchBlocked``, ``FetchFailed`` or ``PayloadTooLarge``.
    """
    try:
        with anyio.fail_after(settings.fetch_timeout_s):
            return await _fetch(url, client, settings, resolver)
    except (TimeoutError, httpx.TimeoutException) as error:
        raise FetchFailed("The site took too long to answer.") from error
    except httpx.HTTPError as error:
        raise FetchFailed("The site could not be reached.") from error
