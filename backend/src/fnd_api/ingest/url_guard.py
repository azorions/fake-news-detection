"""Server-side request forgery guard: only public http(s) addresses may be fetched.

``validate_url`` must be called for the first URL and again for every redirect target.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from fnd_api.ingest import FetchBlocked, FetchFailed

Resolver = Callable[[str, int], Awaitable[list[str]]]

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_NOT_PUBLIC = "This address is not on the public internet, so it cannot be fetched."


async def system_resolver(host: str, port: int) -> list[str]:
    """All addresses the operating system resolves ``host`` to."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_global and not (
        address.is_multicast
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
    )


def is_public_ip(ip: str) -> bool:
    """True only for globally routable unicast addresses. Anything unparseable is not public."""
    try:
        address = ipaddress.ip_address(ip.strip())
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        # Forms that carry an IPv4 address inside are never legitimate answers for a web host,
        # except NAT64, which IPv6-only networks use to reach IPv4 sites.
        if address.ipv4_mapped or address.sixtofour or address.teredo:
            return False
        if address in _NAT64:
            return _is_public(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    return _is_public(address)


async def validate_url(
    url: str, *, resolver: Resolver = system_resolver, allow_private: bool = False
) -> None:
    """Raise ``FetchBlocked`` unless ``url`` is an http(s) URL without credentials whose host
    resolves only to public addresses. ``allow_private`` skips the address check (tests only)."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError as error:
        raise FetchBlocked("The URL is not valid.") from error
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise FetchBlocked("Only http and https URLs can be analysed.")
    if parts.username is not None or parts.password is not None:
        raise FetchBlocked("URLs that contain a user name or password are not accepted.")
    if not host:
        raise FetchBlocked("The URL has no host name.")
    if allow_private:
        return

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not is_public_ip(host):
            raise FetchBlocked(_NOT_PUBLIC)
        return

    try:
        addresses = await resolver(host, port or (443 if scheme == "https" else 80))
    except (OSError, UnicodeError) as error:
        # UnicodeError: a host name the resolver cannot encode (an empty or over-long label).
        raise FetchFailed("The site's address could not be found.") from error
    if not addresses:
        raise FetchFailed("The site's address could not be found.")
    if not all(is_public_ip(address) for address in addresses):
        raise FetchBlocked(_NOT_PUBLIC)
