import pytest

from fnd_api.ingest import FetchBlocked, FetchFailed
from fnd_api.ingest.url_guard import is_public_ip, validate_url

HOSTS = {
    "public.example": ["93.184.216.34"],
    "internal.example": ["10.0.0.5"],
    "mixed.example": ["93.184.216.34", "127.0.0.1"],
    "v6.example": ["2606:2800:220:1:248:1893:25c8:1946"],
}


async def resolver(host: str, port: int) -> list[str]:
    if host not in HOSTS:
        raise OSError(f"cannot resolve {host}")
    return HOSTS[host]


@pytest.mark.parametrize(
    "ip",
    ["93.184.216.34", "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946"],
)
def test_public_addresses(ip: str) -> None:
    assert is_public_ip(ip) is True


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.1",  # carrier-grade NAT
        "0.0.0.0",
        "224.0.0.1",  # multicast
        "240.0.0.1",  # reserved
        "255.255.255.255",
        "::1",
        "::",
        "fe80::1",
        "fe80::1%eth0",
        "fc00::1",
        "ff02::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:10.0.0.5",
        "64:ff9b::7f00:1",  # NAT64 form of 127.0.0.1
        "2002:7f00:1::",  # 6to4 form of 127.0.0.1
        "not an address",
        "",
    ],
)
def test_non_public_addresses(ip: str) -> None:
    assert is_public_ip(ip) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://public.example/",
        "https://public.example:8443/path?q=1",
        "http://v6.example/",
        "http://93.184.216.34/",
        "HTTPS://PUBLIC.EXAMPLE/Story",
    ],
)
async def test_public_urls_are_allowed(url: str) -> None:
    await validate_url(url, resolver=resolver)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8000/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
        "ftp://example.com",
        "file:///etc/passwd",
        "gopher://public.example/",
        "javascript:alert(1)",
        "http://user:pw@example.com",
        "http://user@public.example/",
        "http://internal.example/",
        "http://mixed.example/",  # one private address among the answers is enough to refuse
        "http:///no-host",
        "http://",
        "//public.example/",
        "public.example",
    ],
)
async def test_unsafe_urls_are_blocked(url: str) -> None:
    with pytest.raises(FetchBlocked):
        await validate_url(url, resolver=resolver)


async def test_blocked_message_does_not_reveal_internal_addresses() -> None:
    with pytest.raises(FetchBlocked) as blocked:
        await validate_url("http://internal.example/", resolver=resolver)
    assert "10.0.0.5" not in str(blocked.value)
    assert blocked.value.code == "fetch_blocked"


async def test_unresolvable_host_is_a_failed_fetch() -> None:
    with pytest.raises(FetchFailed):
        await validate_url("http://nowhere.example/", resolver=resolver)


async def test_private_addresses_can_be_allowed_but_schemes_cannot() -> None:
    await validate_url("http://127.0.0.1:8000/", resolver=resolver, allow_private=True)
    await validate_url("http://internal.example/", resolver=resolver, allow_private=True)
    with pytest.raises(FetchBlocked):
        await validate_url("ftp://127.0.0.1/", resolver=resolver, allow_private=True)
    with pytest.raises(FetchBlocked):
        await validate_url("http://user:pw@127.0.0.1/", resolver=resolver, allow_private=True)
