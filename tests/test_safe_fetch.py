import pytest

from src.serving.safe_fetch import UnsafeTarget, fetch_html, validate_target


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"])
def test_rejects_non_public_ip_literals(address):
    with pytest.raises(UnsafeTarget):
        validate_target("http://{}/".format("[{}]".format(address) if ":" in address else address))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "http://example.com:8080", "http://user@example.com"])
def test_rejects_unsupported_targets(url):
    with pytest.raises(UnsafeTarget):
        validate_target(url, resolver=lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 80))])


def test_rejects_hostname_resolving_to_private_ip():
    def resolver(*args, **kwargs):
        return [(None, None, None, None, ("192.168.1.2", 80))]

    with pytest.raises(UnsafeTarget):
        validate_target("https://example.com", resolver=resolver)


def test_allows_public_https_hostname():
    def resolver(*args, **kwargs):
        return [(None, None, None, None, ("93.184.216.34", 443))]

    assert validate_target("https://example.com", resolver=resolver).hostname == "example.com"


def test_redirect_to_private_address_is_rejected_before_second_request():
    calls = []

    class Response:
        status_code = 302
        headers = {"location": "http://127.0.0.1/admin"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, **kwargs):
            calls.append(url)
            return Response()

    def resolver(*args, **kwargs):
        return [(None, None, None, None, ("93.184.216.34", 443))]

    with pytest.raises(UnsafeTarget):
        fetch_html("https://example.com", resolver=resolver, client_factory=Client)
    assert calls == ["https://example.com"]


def test_fetch_rejects_non_html_and_oversized_body():
    class Response:
        status_code = 200

        def __init__(self, content_type, chunks):
            self.headers = {"content-type": content_type}
            self.chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield from self.chunks

    class Client:
        response = None

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return self.response

    def resolver(*args, **kwargs):
        return [(None, None, None, None, ("93.184.216.34", 443))]

    Client.response = Response("text/plain", [b"x"])
    with pytest.raises(UnsafeTarget, match="not HTML"):
        fetch_html("https://example.com", resolver=resolver, client_factory=Client)
    Client.response = Response("text/html", [b"1234"])
    with pytest.raises(UnsafeTarget, match="size limit"):
        fetch_html("https://example.com", max_bytes=3, resolver=resolver, client_factory=Client)
