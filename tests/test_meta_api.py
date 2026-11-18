"""Tests for the Meta catalog client - retries, error handling, payload shape, token hygiene."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from urllib.parse import parse_qs

import httpx
import pytest

from product_catalog import MetaListing
from product_catalog import meta_api
from product_catalog.meta_api import MetaCatalogClient

TOKEN = "secret-token-123"


def _listing(id: str = "chaise-camping") -> MetaListing:
    return MetaListing(
        id=id,
        title="Chaise Camping",
        description="Chaise pliante.",
        price=Decimal("90"),
        link="https://example.com/chaise",
        image_urls=["https://i/1.jpg"],
    )


def _run(handler, action, events: list | None = None):
    """Run `action(client)` against a MockTransport answering with `handler`."""

    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = MetaCatalogClient(
                http, "123", TOKEN, log=(events.append if events is not None else lambda e: None),
                retry_base_delay=0,
            )
            return await action(client)

    return asyncio.run(go())


def test_push_posts_the_batch_requests_as_a_form() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"handles": ["h1"]})

    handles = _run(handler, lambda c: c.push([_listing()]))

    assert handles == ["h1"]
    request = seen[0]
    assert request.url.path == f"/{meta_api.DEFAULT_API_VERSION}/123/items_batch"
    form = parse_qs(request.content.decode())
    assert form["item_type"] == ["PRODUCT_ITEM"]
    assert form["allow_upsert"] == ["true"]
    requests = json.loads(form["requests"][0])
    assert requests[0]["data"]["id"] == "chaise-camping"
    assert requests[0]["data"]["price"] == "90.00 MAD"


def test_the_token_goes_in_the_header_and_never_in_the_url_or_log() -> None:
    seen: list[httpx.Request] = []
    events: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"handles": ["h1"]})

    _run(handler, lambda c: c.push([_listing()]), events)

    assert seen[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in str(seen[0].url)
    assert TOKEN not in json.dumps(events)


def test_a_large_push_is_split_into_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(meta_api, "MAX_BATCH", 2)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"handles": [f"h{calls}"]})

    listings = [_listing(f"p{n}") for n in range(5)]
    handles = _run(handler, lambda c: c.push(listings))

    assert handles == ["h1", "h2", "h3"]


def test_transient_failures_are_retried() -> None:
    answers = [httpx.Response(429, text="slow down"), httpx.Response(503, text="down")]
    answers.append(httpx.Response(200, json={"handles": ["h1"]}))

    handles = _run(lambda request: answers.pop(0), lambda c: c.push([_listing()]))

    assert handles == ["h1"]


def test_a_dropped_connection_is_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json={"handles": ["h1"]})

    assert _run(handler, lambda c: c.push([_listing()])) == ["h1"]


def test_a_client_error_fails_at_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad field")

    with pytest.raises(ValueError, match="meta 400: bad field"):
        _run(handler, lambda c: c.push([_listing()]))
    assert calls == 1


def test_giving_up_after_the_last_attempt() -> None:
    with pytest.raises(ValueError, match="3 attempts failed"):
        _run(lambda request: httpx.Response(500, text="boom"), lambda c: c.push([_listing()]))


def test_a_response_without_handles_is_an_error() -> None:
    with pytest.raises(ValueError, match="no batch handle"):
        _run(lambda request: httpx.Response(200, json={}), lambda c: c.push([_listing()]))


def test_wait_polls_until_the_batch_is_processed() -> None:
    answers = [
        {"data": [{"handle": "h1", "status": "in_progress"}]},
        {"data": [{"handle": "h1", "status": "finished", "errors_total_count": 0}]},
    ]
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=answers.pop(0))

    entry = _run(handler, lambda c: c.wait("h1", interval=0))

    assert entry["status"] == "finished"
    assert seen[0].url.params["handle"] == "h1"
    assert seen[0].url.path.endswith("/check_batch_request_status")


def test_wait_gives_up_on_a_batch_that_never_finishes() -> None:
    body = {"data": [{"handle": "h1", "status": "in_progress"}]}

    with pytest.raises(ValueError, match="still in progress"):
        _run(lambda request: httpx.Response(200, json=body), lambda c: c.wait("h1", interval=0, max_polls=2))
