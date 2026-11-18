"""The Meta Graph API's catalog endpoints: push listings, then check the result.

`MetaCatalogClient` wraps an `httpx.AsyncClient` the caller owns, so tests hand it
a `MockTransport` and a real run hands it a plain client. It follows the same
rules as the image calls in scripts/generate.py: retry the provider's bad minute
(network errors, 5xx, 429), fail a 4xx at once because retrying a wrong request
only repeats it, and never write the access token anywhere.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from product_catalog.listings import MetaListing

GRAPH_URL = "https://graph.facebook.com"
DEFAULT_API_VERSION = "v23.0"
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 5
MAX_BATCH = 1000  # requests per items_batch call; Meta's own ceiling is higher
POLL_INTERVAL = 5.0
MAX_POLLS = 24


def _no_log(event: dict[str, Any]) -> None:
    """Default sink: drop events."""


class MetaCatalogClient:
    """Pushes `MetaListing`s to one catalog and reads back how the push went."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        catalog_id: str,
        access_token: str,
        *,
        api_version: str = DEFAULT_API_VERSION,
        log: Callable[[dict[str, Any]], None] = _no_log,
        retry_base_delay: float = RETRY_BASE_DELAY,
    ) -> None:
        self._client = client
        self._catalog_id = catalog_id
        self._token = access_token
        self._version = api_version
        self._log = log
        self._retry_base_delay = retry_base_delay

    def _url(self, path: str) -> str:
        return f"{GRAPH_URL}/{self._version}/{self._catalog_id}/{path}"

    async def _request(self, method: str, event: str, url: str, **kwargs: Any) -> dict[str, Any]:
        """One Graph call with retries. The token travels only in the header."""
        headers = {"Authorization": f"Bearer {self._token}"}
        failure = "no attempt made"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.request(method, url, headers=headers, **kwargs)
            except httpx.HTTPError as error:
                failure = f"connection failed: {error!s}"
                self._log({"event": event, "attempt": attempt, "error": failure})
            else:
                if response.is_success:
                    self._log({"event": event, "attempt": attempt, "status": response.status_code})
                    return response.json()
                failure = f"meta {response.status_code}: {response.text[:300]}"
                self._log(
                    {"event": event, "attempt": attempt, "status": response.status_code, "error": failure}
                )
                if response.status_code < 500 and response.status_code != 429:
                    raise ValueError(failure)

            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(self._retry_base_delay * 2 ** (attempt - 1))

        raise ValueError(f"{MAX_ATTEMPTS} attempts failed - {failure}")

    async def push(self, listings: Sequence[MetaListing]) -> list[str]:
        """Upsert listings through `items_batch`. Returns the batch handles, one per chunk."""
        handles: list[str] = []
        for start in range(0, len(listings), MAX_BATCH):
            payload = MetaListing.batch_payload(listings[start : start + MAX_BATCH])
            body = await self._request(
                "POST",
                "items_batch",
                self._url("items_batch"),
                data={
                    "item_type": "PRODUCT_ITEM",
                    "allow_upsert": "true",
                    "requests": json.dumps(payload["requests"], ensure_ascii=False),
                },
            )
            chunk = body.get("handles") or []
            if not chunk:
                raise ValueError(f"meta returned no batch handle: {json.dumps(body)[:300]}")
            handles.extend(chunk)
        return handles

    async def status(self, handle: str) -> dict[str, Any]:
        """The newest status entry for one batch handle."""
        body = await self._request(
            "GET",
            "check_batch_request_status",
            self._url("check_batch_request_status"),
            params={"handle": handle, "load_ids_of_invalid_requests": "true"},
        )
        entries = body.get("data") or []
        if not entries:
            raise ValueError(f"meta returned no status for handle {handle}")
        return entries[0]

    async def wait(
        self,
        handle: str,
        *,
        interval: float = POLL_INTERVAL,
        max_polls: int = MAX_POLLS,
    ) -> dict[str, Any]:
        """Poll a handle until Meta has processed it. Raises if it never finishes."""
        for _ in range(max_polls):
            entry = await self.status(handle)
            if entry.get("status") != "in_progress":
                return entry
            await asyncio.sleep(interval)
        raise ValueError(f"batch {handle} still in progress after {max_polls} checks")
