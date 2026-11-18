"""Pushing one approved product to the Meta catalogue.

`MetaPusher` reuses scripts/publish.py (image hosting, the WhatsApp link, .env
settings) and the `MetaCatalogClient`, so the review app and the CLI publish the
same way. The scripts folder is not a package, so it is imported by path, the
same trick scripts/publish.py itself uses for `generate`.

Each product goes to Meta as its own `items_batch` call. That keeps a failure
attributable to one product and lets the reviewer retry just that one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from product_catalog.listings import MetaListing
from product_catalog.meta_api import MetaCatalogClient
from product_catalog.review.catalog import ReviewProduct

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


@dataclass(frozen=True)
class PushResult:
    ok: bool
    message: str


class Pusher(Protocol):
    """What the app needs from a publisher. Tests swap in a fake."""

    def missing(self) -> list[str]:
        """Names of the settings that must be filled in before anything can be pushed."""

    async def push(self, product: ReviewProduct) -> PushResult: ...


def build_listing(product: ReviewProduct, image_urls: list[str], link: str) -> MetaListing:
    """The listing exactly as the reviewer left it: their edits, the CSV's stock and condition."""
    return MetaListing.from_row(
        product.row,
        id=product.name,
        title=product.title,
        description=product.description,
        price=product.price,
        link=link,
        image_urls=image_urls,
    )


def describe_status(handles: list[str], entries: list[dict[str, Any]]) -> PushResult:
    """Turn Meta's batch verdicts into one result. Clean means every item went in."""
    problems: list[str] = []
    for handle, entry in zip(handles, entries):
        for error in entry.get("errors") or []:
            problems.append(error.get("message") or str(error))
        invalid = entry.get("ids_of_invalid_requests") or []
        if invalid:
            problems.append(f"Meta rejected: {', '.join(str(item) for item in invalid)}")
        if entry.get("status") == "failed" and not problems:
            problems.append(f"batch {handle} failed")
    if problems:
        return PushResult(False, "; ".join(problems))
    return PushResult(True, "Pushed to Meta")


def needs_rehost(folder: Path, images: list[Path]) -> bool:
    """True when a shot was regenerated after it was last hosted.

    hosted.json reuses a URL for any file with the same name, which is right for a
    rerun and wrong for a regenerated image: the reviewer approved the new picture,
    so the old hosted copy must not go out. hosted.json is rewritten after every
    upload, so an image newer than it has never been hosted.
    """
    manifest = folder / "hosted.json"
    if not manifest.is_file():
        return True
    hosted_at = manifest.stat().st_mtime
    return any(image.stat().st_mtime > hosted_at for image in images)


class MetaPusher:
    """Hosts a product's images on PixelVault, then upserts it into the catalogue.

    Reads the same `.env` settings as scripts/publish.py. Images are looked up through
    `generate.OUTPUT_DIR`, so this must run against the project's own output folder.
    """

    def __init__(self) -> None:
        self._publish: Any = None

    def _module(self) -> Any:
        if self._publish is None:
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_DIR))
            import publish

            self._publish = publish
        return self._publish

    def missing(self) -> list[str]:
        publish = self._module()
        config = publish.load_meta_config()
        needed = {
            "META_CATALOG_ID": config.catalog_id,
            "META_ACCESS_TOKEN": config.access_token,
            "PIXELVAULT_API_KEY": publish.generate.load_keys()[1],
        }
        return [name for name, value in needed.items() if not value]

    async def push(self, product: ReviewProduct) -> PushResult:
        publish = self._module()
        generate = publish.generate
        config = publish.load_meta_config()
        pixelvault_key = generate.load_keys()[1]
        if not (config.catalog_id and config.access_token and pixelvault_key):
            return PushResult(False, f"not configured: set {', '.join(self.missing())} in .env")

        images = [
            generate.output_path(product.folder_name, shot)
            for shot, _ in generate.SHOTS
            if generate.output_path(product.folder_name, shot).is_file()
        ]
        try:
            upload = publish.make_upload_images(pixelvault_key)
            hosted = await upload(
                {
                    "name": product.name,
                    "folder_name": product.folder_name,
                    "row": product.row,
                    "lang": product.lang,
                    "force": needs_rehost(product.folder, images),
                    "images": images,
                }
            )
            link = product.row.link or publish.whatsapp_link(
                config.whatsapp_number, product.title, product.lang
            )
            listing = build_listing(product, hosted["image_urls"], link)

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)) as http:
                client = MetaCatalogClient(
                    http,
                    config.catalog_id,
                    config.access_token,
                    api_version=config.api_version,
                    log=lambda event: generate.log_event({**event, "product": "meta"}),
                )
                handles = await client.push([listing])
                entries = [await client.wait(handle) for handle in handles]
        except (ValueError, OSError, httpx.HTTPError) as error:
            return PushResult(False, str(error))
        return describe_status(handles, entries)
