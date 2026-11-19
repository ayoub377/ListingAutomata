"""LangGraph workflow: generated images and descriptions in, products in the Meta catalogue out.

    python scripts/publish.py --dry-run             # build every listing offline, push nothing
    python scripts/publish.py --name chaise-camping # one product
    python scripts/publish.py                       # every row in products.csv
    python scripts/publish.py --lang ar             # Arabic descriptions instead of French
    python scripts/publish.py --no-push             # host + write the files, skip Meta
    python scripts/publish.py --force               # re-upload images already hosted

Run scripts/workflow.py first: this picks up its output/<name>/ folders.
Each product runs through a three-node graph:

    load_product   - reads output/<name>/description.json and the staged shots
    upload_images  - hosts the shots on PixelVault (reusing output/<name>/hosted.json)
    build_listing  - turns the row, the text and the hosted URLs into a MetaListing

The listings of the whole run then go to Meta in one items_batch call (upsert, so a
rerun updates items instead of duplicating them) and the batch is polled until Meta
reports on it. The payload and a CSV feed are also written to output/ so a push can
be inspected, or the feed uploaded by hand in Commerce Manager.
One product failing is reported and does not stop the others.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple, TypedDict
from urllib.parse import quote

from langgraph.graph import END, START, StateGraph

from product_catalog import (
    MetaListing,
    ProductRow,
    flush,
    load_products_csv,
    new_batch_id,
    product_name,
    run_config,
    slugify,
)
from product_catalog.meta_api import DEFAULT_API_VERSION, MetaCatalogClient
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # noqa: E402  (sibling script; needs the path line above)
logger = logging.getLogger(__name__)
WHATSAPP_MESSAGE = {
    "fr": "Bonjour, je suis intéressé par : {title}",
    "ar": "مرحبا، أنا مهتم بـ : {title}",
}


class PublishState(TypedDict, total=False):
    """What the three nodes pass along. Inputs first, then each node's output."""

    name: str  # the listing id: product_name(row)
    folder_name: str  # where the output lives, which may be the photo folder's name
    row: ProductRow
    lang: str
    force: bool
    title: str
    description: str
    images: list[Path]  # staged shots on disk, hero first
    image_urls: list[str]  # the same shots hosted, same order
    listing: MetaListing


class MetaConfig(NamedTuple):
    catalog_id: str | None
    access_token: str | None
    api_version: str
    whatsapp_number: str | None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def whatsapp_link(number: str | None, title: str, lang: str = "fr") -> str:
    """A wa.me link that opens a chat with the product's name already typed in."""
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        raise ValueError("no product link: set WHATSAPP_NUMBER in .env or fill the link column")
    message = WHATSAPP_MESSAGE[lang].format(title=title)
    return f"https://wa.me/{digits}?text={quote(message)}"


def description_path(folder_name: str) -> Path:
    return generate.OUTPUT_DIR / folder_name / "description.json"


def write_manifest(folder_name: str, manifest: dict) -> None:
    destination = generate.manifest_path(folder_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_meta_config() -> MetaConfig:
    """Meta settings from .env (generate.load_keys loads it) - names only, never printed."""
    generate.load_keys()
    return MetaConfig(
        catalog_id=os.environ.get("META_CATALOG_ID") or None,
        access_token=os.environ.get("META_ACCESS_TOKEN") or None,
        api_version=os.environ.get("META_API_VERSION") or DEFAULT_API_VERSION,
        whatsapp_number=os.environ.get("WHATSAPP_NUMBER") or None,
    )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


async def load_product(state: PublishState) -> PublishState:
    """Read the generated text and find the staged shots on disk."""
    folder_name = state["folder_name"]
    path = description_path(folder_name)
    if not path.is_file():
        raise ValueError(f"{state['name']}: no {path.name} - run scripts/workflow.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    description = data.get(state["lang"])
    if not description:
        raise ValueError(f"{state['name']}: {path.name} has no '{state['lang']}' description")

    images = [
        generate.output_path(folder_name, shot)
        for shot, _ in generate.SHOTS
        if generate.output_path(folder_name, shot).is_file()
    ]
    if not images:
        raise ValueError(f"{state['name']}: no staged images in {generate.OUTPUT_DIR / folder_name}")
    title = state["row"].title or data.get("title") or state["name"].replace("-", " ")
    return {"title": title, "description": description, "images": images}


def _hosted(manifest: dict, shot: str, filename: str) -> str | None:
    """The URL a shot was already hosted at, if it is the same file."""
    for entry in manifest.get("images", []):
        if entry.get("shot") == shot and entry.get("file") == filename:
            return entry.get("url")
    return None


def make_upload_images(api_key: str, log: Path | None = None):
    """The hosting node: PixelVault for every shot not already in hosted.json."""

    async def upload_images(state: PublishState) -> PublishState:
        import httpx

        folder_name = state["folder_name"]
        manifest = generate.read_manifest(folder_name)
        urls: list[str] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            for shot, _ in generate.SHOTS:
                path = generate.output_path(folder_name, shot)
                if path not in state["images"]:
                    continue
                url = None if state.get("force") else _hosted(manifest, shot, path.name)
                if url is None:
                    url = await generate.host(client, path, folder_name, api_key, log or generate.LOG_PATH)
                    manifest = generate.merge_manifest(manifest, shot, path.name, url)
                    write_manifest(folder_name, manifest)  # keep what was hosted if a later shot fails
                urls.append(url)
        return {"image_urls": urls}

    return upload_images


def make_plan_images():
    """Dry-run stand-in for hosting: known URLs where they exist, a placeholder otherwise."""

    async def plan_images(state: PublishState) -> PublishState:
        manifest = generate.read_manifest(state["folder_name"])
        return {
            "image_urls": [
                _hosted(manifest, path.stem.removeprefix("product_"), path.name)
                or f"UPLOAD://{state['folder_name']}/{path.name}"
                for path in state["images"]
            ]
        }

    return plan_images


def make_build_listing(whatsapp_number: str | None):
    """The listing node: the CSV row plus what the earlier nodes produced."""

    async def build_listing(state: PublishState) -> PublishState:
        row = state["row"]
        # A buyer on the green listing should be asking about the green one.
        label = f"{state['title']} ({row.color})" if row.color else state["title"]
        link = row.link or whatsapp_link(whatsapp_number, label, state["lang"])
        listing = MetaListing.from_row(
            row,
            id=state["name"],
            title=state["title"],
            description=state["description"],
            link=link,
            image_urls=state["image_urls"],
        )
        return {"listing": listing}

    return build_listing


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def build_graph(load_product_node, upload_images, build_listing):
    """START -> load_product -> upload_images -> build_listing -> END."""
    graph = StateGraph(PublishState)
    graph.add_node("load_product", load_product_node, metadata={"agent": "loader"})
    graph.add_node("upload_images", upload_images, metadata={"agent": "host"})
    graph.add_node("build_listing", build_listing, metadata={"agent": "listing"})
    graph.add_edge(START, "load_product")
    graph.add_edge("load_product", "upload_images")
    graph.add_edge("upload_images", "build_listing")
    graph.add_edge("build_listing", END)
    return graph.compile()


def default_graph(pixelvault_key: str, whatsapp_number: str | None, log: Path | None = None):
    """The production wiring: PixelVault for hosting."""
    return build_graph(load_product, make_upload_images(pixelvault_key, log), make_build_listing(whatsapp_number))


def dry_run_graph(whatsapp_number: str | None):
    """Same graph, nothing uploaded: validates every listing without spending anything."""
    return build_graph(load_product, make_plan_images(), make_build_listing(whatsapp_number))


# --------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------


def plan_inputs(
    rows: list[ProductRow], lang: str, force: bool, only: list[str] | None = None
) -> tuple[list[PublishState], list[str]]:
    """Graph inputs for every row with generated output, and the names without any.

    The output folder is filed under the product's name or, when workflow.py was
    run from a photo folder, under that folder's name - the same aliasing
    generate.py applies.
    """
    inputs: list[PublishState] = []
    missing: list[str] = []
    for row in rows:
        name = product_name(row)
        alias = row.folder_alias  # None for a colour variant: its folder is its own
        # --name chaise-camping publishes every colour of it; --name chaise-camping-vert, one.
        known_as = {name, row.group, slugify(row.product_folder.name)} - {None}
        if only and not known_as & set(only):
            continue
        folder_name = next(
            (c for c in (name, alias) if c and description_path(c).is_file()), None
        )
        if folder_name is None:
            missing.append(name)
            continue
        inputs.append(
            {"name": name, "folder_name": folder_name, "row": row, "lang": lang, "force": force}
        )
    return inputs, missing


async def run_product(graph, state: PublishState, batch_id: str | None = None) -> PublishState:
    """One product through the graph. Failures propagate; the batch decides what to do."""
    return await graph.ainvoke(state, config=run_config(state, batch_id or new_batch_id()))


async def run_batch(
    graph, inputs: list[PublishState], concurrency: int
) -> tuple[list[MetaListing], list[str]]:
    """Run every input through `run_product`. Returns (listings built, names that failed).

    `return_exceptions` keeps one bad product from cancelling the rest.
    """
    limit = asyncio.Semaphore(concurrency)
    batch_id = new_batch_id()

    async def one(state: PublishState) -> PublishState:
        async with limit:
            return await run_product(graph, state, batch_id)

    results = await asyncio.gather(*(one(state) for state in inputs), return_exceptions=True)
    listings: list[MetaListing] = []
    failed: list[str] = []
    for state, result in zip(inputs, results):
        if isinstance(result, Exception):
            failed.append(state["name"])
            print(f"failed {state['name']}: {result}", file=sys.stderr)
            continue
        listings.append(result["listing"])
        print(f"built  {state['name']}")
    return listings, failed


def write_files(listings: list[MetaListing]) -> tuple[Path, Path]:
    """The Batch API body and the CSV feed, side by side in output/."""
    generate.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch = generate.OUTPUT_DIR / "meta-batch.json"
    batch.write_text(
        json.dumps(MetaListing.batch_payload(listings), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    feed = generate.OUTPUT_DIR / "meta-feed.csv"
    with feed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MetaListing.feed_columns)
        writer.writeheader()
        writer.writerows(listing.to_feed_row() for listing in listings)
    return batch, feed


def report_status(handle: str, entry: dict) -> bool:
    """Print how Meta processed one batch. True when every item went in."""
    errors = entry.get("errors") or []
    invalid = entry.get("ids_of_invalid_requests") or []
    status = entry.get("status")
    clean = status != "failed" and not errors and not invalid
    print(f"meta   batch {handle}: {status}, {entry.get('errors_total_count', len(errors))} error(s)")
    for error in errors:
        print(f"       {error.get('message') or json.dumps(error)}", file=sys.stderr)
    if invalid:
        print(f"       rejected items: {', '.join(str(item) for item in invalid)}", file=sys.stderr)
    return clean


async def push_to_meta(listings: list[MetaListing], config: MetaConfig) -> bool:
    """Send every listing to the catalogue and wait for Meta's verdict. True on a clean push."""
    import httpx

    assert config.catalog_id and config.access_token  # main() checks before calling
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)) as http:
        client = MetaCatalogClient(
            http,
            config.catalog_id,
            config.access_token,
            api_version=config.api_version,
            log=lambda event: generate.log_event({**event, "product": "meta"}),
        )
        handles = await client.push(listings)
        statuses = [(handle, await client.wait(handle)) for handle in handles]
    return all([report_status(handle, entry) for handle, entry in statuses])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logger.info('Started')
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=generate.PROJECT_ROOT / "products.csv")
    parser.add_argument(
        "--name", action="append", dest="names",
        help="Publish only this product (output name or photo folder name; repeatable).",
    )
    parser.add_argument("--lang", choices=sorted(WHATSAPP_MESSAGE), default="fr", help="Description language.")
    parser.add_argument("--force", action="store_true", help="Re-upload images already hosted.")
    parser.add_argument("--no-push", action="store_true", help="Host and write files, skip Meta.")
    parser.add_argument("--dry-run", action="store_true", help="Build listings offline; upload and push nothing.")
    parser.add_argument(
        "--concurrency", type=int, default=generate.DEFAULT_CONCURRENCY,
        help="Products in flight at once.",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    try:
        rows = load_products_csv(args.csv).listings()
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    inputs, missing = plan_inputs(rows, args.lang, args.force, args.names)
    for name in missing:
        print(f"skip   {name} (no description.json - run scripts/workflow.py first)", file=sys.stderr)
    if not inputs:
        print("nothing to do", file=sys.stderr)
        return 1

    config = load_meta_config()
    pixelvault_key = generate.load_keys()[1]
    if not args.dry_run:
        needed = {"PIXELVAULT_API_KEY": pixelvault_key}
        if not args.no_push:
            needed |= {"META_CATALOG_ID": config.catalog_id, "META_ACCESS_TOKEN": config.access_token}
        if unset := [key for key, value in needed.items() if not value]:
            print(f"{', '.join(unset)} not set (.env or environment)", file=sys.stderr)
            return 1

    graph = (
        dry_run_graph(config.whatsapp_number)
        if args.dry_run
        else default_graph(pixelvault_key, config.whatsapp_number)
    )
    try:
        listings, failed = asyncio.run(run_batch(graph, inputs, args.concurrency))
    finally:
        flush()  # traces upload in the background; don't exit before the last ones go

    pushed = True
    if listings and args.dry_run:
        # Placeholder URLs must not overwrite a real payload on disk, so show one instead.
        print(json.dumps(listings[0].to_batch_request(), ensure_ascii=False, indent=2))
    elif listings:
        batch_file, feed_file = write_files(listings)
        print(f"wrote  {generate.relative(batch_file)}  {generate.relative(feed_file)}")
        if not args.no_push:
            try:
                pushed = asyncio.run(push_to_meta(listings, config))
            except ValueError as error:
                print(f"meta push failed: {error}", file=sys.stderr)
                pushed = False
    print(f"{len(listings)} built, {len(failed)} failed, {len(missing)} skipped")
    return 0 if pushed and not failed and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
