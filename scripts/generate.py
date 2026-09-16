"""Stage the segmented cutouts into catalogue photos with Seedream, then host them.

    python scripts/generate.py                     # every row in products.csv
    python scripts/generate.py --csv batch.csv
    python scripts/generate.py --name chaise       # one product, scenery from the CSV
    python scripts/generate.py --name chaise --scenery "forest near a lake"
    python scripts/generate.py --dry-run           # print the plan and cost, call nothing
    python scripts/generate.py --no-host           # generate only
    python scripts/generate.py --host-only         # upload what is already in output/

Reads the cutouts scripts/segment.py wrote to images/<product>/, sends them to
ByteDance Seedream 4.5 through OpenRouter as reference images, and writes two
staged shots — a hero angle and a wider context one — to output/<product>/.
The product's `scenery` cell decides where it is staged; without one it lands on
plain white.

--name matches a CSV row by its product name or its photo folder's name, so the
scenery comes along even when the cutouts were segmented under a short name.
--scenery overrides it; a product in neither place is staged on white and says so.

Each shot is then uploaded to PixelVault and its CDN URL recorded in
output/<product>/hosted.json, which is what the Meta feed step reads for
image_link and additional_image_link.

Every request, the prompt it carried and the response it got are appended to
output/generation-log.jsonl — one JSON object per line, no keys and no image
payloads, so a disappointing render can be traced back to the words that caused it.

Generation costs money ($0.04 per image), so a shot already on disk is skipped
unless --force is passed, and --dry-run prints the bill before anything is sent.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageOps

from product_catalog import (
    PRIMARY_MARKER,
    load_products_csv,
    product_name,
    slugify,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_PATH = OUTPUT_DIR / "generation-log.jsonl"

MODEL_ID = "bytedance-seed/seedream-4.5"
IMAGES_ENDPOINT = "https://openrouter.ai/api/v1/images"
PIXELVAULT_ENDPOINT = "https://api.pixelvault.dev/v1/images"

RESOLUTION = "2K"
ASPECT_RATIO = "1:1"
OUTPUT_FORMAT = "jpeg"
COST_PER_IMAGE = 0.04

# Seedream accepts 14 references; past a handful they cost tokens without adding
# fidelity, since the extra shots are the same product from the same afternoon.
MAX_REFERENCES = 6
# A phone cutout is a 3024x4032 RGBA PNG - 17 MB each, and base64 adds a third
# on top. Sending them raw built a ~47 MB request that Seedream timed out on
# (524) and that the next attempt had reset mid-upload. The model renders at 2K,
# so a reference longer than this on its edge buys nothing but upload time.
REFERENCE_MAX_EDGE = 1536
REFERENCE_QUALITY = 92
# 524s are upstream timeouts and usually clear on a retry.
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 5
DEFAULT_CONCURRENCY = 4
# PixelVault's free tier rejects anything larger, and says so late and vaguely.
PIXELVAULT_MAX_BYTES = 5 * 1024 * 1024

QUALITY = (
    "Ultra-realistic, visually stunning commercial product photograph. "
    "Sharp focus on the product, natural light, believable shadows where it "
    "meets the ground, rich colour, shallow depth of field."
)
FIDELITY = (
    "The product must match the reference images exactly - same shape, "
    "proportions, colour, materials, texture and markings. Do not redesign it, "
    "do not add or remove any part of it, do not invent a logo."
)
# The first run put a man with a cropped face in the frame, because the context
# shot used to ask for the product "in use". People are not banned - a scenery
# cell can ask for them, as the Bestway row's does - but they have to be whole.
FRAMING = (
    "Show the entire product, nothing cropped at the edges. Any person in the "
    "scene must be fully and naturally framed, never cut through the face or body."
)
CLEANUP = "No text, no watermark, no logo overlay, no border, no collage."

WHITE_BACKDROP = "Place it on a clean seamless white studio backdrop."
SCENE_TEMPLATE = (
    "Place the product in this setting: {scenery}. "
    "The setting must be clearly visible, richly detailed and fill the "
    "background - it is the point of the image, not a blurred suggestion. "
    "Light the product to match that setting."
)

SHOTS = (
    (
        "angle",
        "Hero three-quarter view from slightly above, the product large in frame.",
    ),
    (
        "context",
        "Wider establishing shot with the product sitting naturally in the scene, "
        "showing much more of the surroundings around it.",
    ),
)


class Product(NamedTuple):
    """One product to stage: what to call it, and where to put it."""

    name: str
    title: str
    scenery: str | None
    aliases: tuple[str, ...] = ()

    def answers_to(self, wanted: str) -> bool:
        """True when `wanted` names this product, by output name or photo folder."""
        return wanted == self.name or wanted in self.aliases


class Job(NamedTuple):
    """One image to generate: what to ask for, and where the result lands."""

    name: str
    shot: str
    prompt: str
    destination: Path


# --------------------------------------------------------------------------
# Naming and references
# --------------------------------------------------------------------------


def cutout_dir(name: str) -> Path:
    """Where segment.py left this product's cutouts."""
    return IMAGES_DIR / name


def _cutout_order(path: Path) -> tuple[int, int]:
    """Sort key: primary first, then numerically - `product_10` is not before `product_2`."""
    stem = path.stem.removeprefix("product_").removesuffix("_seg")
    if stem == PRIMARY_MARKER:
        return (0, 0)
    return (1, int(stem) if stem.isdigit() else sys.maxsize)


def cutout_name(product: Product) -> str | None:
    """Which of a product's names actually has cutouts on disk, if any.

    segment.py run from the CSV files them under the product name, but run with
    --folder it files them under the photo folder's name. Both spellings are the
    same product, so try the canonical one and then its aliases rather than
    reporting `chaise-camping` missing while `images/chaise/` sits there full.
    """
    return next(
        (name for name in (product.name, *product.aliases) if reference_cutouts(name)), None
    )


def reference_cutouts(name: str) -> list[Path]:
    """The cutouts to generate from, primary first, at most MAX_REFERENCES.

    The primary leads because it is the shot the user picked to build the scenes
    on; the numbered ones follow so the model sees the product's other sides and
    keeps them consistent.
    """
    folder = cutout_dir(name)
    if not folder.is_dir():
        return []
    cutouts = [child for child in folder.glob("product_*_seg.png") if child.is_file()]
    return sorted(cutouts, key=_cutout_order)[:MAX_REFERENCES]


def encode_reference(path: Path) -> bytes:
    """Shrink a cutout down to something a reference image should weigh.

    Two things happen here. The cutout is scaled so its long edge is at most
    REFERENCE_MAX_EDGE, which is where the 47 MB request came from. And its
    transparency is flattened onto white rather than shipped as alpha: the
    segmentation has already thrown the background away, and a product on white
    is unambiguous to the model in a way an alpha channel it may or may not read
    is not.
    """
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle)
        image.thumbnail((REFERENCE_MAX_EDGE, REFERENCE_MAX_EDGE), Image.LANCZOS)
        backdrop = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode in ("RGBA", "LA"):
            backdrop.paste(image, mask=image.split()[-1])
        else:
            backdrop.paste(image.convert("RGB"))

    buffer = io.BytesIO()
    backdrop.save(buffer, "JPEG", quality=REFERENCE_QUALITY)
    return buffer.getvalue()


@lru_cache(maxsize=MAX_REFERENCES * 2)
def data_url(path: Path) -> str:
    """A cutout as an inline reference - the files are local, so base64 it is.

    Cached because both shots of a product send the same references, and
    re-reading a 17 MB PNG to produce identical bytes is pure latency.
    """
    return f"data:image/jpeg;base64,{base64.b64encode(encode_reference(path)).decode()}"


# --------------------------------------------------------------------------
# Prompts and payloads
# --------------------------------------------------------------------------


def build_prompt(title: str, scenery: str | None, instruction: str) -> str:
    """The text half of one request - the only place prompt wording lives.

    The scene clause is emphatic on purpose: the first run returned plain white
    backgrounds, so the setting is now stated as the point of the image rather
    than mentioned in passing.
    """
    scene = SCENE_TEMPLATE.format(scenery=scenery.strip().rstrip(".")) if scenery else WHITE_BACKDROP
    return " ".join([QUALITY, f"Subject: {title}.", FIDELITY, instruction, scene, FRAMING, CLEANUP])


def build_payload(prompt: str, references: list[Path]) -> dict:
    """The OpenRouter /images request body."""
    return {
        "model": MODEL_ID,
        "prompt": prompt,
        "n": 1,
        "resolution": RESOLUTION,
        "aspect_ratio": ASPECT_RATIO,
        "output_format": OUTPUT_FORMAT,
        "input_references": [
            {"type": "image_url", "image_url": {"url": data_url(reference)}}
            for reference in references
        ],
    }


def decode_image(payload: dict) -> bytes:
    """Pull the image bytes out of a response - they arrive base64, never as a URL."""
    if error := payload.get("error"):
        message = error.get("message", error) if isinstance(error, dict) else error
        raise ValueError(f"openrouter refused the request: {message}")
    data = payload.get("data") or []
    if not data or not data[0].get("b64_json"):
        raise ValueError(f"no image in the response: {json.dumps(payload)[:300]}")
    return base64.b64decode(data[0]["b64_json"])


def output_path(name: str, shot: str) -> Path:
    """Where one staged shot lands."""
    return OUTPUT_DIR / name / f"product_{shot}.jpg"


def manifest_path(name: str) -> Path:
    return OUTPUT_DIR / name / "hosted.json"


def plan_generations(products: list[Product]) -> list[Job]:
    """Every (product, shot) pair the batch would generate, with its prompt.

    Built up front so the whole bill is known - and printable - before the first
    request goes out.
    """
    return [
        Job(
            product.name,
            shot,
            build_prompt(product.title, product.scenery, instruction),
            output_path(product.name, shot),
        )
        for product in products
        for shot, instruction in SHOTS
    ]


# --------------------------------------------------------------------------
# The log
# --------------------------------------------------------------------------


def log_event(event: dict, path: Path | None = None) -> None:
    """Append one JSON object to the run log.

    Deliberately never carries an API key, a base64 payload or image bytes: the
    log is for reading, and a line with 600 KB of base64 in it is not readable.
    """
    path = path or LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **event}, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summarize_response(payload: dict, image: bytes | None) -> dict:
    """The parts of a response worth keeping: usage and cost, never the pixels."""
    entry = payload.get("data", [{}])
    media_type = entry[0].get("media_type") if entry else None
    return {
        "usage": payload.get("usage"),
        "media_type": media_type,
        "image_bytes": len(image) if image else 0,
    }


# --------------------------------------------------------------------------
# The network, async
# --------------------------------------------------------------------------


async def generate(client, job: Job, references: list[Path], api_key: str, log: Path) -> bytes:
    """One image out of Seedream, retrying the transient upstream failures.

    5xx and dropped connections are the provider having a bad minute; a 4xx is
    the request being wrong, and retrying that just spends the same money twice.
    """
    import httpx

    payload = build_payload(job.prompt, references)
    request_bytes = len(json.dumps(payload))
    base = {
        "event": "generate",
        "product": job.name,
        "shot": job.shot,
        "model": MODEL_ID,
        "prompt": job.prompt,
        "references": [reference.name for reference in references],
        "request_bytes": request_bytes,
    }
    failure = "no attempt made"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            response = await client.post(
                IMAGES_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        except httpx.HTTPError as error:
            failure = f"connection failed: {error!s}"
            log_event({**base, "attempt": attempt, "elapsed_s": round(time.monotonic() - started, 1),
                       "error": failure}, log)
        else:
            elapsed = round(time.monotonic() - started, 1)
            if response.is_success:
                body = response.json()
                image = decode_image(body)
                log_event({**base, "attempt": attempt, "elapsed_s": elapsed, "status": response.status_code,
                           **summarize_response(body, image)}, log)
                return image
            failure = f"openrouter {response.status_code}: {response.text[:300]}"
            log_event({**base, "attempt": attempt, "elapsed_s": elapsed,
                       "status": response.status_code, "error": failure}, log)
            if response.status_code < 500 and response.status_code != 429:
                raise ValueError(failure)

        if attempt < MAX_ATTEMPTS:
            delay = RETRY_BASE_DELAY * 2 ** (attempt - 1)
            print(f"       retrying {job.name} {job.shot} in {delay}s ({failure})", file=sys.stderr)
            await asyncio.sleep(delay)

    raise ValueError(f"{MAX_ATTEMPTS} attempts failed - {failure}")


async def host(client, path: Path, folder: str, api_key: str, log: Path) -> str:
    """Upload one file to PixelVault and return its CDN URL."""
    size = path.stat().st_size
    if size > PIXELVAULT_MAX_BYTES:
        raise ValueError(
            f"{path.name} is {size / 1_048_576:.1f} MB, over PixelVault's "
            f"{PIXELVAULT_MAX_BYTES // 1_048_576} MB limit"
        )
    response = await client.post(
        PIXELVAULT_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (path.name, path.read_bytes(), "image/jpeg")},
        data={"folder": folder},
    )
    base = {"event": "host", "product": folder, "file": path.name, "file_bytes": size,
            "status": response.status_code}
    if not response.is_success:
        failure = f"pixelvault {response.status_code}: {response.text[:300]}"
        log_event({**base, "error": failure}, log)
        raise ValueError(failure)
    url = (response.json().get("data") or {}).get("url")
    if not url:
        failure = f"pixelvault returned no url: {response.text[:300]}"
        log_event({**base, "error": failure}, log)
        raise ValueError(failure)
    log_event({**base, "url": url}, log)
    return url


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def read_manifest(name: str) -> dict:
    """The product's hosted URLs so far, or an empty one."""
    path = manifest_path(name)
    if not path.is_file():
        return {"product": name, "images": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"product": name, "images": []}


def merge_manifest(manifest: dict, shot: str, file: str, url: str) -> dict:
    """Record one hosted shot, replacing that shot's entry rather than appending.

    Merged instead of rewritten so hosting one shot on its own does not drop the
    other shot's URL.
    """
    order = [known for known, _ in SHOTS]
    images = [entry for entry in manifest.get("images", []) if entry.get("shot") != shot]
    images.append({"shot": shot, "file": file, "url": url})
    images.sort(key=lambda entry: order.index(entry["shot"]) if entry["shot"] in order else len(order))
    return {"product": manifest.get("product"), "images": images}


# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------


def products_from_csv(path: Path) -> list[Product]:
    """One Product per CSV row, aliased to its photo folder's name.

    The alias matters: segment.py run with --folder files cutouts under the
    folder's name, while a CSV row is named after its title. `images/chaise/`
    and the row `Chaise Camping` are the same product, and --name must find it
    either way or the scenery silently goes missing.
    """
    batch = load_products_csv(path)
    products = []
    for row in batch.rows:
        name = product_name(row)
        folder_alias = slugify(row.product_folder.name)
        products.append(
            Product(
                name=name,
                title=row.title or name.replace("-", " "),
                scenery=row.scenery_description,
                aliases=() if folder_alias == name else (folder_alias,),
            )
        )
    return products


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_keys() -> tuple[str | None, str | None]:
    """OpenRouter and PixelVault keys from .env, without ever printing them.

    The project's .env spells the first one `OPEN_ROUTER_API_KEY`; the vendor and
    every tutorial spell it `OPENROUTER_API_KEY`. Accept both, so a tidy-up rename
    does not read as a missing key.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # env vars may still be exported by the shell
        pass
    openrouter = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    return openrouter or None, os.environ.get("PIXELVAULT_API_KEY") or None


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


async def run_generation(jobs: list[Job], api_key: str, concurrency: int, log: Path) -> list[str]:
    """Generate every job concurrently. Returns the products that produced a file."""
    import httpx

    limit = asyncio.Semaphore(concurrency)

    async def one(client, job: Job) -> str | None:
        async with limit:
            references = reference_cutouts(job.name)
            try:
                image = await generate(client, job, references, api_key, log)
            except (OSError, ValueError) as error:
                print(f"failed {job.name} {job.shot}: {error}", file=sys.stderr)
                return None
            job.destination.parent.mkdir(parents=True, exist_ok=True)
            job.destination.write_bytes(image)
            print(f"wrote  {_relative(job.destination)}  {len(image) / 1024:.0f} KB")
            return job.name

    timeout = httpx.Timeout(300.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(*(one(client, job) for job in jobs))

    done: list[str] = []
    for name in results:
        if name and name not in done:
            done.append(name)
    return done


async def run_hosting(names: list[str], api_key: str, concurrency: int, log: Path) -> int:
    """Upload every staged shot on disk for these products. Returns an exit code."""
    import httpx

    limit = asyncio.Semaphore(concurrency)
    uploads = [
        (name, shot, output_path(name, shot))
        for name in names
        for shot, _ in SHOTS
        if output_path(name, shot).is_file()
    ]

    async def one(client, name: str, shot: str, path: Path):
        async with limit:
            try:
                url = await host(client, path, name, api_key, log)
            except (OSError, ValueError) as error:
                print(f"host failed {_relative(path)}: {error}", file=sys.stderr)
                return None
            print(f"hosted {_relative(path)}  {url}")
            return name, shot, path.name, url

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
        results = await asyncio.gather(*(one(client, *upload) for upload in uploads))

    manifests = {name: read_manifest(name) for name in names}
    for result in results:
        if result:
            name, shot, filename, url = result
            manifests[name] = merge_manifest(manifests[name], shot, filename, url)

    for name, manifest in manifests.items():
        if manifest["images"]:
            destination = manifest_path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return 1 if any(result is None for result in results) else 0


def resolve_products(args, parser) -> list[Product] | None:
    """Work out what to stage, and where its scenery comes from."""
    catalogue: list[Product] = []
    csv_error: Exception | None = None
    try:
        catalogue = products_from_csv(args.csv)
    except (OSError, ValueError) as error:
        csv_error = error

    if not args.name:
        if csv_error:
            print(csv_error, file=sys.stderr)
            return None
        return catalogue

    wanted = slugify(args.name)
    match = next((product for product in catalogue if product.answers_to(wanted)), None)

    if match:
        scenery = args.scenery or match.scenery
        title = args.title or match.title
    else:
        if csv_error:
            print(f"note: {args.csv.name} could not be read ({csv_error})", file=sys.stderr)
        scenery = args.scenery
        title = args.title or wanted.replace("-", " ")

    if not scenery:
        where = "has no scenery cell" if match else f"is not in {args.csv.name}"
        print(
            f"note: {wanted} {where} and no --scenery was given - "
            f"staging on a plain white backdrop",
            file=sys.stderr,
        )
    return [Product(name=wanted, title=title, scenery=scenery)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "products.csv")
    parser.add_argument("--name", help="Stage one product; its scenery still comes from the CSV.")
    parser.add_argument("--scenery", help="Override the CSV scenery for --name.")
    parser.add_argument("--title", help="Override the product wording for --name.")
    parser.add_argument("--force", action="store_true", help="Redo shots already generated.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and cost only.")
    parser.add_argument("--no-host", action="store_true", help="Generate without uploading.")
    parser.add_argument("--host-only", action="store_true", help="Upload output/ as it stands.")
    parser.add_argument("--log", type=Path, default=LOG_PATH, help="Where to append the run log.")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Requests in flight at once."
    )
    args = parser.parse_args(argv)

    if args.scenery and not args.name:
        parser.error("--scenery applies to --name; the CSV carries scenery for a batch")
    if args.no_host and args.host_only:
        parser.error("--no-host and --host-only ask for opposite things")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    products = resolve_products(args, parser)
    if products is None:
        return 1

    openrouter_key, pixelvault_key = load_keys()

    if args.host_only:
        if not pixelvault_key:
            print("PIXELVAULT_API_KEY is not set (.env or environment)", file=sys.stderr)
            return 1
        return asyncio.run(
            run_hosting([p.name for p in products], pixelvault_key, args.concurrency, args.log)
        )

    ready: list[Product] = []
    for product in products:
        source = cutout_name(product)
        if source is None:
            looked = ", ".join(
                _relative(cutout_dir(name)) for name in (product.name, *product.aliases)
            )
            print(
                f"skip   {product.name} (no cutouts in {looked} - run scripts/segment.py first)",
                file=sys.stderr,
            )
            continue
        # Stage under whichever name the cutouts are filed as, so output/ and
        # images/ stay the same folder name for a given product.
        ready.append(product._replace(name=source))

    pending = []
    for job in plan_generations(ready):
        if job.destination.exists() and not args.force:
            print(f"skip   {_relative(job.destination)} (exists, --force to redo)")
            continue
        pending.append(job)

    if not pending:
        print("nothing to generate")
        return 0 if ready else 1

    print(f"{len(pending)} image(s) via {MODEL_ID}, ~${len(pending) * COST_PER_IMAGE:.2f}")

    if args.dry_run:
        for job in pending:
            print(f"\n  {_relative(job.destination)}\n    {job.prompt}")
        return 0

    if not openrouter_key:
        print("OPEN_ROUTER_API_KEY is not set (.env or environment)", file=sys.stderr)
        return 1
    if not args.no_host and not pixelvault_key:
        print(
            "PIXELVAULT_API_KEY is not set (.env or environment) - "
            "pass --no-host to generate without uploading",
            file=sys.stderr,
        )
        return 1

    log_event({"event": "run", "images": len(pending), "concurrency": args.concurrency,
               "products": [product.name for product in ready]}, args.log)

    generated = asyncio.run(run_generation(pending, openrouter_key, args.concurrency, args.log))
    failures = len(pending) - sum(
        1 for job in pending if job.name in generated and job.destination.is_file()
    )
    print(f"log    {_relative(args.log)}")

    if args.no_host or not generated:
        return 1 if failures else 0
    hosting = asyncio.run(run_hosting(generated, pixelvault_key, args.concurrency, args.log))
    return hosting or (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
