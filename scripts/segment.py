"""Cut a product out of its photos with BiRefNet.

    python scripts/segment.py                      # every row in products.csv
    python scripts/segment.py --csv batch.csv
    python scripts/segment.py --folder "C:/photos/my-pool" --name my-pool
    python scripts/segment.py --image "C:/photos/pool.jpeg" --name my-pool

Each CSV row names a product_folder, and every photo in it is segmented — so one
product with three shots yields three cutouts under one name.

Writes RGBA cutouts to images/<prod_name>/product_<n>_seg.png, numbered in the
folder's name order. A photo whose filename carries `primary` — the shot the
staged scenes get generated from — is filed as product_primary_seg.png instead
and takes no number; it is optional, and at most one per product. Photos whose
output already exists are skipped; pass --force to redo them.

The model is the full BiRefNet (~900 MB, PyTorch) rather than the BiRefNet_lite
ONNX export the rest of the pipeline uses, so the first run downloads weights
into ~/.cache/huggingface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

from product_catalog import (
    PHOTO_SUFFIXES,
    PRIMARY_MARKER,
    is_primary_photo,
    load_products_csv,
    product_name,
    slugify,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
MODEL_ID = "zhengpeng7/BiRefNet"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MATTE_FLOOR = 0.20
MATTE_CEILING = 0.95


def folder_photos(folder: Path) -> list[Path]:
    """Photos inside a folder, in name order — the --folder twin of row.photos()."""
    return sorted(
        child
        for child in folder.iterdir()
        if child.is_file() and child.suffix.lower() in PHOTO_SUFFIXES
    )


def load_model(device: str):
    """Load BiRefNet once and reuse it for the whole batch.

    `trust_remote_code=True` runs the modeling code published in the model repo;
    it is how BiRefNet is distributed, and it is what pins timm as a dependency.

    The published weights are fp16 and transformers now keeps a checkpoint's own
    dtype, so ask for fp32 on CPU, where half-precision convolutions are slow
    where they are supported at all.
    """
    import torch
    from transformers import AutoModelForImageSegmentation

    torch.set_float32_matmul_precision("high")
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model


def load_photo(path: Path) -> Image.Image:
    """Open a photo the way its camera meant it to be seen.

    Phones store the sensor's raw landscape frame and record the quarter turn in
    EXIF Orientation instead of rotating the pixels. A bare `Image.open` ignores
    that tag, so the product gets segmented — and listed — lying on its side.
    """
    with Image.open(path) as handle:
        return ImageOps.exif_transpose(handle).convert("RGB")


def firm_up(matte: Image.Image) -> Image.Image:
    """Pull the matte's two tails to fully transparent and fully opaque.

    BiRefNet leaves floor reflections at a faint 15-20% and most of the product
    body at 251-254 rather than 255: the first prints as grey ghosts around the
    cutout, the second as a veil over the whole product. Everything between the
    two thresholds keeps its ramp, so genuine soft edges — tubing, mesh, straps —
    stay anti-aliased.
    """
    floor, ceiling = round(MATTE_FLOOR * 255), round(MATTE_CEILING * 255)
    span = ceiling - floor
    return matte.point(
        lambda value: 0 if value <= floor else 255 if value >= ceiling
        else round((value - floor) * 255 / span)
    )


def letterbox(image: Image.Image, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Fit the photo into a square without squashing it.

    Resizing a 3:4 frame straight to 1:1 shears the product and walks the matte
    off its edges, so pad instead and remember the box to undo it.
    """
    width, height = image.size
    scale = size / max(width, height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset_x, offset_y = (size - new_width) // 2, (size - new_height) // 2
    canvas.paste(image.resize((new_width, new_height), Image.LANCZOS), (offset_x, offset_y))
    return canvas, (offset_x, offset_y, new_width, new_height)


def segment(image: Image.Image, model, device: str, resolution: int) -> Image.Image:
    """Return the photo with BiRefNet's matte as its alpha channel."""
    import torch
    from torchvision import transforms

    to_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )

    padded, (offset_x, offset_y, box_width, box_height) = letterbox(image, resolution)
    dtype = next(model.parameters()).dtype
    batch = to_tensor(padded).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        prediction = model(batch)[-1].sigmoid().float().cpu()[0].squeeze()

    matte = firm_up(transforms.ToPILImage()(prediction))
    matte = matte.crop(
        (offset_x, offset_y, offset_x + box_width, offset_y + box_height)
    ).resize(image.size, Image.LANCZOS)

    cutout = image.convert("RGBA")
    cutout.putalpha(matte)
    return cutout


def output_path(name: str, index: int | str) -> Path:
    """Where one cutout lands — `index` is its 1-based number, or `primary`."""
    return IMAGES_DIR / name / f"product_{index}_seg.png"


def plan_destinations(jobs: list[tuple[str, Path]]) -> list[tuple[Path, Path]]:
    """Pair every photo with the cutout it writes, numbering within each product.

    The primary shot is filed under its marker rather than a number, so the
    ordinary photos stay 1..n whether or not the product has one — adding a
    primary later does not renumber the cutouts already on disk.

    Raises ValueError if a product marks two photos primary: they would write the
    same file, and guessing which one the scenes should be built on is worse than
    asking.
    """
    counts: dict[str, int] = {}
    primaries: dict[str, Path] = {}
    plan: list[tuple[Path, Path]] = []
    for name, photo in jobs:
        if is_primary_photo(photo):
            if name in primaries:
                raise ValueError(
                    f"{name}: two photos marked {PRIMARY_MARKER} "
                    f"({primaries[name].name} and {photo.name}) - keep one"
                )
            primaries[name] = photo
            index: int | str = PRIMARY_MARKER
        else:
            counts[name] = counts.get(name, 0) + 1
            index = counts[name]
        plan.append((photo, output_path(name, index)))
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, default=PROJECT_ROOT / "products.csv")
    source.add_argument("--folder", type=Path, help="Segment one product folder instead of a batch.")
    source.add_argument("--image", type=Path, help="Segment one photo instead of a batch.")
    parser.add_argument(
        "--name",
        help="Output folder name for --folder/--image. Defaults to the source name.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resolution", type=int, default=1024, help="Model input size.")
    parser.add_argument("--force", action="store_true", help="Redo photos already segmented.")
    args = parser.parse_args(argv)

    if args.image:
        if not args.image.is_file():
            parser.error(f"no such photo: {args.image}")
        jobs = [(args.name or slugify(args.image.stem), args.image)]
    elif args.folder:
        if not args.folder.is_dir():
            parser.error(f"no such folder: {args.folder}")
        photos = folder_photos(args.folder)
        if not photos:
            parser.error(f"no photos in {args.folder}")
        name = args.name or slugify(args.folder.name)
        jobs = [(name, photo) for photo in photos]
    else:
        try:
            batch = load_products_csv(args.csv)
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
        if missing := batch.missing_photos():
            for row in missing:
                reason = "no photos in" if row.product_folder.is_dir() else "missing folder:"
                print(f"{reason} {row.product_folder}", file=sys.stderr)
            return 1
        jobs = [(product_name(row), photo) for row in batch.rows for photo in row.photos()]

    try:
        plan = plan_destinations(jobs)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    pending = []
    for photo, destination in plan:
        if destination.exists() and not args.force:
            print(f"skip   {destination.relative_to(PROJECT_ROOT)} (exists, --force to redo)")
            continue
        pending.append((photo, destination))

    if not pending:
        print("nothing to do")
        return 0

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device} ({len(pending)} photo(s))")
    model = load_model(device)

    for photo, destination in pending:
        image = load_photo(photo)
        cutout = segment(image, model, device, args.resolution)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cutout.save(destination)
        print(f"wrote  {destination.relative_to(PROJECT_ROOT)}  {cutout.width}x{cutout.height}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
