"""Finding the generated products that are waiting for review.

A product shows up once its output folder holds a `description.json` and at least
one image, which is when workflow.py has finished with it. The CSV row supplies
what generation does not: the price, availability and any product link.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from product_catalog.models import PHOTO_SUFFIXES, ProductRow, load_products_csv
from product_catalog.naming import product_name, slugify
from product_catalog.review.store import (
    DESCRIPTION_FILE,
    Review,
    Status,
    fingerprint,
    load_review,
)

PREVIOUS_DIR = "prev"

LANGS = ("fr", "ar")


@dataclass(frozen=True)
class ReviewSettings:
    """Where the review app reads from."""

    output_dir: Path
    csv_path: Path

    @classmethod
    def from_project(cls, root: Path) -> ReviewSettings:
        return cls(output_dir=root / "output", csv_path=root / "products.csv")


@dataclass(frozen=True)
class ReviewProduct:
    """One generated product: its CSV row, its generated content and its review."""

    name: str  # the Meta listing id
    folder_name: str  # where the output lives, which may be the photo folder's name
    folder: Path
    row: ProductRow
    generated: dict
    images: tuple[str, ...]
    review: Review
    fingerprint: str

    @property
    def status(self) -> Status:
        return self.review.status

    @property
    def lang(self) -> str:
        return self.review.lang

    @property
    def langs(self) -> list[str]:
        """Languages the generator produced a description for."""
        return [lang for lang in LANGS if self.generated.get(lang)]

    @property
    def color(self) -> str | None:
        return self.row.color

    @property
    def base_title(self) -> str:
        """The title before any edit. Same precedence as scripts/publish.py."""
        return self.row.title or self.generated.get("title") or self.name.replace("-", " ")

    @property
    def base_price(self) -> Decimal:
        return self.row.price

    def generated_description(self, lang: str) -> str:
        return self.generated.get(lang) or ""

    @property
    def title(self) -> str:
        return self.review.title or self.base_title

    @property
    def price(self) -> Decimal:
        return self.review.price or self.base_price

    @property
    def description(self) -> str:
        return self.review.description or self.generated_description(self.lang)

    @property
    def scenery(self) -> str:
        """The scene the image prompt uses: the reviewer's override, else the CSV's."""
        return self.review.scenery or self.row.scenery_description or ""

    @staticmethod
    def shot_of(image: str) -> str:
        """`product_angle.jpg` -> `angle`."""
        return Path(image).stem.removeprefix("product_")

    def has_previous(self, image: str) -> bool:
        """True when a regeneration kept the version it replaced."""
        return (self.folder / PREVIOUS_DIR / image).is_file()


@dataclass(frozen=True)
class Discovery:
    products: list[ReviewProduct]
    problems: list[str] = field(default_factory=list)


def _image_names(folder: Path, generated: dict) -> tuple[str, ...]:
    """The staged shots on disk, in the order description.json lists them."""
    listed = generated.get("images")
    names = [
        name
        for name in (listed if isinstance(listed, list) else [])
        if isinstance(name, str)
        and Path(name).name == name  # a bare filename, never a path
        and Path(name).suffix.lower() in PHOTO_SUFFIXES
        and (folder / name).is_file()
    ]
    if not names:
        names = sorted(
            child.name
            for child in folder.glob("product_*")
            if child.suffix.lower() in PHOTO_SUFFIXES
        )
    return tuple(names)


def discover(settings: ReviewSettings) -> Discovery:
    """Every CSV row that has generated output, plus anything worth telling the reviewer."""
    try:
        rows = load_products_csv(settings.csv_path).listings()
    except (OSError, ValueError) as error:
        return Discovery([], [str(error)])

    products: list[ReviewProduct] = []
    problems: list[str] = []
    claimed: set[str] = set()
    for row in rows:
        name = product_name(row)
        alias = row.folder_alias  # None for a colour variant: its folder is its own
        folder_name = next(
            (
                c
                for c in (name, alias)
                if c and (settings.output_dir / c / DESCRIPTION_FILE).is_file()
            ),
            None,
        )
        if folder_name is None:
            continue  # not generated yet
        claimed.add(folder_name)
        folder = settings.output_dir / folder_name
        try:
            generated = json.loads((folder / DESCRIPTION_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append(f"{folder_name}: {DESCRIPTION_FILE} is unreadable ({error})")
            continue
        if not isinstance(generated, dict):
            problems.append(f"{folder_name}: {DESCRIPTION_FILE} is not a JSON object")
            continue
        images = _image_names(folder, generated)
        if not images:
            problems.append(f"{folder_name}: no staged images yet")
            continue
        current = fingerprint(folder)
        products.append(
            ReviewProduct(
                name=name,
                folder_name=folder_name,
                folder=folder,
                row=row,
                generated=generated,
                images=images,
                review=load_review(folder, current),
                fingerprint=current,
            )
        )

    if settings.output_dir.is_dir():
        for child in sorted(settings.output_dir.iterdir()):
            if child.is_dir() and child.name not in claimed and (child / DESCRIPTION_FILE).is_file():
                problems.append(
                    f"output/{child.name} has generated content but no matching row in "
                    f"{settings.csv_path.name}, so it has no price and is not listed"
                )
    return Discovery(products, problems)
