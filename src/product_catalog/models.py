"""Pydantic models for a batch listed in products.csv.

One CSV row is one product, and `product_folder` points at the folder holding
its photos — one or several shots of the same item. Everything the CSV omits
falls back to the batch defaults documented in the README (MAD, in stock,
condition new).
"""

from __future__ import annotations

import csv
import re
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from product_catalog.naming import product_name, slugify


PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
PRIMARY_MARKER = "primary"


def is_primary_photo(path: Path) -> bool:
    """True when a photo is opted in as the product's primary shot.

    The primary is the one shot the staged scenes are generated from, and it is
    chosen the only way a phone gallery allows: by renaming the file, e.g.
    `IMG_4821_primary.jpeg`. It is optional — a folder without one is just a set
    of numbered photos.
    """
    return PRIMARY_MARKER in path.stem.lower()


class Availability(StrEnum):
    """Meta feed `availability` values."""

    IN_STOCK = "in stock"
    OUT_OF_STOCK = "out of stock"
    PREORDER = "preorder"
    AVAILABLE_FOR_ORDER = "available for order"
    DISCONTINUED = "discontinued"


class Condition(StrEnum):
    """Meta feed `condition` values. Describes the product, not its carton."""

    NEW = "new"
    REFURBISHED = "refurbished"
    USED = "used"


def _is_blank(value: object) -> bool:
    """A blank cell is an omitted value, so the batch default applies."""
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_enum(value: object) -> object:
    """`Out_of_Stock` and `OUT OF STOCK` are the same answer typed differently."""
    return value.strip().lower().replace("_", " ") if isinstance(value, str) else value


def _blank_to_none(value: object) -> object:
    return None if _is_blank(value) else value


def _strip(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


OptionalCell = Annotated[str | None, BeforeValidator(_blank_to_none), BeforeValidator(_strip)]

MAX_COLOR = 100  # Meta's limit for a colour in the Batch API


def _split_colors(value: object) -> object:
    """`vert; bleu | rouge` -> ("vert", "bleu", "rouge"). Blank means no variants.

    Semicolon and pipe are the separators to use, because a comma has to be
    quoted in a CSV cell; a quoted comma list works too. Two spellings of the same
    colour (`Vert`, `vert `, `VERT`) are one colour, since they would get one id.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[;|,]", value)]
    elif isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value]
    else:
        return value
    seen: dict[str, str] = {}
    for part in parts:
        if part:
            seen.setdefault(slugify(part) or part.casefold(), part)
    return tuple(seen.values())


class ProductRow(BaseModel):
    """A single row of products.csv.

    Only `product_folder` and `price` are required — `title` is normally left for
    the vision step to fill in, and the rest carry the batch defaults.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        use_enum_values=False,
        frozen=True,
    )

    product_folder: Path = Field(
        description="Folder of this product's photos, wherever it already lives."
    )
    price: Decimal = Field(gt=0, description="Unit price, bare number in `currency`.")
    title: OptionalCell = Field(default=None, max_length=200)
    description: OptionalCell = Field(default=None, max_length=9999)
    brand: OptionalCell = Field(default=None, max_length=100)
    scenery_description: OptionalCell = Field(
        default=None,
        max_length=500,
        validation_alias=AliasChoices("scenery_description", "scenery"),
        description=(
            "Where the product should be staged, e.g. `sunny backyard lawn`. "
            "Filled into the image-generation prompt; None means no scene was "
            "asked for and the product stays on plain white."
        ),
    )
    currency: str = Field(default="MAD", pattern=r"^[A-Z]{3}$")
    availability: Availability = Availability.IN_STOCK
    condition: Condition = Condition.NEW
    link: OptionalCell = Field(
        default=None,
        description="Product page URL. None means the run fills in a flagged placeholder.",
    )
    colors: Annotated[tuple[str, ...], BeforeValidator(_split_colors)] = Field(
        default=(),
        description=(
            "Colour variants, separated by `;` or `|`, e.g. `vert; bleu; rouge`. Each "
            "colour becomes its own listing, with its own generated photos, grouped "
            "under one product in the Meta catalogue. Blank means a single listing."
        ),
    )
    photo_color: OptionalCell = Field(
        default=None,
        max_length=MAX_COLOR,
        description=(
            "The colour the product has in its photos. That variant is staged as the "
            "photo shows it; every other colour is recoloured from it. Blank means "
            "every variant is recoloured."
        ),
    )
    color: OptionalCell = Field(
        default=None,
        max_length=MAX_COLOR,
        description=(
            "The colour of a product that comes in just one. Sent to Meta as `color`. "
            "For several colours use `colors` instead."
        ),
    )
    # Set when `colors` is expanded into variants, never read from the CSV: the alias is
    # something no header will be called, and `model_copy` sets it by field name.
    group: str | None = Field(
        default=None,
        validation_alias="_variant_group",
        description="The product a colour variant belongs to; Meta's `item_group_id`.",
    )

    @field_validator("product_folder", mode="before")
    @classmethod
    def _require_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("product_folder is empty - every row needs a photo folder")
        return value

    @field_validator("price", mode="before")
    @classmethod
    def _clean_price(cls, value: object) -> object:
        """Accept `4 300`, `4,300` and `4300 MAD` — all of them get typed by hand."""
        if isinstance(value, str):
            return value.replace(",", "").replace(" ", "").replace(" ", "").upper().removesuffix("MAD")
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def _default_currency(cls, value: object) -> object:
        return "MAD" if _is_blank(value) else value

    @field_validator("availability", mode="before")
    @classmethod
    def _default_availability(cls, value: object) -> object:
        return Availability.IN_STOCK if _is_blank(value) else _normalize_enum(value)

    @field_validator("condition", mode="before")
    @classmethod
    def _default_condition(cls, value: object) -> object:
        return Condition.NEW if _is_blank(value) else _normalize_enum(value)

    @field_validator("colors")
    @classmethod
    def _colors_make_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """A colour goes into the listing id, so it needs at least one latin letter or digit."""
        for color in value:
            if len(color) > MAX_COLOR:
                raise ValueError(f"colour `{color[:20]}...` is over {MAX_COLOR} characters")
            if not slugify(color):
                raise ValueError(
                    f"colour `{color}` has no letters or digits that can go in a product id - "
                    "give it a latin name, e.g. `vert` instead of `أخضر`"
                )
        return value

    @model_validator(mode="after")
    def _colour_columns_agree(self) -> Self:
        if self.colors and self.color:
            raise ValueError("use `color` for a single colour or `colors` for several, not both")
        if self.photo_color and not self.colors:
            raise ValueError("`photo_color` only means something next to `colors`")
        return self

    def variants(self) -> list[ProductRow]:
        """This row as the listings it makes: one per colour, or just itself.

        Every variant keeps the row's price, photos, scene and text settings, gets its
        own colour, and shares one `group` (the product's name before the colour is
        added), which is what ties them together in the catalogue.
        """
        if not self.colors:
            return [self]
        base = product_name(self)
        return [
            self.model_copy(update={"color": color, "colors": (), "group": base})
            for color in self.colors
        ]

    @property
    def staged_from_color(self) -> str | None:
        """The colour the product has in the photos it is staged from.

        A variant says so in `photo_color`. A single-colour product's photos show its
        own colour, so it is never recoloured.
        """
        return self.photo_color or (None if self.group else self.color)

    @property
    def folder_alias(self) -> str | None:
        """The photo folder's name, which an output folder may be filed under.

        Only a single listing can go by it. A variant never does, or every colour of a
        product would claim the same folder.
        """
        return None if self.group else slugify(self.product_folder.name)

    @property
    def slug_source(self) -> str:
        """Fallback slug when products.json has no `source_image` match."""
        return self.product_folder.name

    def photos(self) -> list[Path]:
        """The product's photos, in name order.

        Sorted so a rerun numbers the cutouts the same way it did last time, and
        filtered by suffix so a stray thumbs.db or .txt never reaches the model.
        """
        if not self.product_folder.is_dir():
            return []
        return sorted(
            child
            for child in self.product_folder.iterdir()
            if child.is_file() and child.suffix.lower() in PHOTO_SUFFIXES
        )

    def primary_photo(self) -> Path | None:
        """The photo marked primary, or None when the product has no marked shot.

        First in name order wins if several carry the marker; segment.py refuses
        that folder outright rather than pick, since the cutouts would collide.
        """
        return next((photo for photo in self.photos() if is_primary_photo(photo)), None)

    def exists(self) -> bool:
        """True when the folder is on disk and holds at least one photo."""
        return bool(self.photos())


class ProductBatch(BaseModel):
    """Every row of one products.csv, in file order."""

    model_config = ConfigDict(frozen=True)

    source: Path
    rows: list[ProductRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_duplicate_folders(self) -> Self:
        seen: dict[Path, int] = {}
        for index, row in enumerate(self.rows, start=2):  # row 1 is the header
            key = Path(str(row.product_folder).casefold())
            if key in seen:
                raise ValueError(
                    f"{row.product_folder} is listed twice (rows {seen[key]} and {index})"
                )
            seen[key] = index
        return self

    @model_validator(mode="after")
    def _no_duplicate_names(self) -> Self:
        """Two rows must not share a product name.

        The name is the Meta `retailer_id` (and the output/ folder), derived from
        the title, so two rows titled alike - or differing only in case, accents or
        punctuation - collide and Meta rejects the batch as a duplicate retailer_id.
        Ids must stay stable between runs, so this refuses instead of renaming one.
        """
        seen: dict[str, int] = {}
        for index, row in enumerate(self.rows, start=2):  # row 1 is the header
            for name in (product_name(variant) for variant in row.variants()):
                if name in seen:
                    raise ValueError(
                        f"rows {seen[name]} and {index} both become the product id "
                        f"`{name}` - give each product a distinct title"
                    )
                seen[name] = index
        return self

    def listings(self) -> list[ProductRow]:
        """Every listing the batch makes, in file order: colour variants expanded."""
        return [variant for row in self.rows for variant in row.variants()]

    def missing_photos(self) -> list[ProductRow]:
        """Rows with nothing to segment — folder absent, or holding no photos.

        Either way the row cannot produce a cutout, and that aborts the batch.
        """
        return [row for row in self.rows if not row.exists()]


def load_products_csv(path: Path | str = "products.csv") -> ProductBatch:
    """Read and validate a batch, reporting every bad row at once.

    Raises ValueError listing each failure with its CSV line number, so a typo in
    row 9 does not hide one in row 12.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))

    rows: list[ProductRow] = []
    problems: list[str] = []
    for line, record in enumerate(records, start=2):
        if record.get(None):
            # DictReader files columns past the header under a None key, and the
            # cleanup below drops them. An unquoted comma in a scenery cell would
            # otherwise truncate the scene silently, halfway through a sentence.
            extra = ",".join(str(value) for value in record[None])
            problems.append(
                f"{path.name}:{line}: too many columns - wrap any cell containing "
                f'a comma in double quotes (stray text: "{extra.strip()}")'
            )
            continue
        cleaned = {key.strip(): value for key, value in record.items() if key and key.strip()}
        try:
            rows.append(ProductRow.model_validate(cleaned))
        except ValidationError as error:
            for issue in error.errors():
                field = ".".join(str(part) for part in issue["loc"]) or "row"
                problems.append(f"{path.name}:{line}: {field}: {issue['msg']}")

    if problems:
        raise ValueError("\n".join(problems))

    return ProductBatch(source=path, rows=rows)

