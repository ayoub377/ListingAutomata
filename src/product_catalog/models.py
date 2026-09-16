"""Pydantic models for a batch listed in products.csv.

One CSV row is one product, and `product_folder` points at the folder holding
its photos — one or several shots of the same item. Everything the CSV omits
falls back to the batch defaults documented in the README (MAD, in stock,
condition new).
"""

from __future__ import annotations

import csv
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


PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


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
