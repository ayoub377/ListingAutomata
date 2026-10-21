"""One product, as a marketplace wants to see it.

`Listing` holds the fields every marketplace asks for and the contract each one
must fill in: its name, its feed columns, and how a listing becomes a feed row.
A new marketplace (eBay, Avito...) is a subclass that adds its own fields and
limits and implements `to_feed_row`; nothing upstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from product_catalog.models import Availability, Condition, ProductRow


class Listing(BaseModel, ABC):
    """Fields shared by every marketplace. Platform rules live in subclasses."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    platform: ClassVar[str]
    feed_columns: ClassVar[tuple[str, ...]]

    id: str = Field(min_length=1, description="Our own stable SKU, unique within a feed.")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    price: Decimal = Field(gt=0, description="Unit price, bare number in `currency`.")
    currency: str = Field(default="MAD", pattern=r"^[A-Z]{3}$")
    availability: Availability = Availability.IN_STOCK
    condition: Condition = Condition.NEW
    brand: str | None = None
    link: str = Field(min_length=1, description="Product page URL.")
    image_urls: list[str] = Field(
        min_length=1, description="Hosted photo URLs, main shot first."
    )

    @classmethod
    def from_row(cls, row: ProductRow, **fields: object) -> Self:
        """Build from a CSV row, with `fields` supplying what the row cannot.

        A row has no id, hosted image URLs, or (until the vision step runs) title
        and description. Anything passed in `fields` wins over the row's value;
        anything still missing is reported by validation.
        """
        from_row = row.model_dump(
            include={
                "title",
                "description",
                "brand",
                "price",
                "currency",
                "availability",
                "condition",
                "link",
            },
            exclude_none=True,
        )
        return cls.model_validate({**from_row, **fields})

    @abstractmethod
    def to_feed_row(self) -> dict[str, str]:
        """This listing as one feed row, keyed by `feed_columns`."""


class MetaListing(Listing):
    """A product in a Meta (Facebook / Instagram) catalog feed.

    Column names and limits follow Meta's catalog feed spec: `price` is written
    as `4300.00 MAD`, `image_link` is the main photo and the rest go in
    `additional_image_link`, comma-separated.
    """

    platform: ClassVar[str] = "meta"
    feed_columns: ClassVar[tuple[str, ...]] = (
        "id",
        "title",
        "description",
        "availability",
        "condition",
        "price",
        "link",
        "image_link",
        "additional_image_link",
        "brand",
        "google_product_category",
    )
    max_title: ClassVar[int] = 200
    max_description: ClassVar[int] = 9999
    max_additional_images: ClassVar[int] = 20

    google_product_category: str | None = Field(
        default=None, description="Google taxonomy id or path, e.g. `Home & Garden`."
    )

    @model_validator(mode="after")
    def _within_meta_limits(self) -> Self:
        if len(self.title) > self.max_title:
            raise ValueError(f"title is over Meta's {self.max_title} characters")
        if len(self.description) > self.max_description:
            raise ValueError(f"description is over Meta's {self.max_description} characters")
        if len(self.image_urls) - 1 > self.max_additional_images:
            raise ValueError(
                f"{len(self.image_urls)} images - Meta takes one main shot "
                f"plus {self.max_additional_images} more"
            )
        return self

    def to_feed_row(self) -> dict[str, str]:
        main, *others = self.image_urls
        row = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "availability": self.availability.value,
            "condition": self.condition.value,
            "price": f"{self.price:.2f} {self.currency}",
            "link": self.link,
            "image_link": main,
            "additional_image_link": ",".join(others),
            "brand": self.brand or "",
            "google_product_category": self.google_product_category or "",
        }
        return {column: row[column] for column in self.feed_columns}
