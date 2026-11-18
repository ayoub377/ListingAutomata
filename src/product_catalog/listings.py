"""One product, as a marketplace wants to see it.

`Listing` holds the fields every marketplace asks for and the contract each one
must fill in: its name, its feed columns, and how a listing becomes a feed row.
A new marketplace (eBay, Avito...) is a subclass that adds its own fields and
limits and implements `to_feed_row`; nothing upstream changes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
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
    color: str | None = Field(default=None, max_length=100)
    item_group_id: str | None = Field(
        default=None,
        max_length=100,
        description="Ties the colour variants of one product together; same value on each.",
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
                "color",
            },
            exclude_none=True,
        )
        if row.group:
            from_row["item_group_id"] = row.group
        return cls.model_validate({**from_row, **fields})

    @abstractmethod
    def to_feed_row(self) -> dict[str, str]:
        """This listing as one feed row, keyed by `feed_columns`."""


class MetaListing(Listing):
    """A product in a Meta (Facebook / Instagram) catalog feed.

    Column names and limits follow Meta's catalog feed spec: `price` is written
    as `4300.00 MAD`, `image_link` is the main photo and the rest go in
    `additional_image_link`, comma-separated. `to_batch_request` writes the same
    data as an `items_batch` request (see meta-catalog-formats.md).
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
        "item_group_id",
        "color",
    )
    max_title: ClassVar[int] = 200
    max_description: ClassVar[int] = 9999
    max_additional_images: ClassVar[int] = 20
    max_id: ClassVar[int] = 100

    google_product_category: str | None = Field(
        default=None, description="Google taxonomy id or path, e.g. `Home & Garden`."
    )

    @model_validator(mode="after")
    def _a_variant_has_its_variant_fields(self) -> Self:
        """Meta: every item in a group must fill in the variant fields the group uses.

        A grouped item with no colour is the "parent row" Meta's own guide calls the
        wrong way to set variants up.
        """
        if self.item_group_id and not self.color:
            raise ValueError("a listing in a variant group needs its colour")
        return self

    @model_validator(mode="after")
    def _within_meta_limits(self) -> Self:
        if len(self.id) > self.max_id or not re.fullmatch(r"[a-z0-9-]+", self.id):
            raise ValueError(
                f"id must be a slug of [a-z0-9-], at most {self.max_id} characters"
            )
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
            "item_group_id": self.item_group_id or "",
            "color": self.color or "",
        }
        return {column: row[column] for column in self.feed_columns}

    def to_batch_request(self) -> dict[str, object]:
        """This listing as one Batch API request (`items_batch`).

        `items_batch` takes `{"method", "data"}` with the same field names as the
        feed: the item's id is `data.id` (there is no top-level `retailer_id`;
        Meta only echoes that name back in its response), `price` is a string
        like `90.00 MAD`, and the photos are one `image` array, main shot first.
        `UPDATE` plus the payload's `allow_upsert` creates the item when it is
        missing and updates it otherwise, so a rerun never errors on an existing id.
        Blank optional fields are left out rather than sent empty.
        """
        data: dict[str, object] = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "availability": self.availability.value,
            "condition": self.condition.value,
            "price": f"{self.price:.2f} {self.currency}",
            "link": self.link,
            "image": [{"url": url} for url in self.image_urls],
        }
        if self.brand:
            data["brand"] = self.brand
        if self.google_product_category:
            data["google_product_category"] = self.google_product_category
        if self.item_group_id:
            data["item_group_id"] = self.item_group_id
        if self.color:
            data["color"] = self.color
        return {"method": "UPDATE", "data": data}

    @classmethod
    def batch_payload(cls, listings: Sequence[MetaListing]) -> dict[str, object]:
        """The body for `POST /{catalog_id}/items_batch`; upsert so re-runs update."""
        return {
            "allow_upsert": True,
            "requests": [listing.to_batch_request() for listing in listings],
        }
