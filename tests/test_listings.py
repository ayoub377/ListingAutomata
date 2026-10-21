"""Tests for the marketplace listings — the shared base and the Meta feed row."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from product_catalog import Listing, MetaListing, ProductRow


def _meta(**overrides: object) -> MetaListing:
    fields: dict[str, object] = {
        "id": "chaise-camping",
        "title": "Chaise Camping",
        "description": "Chaise pliante.",
        "price": Decimal("1300"),
        "link": "https://example.com/chaise",
        "image_urls": ["https://img.example.com/1.jpg"],
    }
    return MetaListing(**{**fields, **overrides})


def test_the_base_class_cannot_be_instantiated() -> None:
    """A marketplace must say how it writes a feed row."""
    with pytest.raises(TypeError):
        Listing(  # type: ignore[abstract]
            id="x",
            title="x",
            description="x",
            price=1,
            link="https://example.com",
            image_urls=["https://img.example.com/1.jpg"],
        )


def test_a_subclass_without_to_feed_row_is_rejected() -> None:
    class Half(Listing):
        platform = "half"
        feed_columns = ("id",)

    with pytest.raises(TypeError):
        Half(  # type: ignore[abstract]
            id="x",
            title="x",
            description="x",
            price=1,
            link="https://example.com",
            image_urls=["https://img.example.com/1.jpg"],
        )


def test_meta_feed_row_uses_metas_formats() -> None:
    row = _meta(brand="Bestway").to_feed_row()

    assert list(row) == list(MetaListing.feed_columns)
    assert row["price"] == "1300.00 MAD"
    assert row["availability"] == "in stock"
    assert row["condition"] == "new"
    assert row["brand"] == "Bestway"


def test_first_image_is_main_and_the_rest_are_additional() -> None:
    row = _meta(image_urls=["https://i/1.jpg", "https://i/2.jpg", "https://i/3.jpg"]).to_feed_row()

    assert row["image_link"] == "https://i/1.jpg"
    assert row["additional_image_link"] == "https://i/2.jpg,https://i/3.jpg"


def test_a_single_image_leaves_additional_blank() -> None:
    assert _meta().to_feed_row()["additional_image_link"] == ""


def test_optional_fields_are_blank_cells_not_none() -> None:
    row = _meta().to_feed_row()

    assert row["brand"] == ""
    assert row["google_product_category"] == ""


def test_meta_rejects_a_title_over_its_limit() -> None:
    with pytest.raises(ValidationError, match="title is over"):
        _meta(title="x" * 201)


def test_meta_rejects_too_many_images() -> None:
    with pytest.raises(ValidationError, match="images"):
        _meta(image_urls=[f"https://i/{n}.jpg" for n in range(22)])


def test_a_listing_needs_at_least_one_image() -> None:
    with pytest.raises(ValidationError):
        _meta(image_urls=[])


def test_from_row_takes_the_rows_values_and_the_extras(tmp_path: Path) -> None:
    row = ProductRow(
        product_folder=tmp_path,
        price=4300,
        title="Piscine Bestway 4.6m",
        brand="Bestway",
        link="https://example.com/piscine",
    )

    listing = MetaListing.from_row(
        row,
        id="piscine-bestway-4-6m",
        description="Piscine ronde.",
        image_urls=["https://i/1.jpg"],
    )

    assert listing.title == "Piscine Bestway 4.6m"
    assert listing.price == Decimal(4300)
    assert listing.brand == "Bestway"
    assert listing.currency == "MAD"


def test_from_row_extras_win_over_the_row(tmp_path: Path) -> None:
    row = ProductRow(product_folder=tmp_path, price=90, title="Old", link="https://a")

    listing = MetaListing.from_row(
        row, id="x", title="New", description="d", image_urls=["https://i/1.jpg"]
    )

    assert listing.title == "New"


def test_from_row_reports_what_is_still_missing(tmp_path: Path) -> None:
    """A row with no title yet must not become a listing with a blank one."""
    row = ProductRow(product_folder=tmp_path, price=90)

    with pytest.raises(ValidationError) as error:
        MetaListing.from_row(row, id="x", image_urls=["https://i/1.jpg"])

    missing = {issue["loc"][0] for issue in error.value.errors()}
    assert missing == {"title", "description", "link"}
