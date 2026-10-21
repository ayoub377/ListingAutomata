"""Tests for the products.csv ingestion — one row is one product folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from product_catalog import ProductRow, is_primary_photo, load_products_csv


def _write_csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "products.csv"
    path.write_text(body, encoding="utf-8")
    return path


def _photo_folder(tmp_path: Path, name: str, *photos: str) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    for photo in photos:
        (folder / photo).touch()
    return folder


def test_photos_are_sorted_and_filtered(tmp_path: Path) -> None:
    """A product with several shots numbers them the same way on every run."""
    folder = _photo_folder(tmp_path, "chaise", "chaise_kass2.jpeg", "chaise_kass1.jpeg", "notes.txt")
    row = ProductRow(product_folder=folder, price=1300)

    assert row.photos() == [folder / "chaise_kass1.jpeg", folder / "chaise_kass2.jpeg"]


def test_photos_is_empty_for_a_folder_that_is_not_there(tmp_path: Path) -> None:
    row = ProductRow(product_folder=tmp_path / "nope", price=1300)

    assert row.photos() == []
    assert not row.exists()


def test_an_empty_folder_counts_as_missing(tmp_path: Path) -> None:
    """The folder exists but yields no cutout, so the batch must not start."""
    folder = _photo_folder(tmp_path, "empty")
    csv_path = _write_csv(tmp_path, f"product_folder,price\n{folder},1300\n")

    assert load_products_csv(csv_path).missing_photos()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("chaise_primary.jpeg", True),
        ("PRIMARY.jpeg", True),  # marker is typed by hand, in whatever case
        ("IMG_4821-Primary.jpeg", True),
        ("chaise_kass1.jpeg", False),
        ("primaire.jpeg", False),
        ("chaise.primary.jpeg", True),  # marker survives a dotted name
    ],
)
def test_is_primary_photo_reads_the_marker(filename: str, expected: bool) -> None:
    assert is_primary_photo(Path("/photos/chaise") / filename) is expected


def test_primary_photo_finds_the_marked_shot(tmp_path: Path) -> None:
    folder = _photo_folder(tmp_path, "chaise", "chaise_kass1.jpeg", "chaise_primary.jpeg")
    row = ProductRow(product_folder=folder, price=1300)

    assert row.primary_photo() == folder / "chaise_primary.jpeg"


def test_primary_photo_is_none_when_nothing_is_marked(tmp_path: Path) -> None:
    """The marker is optional — most folders are just numbered shots."""
    folder = _photo_folder(tmp_path, "chaise", "chaise_kass1.jpeg", "chaise_kass2.jpeg")
    row = ProductRow(product_folder=folder, price=1300)

    assert row.primary_photo() is None


def test_primary_photo_ignores_a_marked_non_photo(tmp_path: Path) -> None:
    """It picks from photos(), so a primary.txt next to the shots is not a shot."""
    folder = _photo_folder(tmp_path, "chaise", "primary.txt", "chaise_kass1.jpeg")
    row = ProductRow(product_folder=folder, price=1300)

    assert row.primary_photo() is None


def test_slug_source_is_the_folder_name(tmp_path: Path) -> None:
    row = ProductRow(product_folder=_photo_folder(tmp_path, "4.6Bestway", "a.jpeg"), price=4300)

    assert row.slug_source == "4.6Bestway"


def test_load_csv_reads_folders(tmp_path: Path) -> None:
    first = _photo_folder(tmp_path, "4.6Bestway", "4.6xbestway.jpeg")
    second = _photo_folder(tmp_path, "chaise", "chaise_kass1.jpeg", "chaise_kass2.jpeg")
    csv_path = _write_csv(
        tmp_path,
        f"product_folder,title,price\n{first},Piscine Bestway 4.6m,4300\n{second},,1300\n",
    )

    batch = load_products_csv(csv_path)

    assert batch.missing_photos() == []
    assert [len(row.photos()) for row in batch.rows] == [1, 2]


def test_the_same_folder_twice_is_rejected(tmp_path: Path) -> None:
    folder = _photo_folder(tmp_path, "chaise", "chaise_kass1.jpeg")
    csv_path = _write_csv(tmp_path, f"product_folder,price\n{folder},1300\n{folder},1400\n")

    with pytest.raises(ValueError, match="listed twice"):
        load_products_csv(csv_path)


def test_an_unquoted_comma_is_rejected_instead_of_truncating(tmp_path: Path) -> None:
    """DictReader would drop the tail, staging the product in half a scene."""
    folder = _photo_folder(tmp_path, "chaise", "chaise_kass1.jpeg")
    csv_path = _write_csv(
        tmp_path,
        f"product_folder,price,scenery\n{folder},90,forest near a lake, and green grass\n",
    )

    with pytest.raises(ValueError, match="too many columns"):
        load_products_csv(csv_path)


def test_a_quoted_comma_survives(tmp_path: Path) -> None:
    folder = _photo_folder(tmp_path, "chaise", "chaise_kass1.jpeg")
    csv_path = _write_csv(
        tmp_path,
        f'product_folder,price,scenery\n{folder},90,"forest near a lake, and green grass"\n',
    )

    row = load_products_csv(csv_path).rows[0]

    assert row.scenery_description == "forest near a lake, and green grass"


def test_a_blank_folder_cell_names_the_column(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, "product_folder,price\n   ,1300\n")

    with pytest.raises(ValueError, match="product_folder is empty"):
        load_products_csv(csv_path)
