"""Tests for the products.csv ingestion — one row is one product folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from product_catalog import ProductRow, load_products_csv


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


def test_a_blank_folder_cell_names_the_column(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, "product_folder,price\n   ,1300\n")

    with pytest.raises(ValueError, match="product_folder is empty"):
        load_products_csv(csv_path)
