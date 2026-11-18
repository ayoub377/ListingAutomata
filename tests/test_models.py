"""Tests for the products.csv ingestion — one row is one product folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from product_catalog import ProductRow, is_primary_photo, load_products_csv, product_name


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


def test_two_products_with_the_same_id_are_rejected(tmp_path: Path) -> None:
    """Titles that slug alike would push the same retailer_id twice."""
    first = _photo_folder(tmp_path, "chaise-bleue", "a.jpeg")
    second = _photo_folder(tmp_path, "chaise-rouge", "b.jpeg")
    csv_path = _write_csv(
        tmp_path,
        f"product_folder,title,price\n{first},Chaise Camping,90\n{second},chaise  camping!,95\n",
    )

    with pytest.raises(ValueError, match=r"rows 2 and 3 both become the product id `chaise-camping`"):
        load_products_csv(csv_path)


def test_an_untitled_row_collides_with_a_titled_one_by_folder_name(tmp_path: Path) -> None:
    """With no title the id comes from the photo folder, so that can collide too."""
    titled = _photo_folder(tmp_path, "a", "a.jpeg")
    untitled = _photo_folder(tmp_path, "Chaise Camping", "b.jpeg")
    csv_path = _write_csv(
        tmp_path,
        f"product_folder,title,price\n{titled},Chaise Camping,90\n{untitled},,95\n",
    )

    with pytest.raises(ValueError, match="both become the product id `chaise-camping`"):
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


# --------------------------------------------------------------------------
# Colour variants
# --------------------------------------------------------------------------


def _chair(tmp_path: Path, **fields: object) -> ProductRow:
    values = {"product_folder": tmp_path / "chaise", "title": "Chaise Camping", "price": 90}
    return ProductRow(**{**values, **fields})


def test_colors_expand_into_one_listing_each_sharing_a_group(tmp_path: Path) -> None:
    variants = _chair(tmp_path, colors="vert; Bleu | rouge").variants()

    assert [variant.color for variant in variants] == ["vert", "Bleu", "rouge"]
    assert [product_name(variant) for variant in variants] == [
        "chaise-camping-vert",
        "chaise-camping-bleu",
        "chaise-camping-rouge",
    ]
    assert {variant.group for variant in variants} == {"chaise-camping"}
    assert all(variant.price == 90 and variant.colors == () for variant in variants)


def test_a_row_without_colors_is_one_listing_and_keeps_its_id(tmp_path: Path) -> None:
    row = _chair(tmp_path)

    assert row.variants() == [row]
    assert product_name(row) == "chaise-camping"
    assert row.group is None


def test_a_blank_colors_cell_means_no_variants(tmp_path: Path) -> None:
    assert _chair(tmp_path, colors="   ").colors == ()
    assert _chair(tmp_path, colors=None).colors == ()


def test_colors_take_a_quoted_comma_list_and_drop_duplicates_and_blanks(tmp_path: Path) -> None:
    """`Vert` and `vert ` would both become `chaise-camping-vert`, so they are one colour."""
    assert _chair(tmp_path, colors="vert, bleu, Vert ,,  ").colors == ("vert", "bleu")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"colors": "vert", "color": "bleu"}, "not both"),
        ({"photo_color": "vert"}, "only means something next to `colors`"),
        ({"colors": "أخضر"}, "no letters or digits"),
        ({"colors": "x" * 101}, "over 100"),
        ({"group": "chaise"}, "Extra inputs are not permitted"),  # set by expansion, never by a CSV column
    ],
)
def test_bad_colour_cells_are_refused(tmp_path: Path, fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _chair(tmp_path, **fields)


def test_a_variant_of_an_untitled_row_is_named_after_its_folder(tmp_path: Path) -> None:
    row = ProductRow(product_folder=tmp_path / "Tente 2P", price=500, colors="vert")

    assert [product_name(variant) for variant in row.variants()] == ["tente-2p-vert"]


def test_a_single_colour_product_keeps_its_id_and_gets_no_group(tmp_path: Path) -> None:
    row = _chair(tmp_path, color="vert")

    assert (row.color, row.group, product_name(row)) == ("vert", None, "chaise-camping")


def test_only_a_single_listing_can_go_by_its_photo_folders_name(tmp_path: Path) -> None:
    """Otherwise every colour of a product would claim the same output folder."""
    assert _chair(tmp_path).folder_alias == "chaise"
    assert {variant.folder_alias for variant in _chair(tmp_path, colors="vert; bleu").variants()} == {None}


def test_the_photo_colour_is_the_one_the_csv_names_or_the_single_colour(tmp_path: Path) -> None:
    assert _chair(tmp_path, colors="vert; bleu", photo_color="vert").variants()[1].staged_from_color == "vert"
    assert _chair(tmp_path, colors="vert; bleu").variants()[0].staged_from_color is None  # recolour all
    assert _chair(tmp_path, color="vert").staged_from_color == "vert"  # its photo already shows it
    assert _chair(tmp_path).staged_from_color is None


def test_a_csv_with_a_colors_column_loads_and_flattens_in_file_order(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "product_folder,title,price,colors,photo_color\n"
        f"{tmp_path / 'chaise'},Chaise Camping,90,vert; bleu,vert\n"
        f"{tmp_path / 'table'},Table Pliante,250,,\n",
    )

    batch = load_products_csv(path)

    assert len(batch.rows) == 2  # the folders are still listed once each
    assert [product_name(listing) for listing in batch.listings()] == [
        "chaise-camping-vert",
        "chaise-camping-bleu",
        "table-pliante",
    ]
    assert batch.listings()[0].photo_color == "vert"


def test_a_variant_that_collides_with_another_product_is_reported_with_its_row(tmp_path: Path) -> None:
    """`Tente` in `vert` is `tente-vert`, which a product titled `Tente Vert` already is."""
    path = _write_csv(
        tmp_path,
        "product_folder,title,price,colors\n"
        f"{tmp_path / 'a'},Tente,500,vert; bleu\n"
        f"{tmp_path / 'b'},Tente Vert,300,\n",
    )

    with pytest.raises(ValueError, match=r"rows 2 and 3 both become the product id `tente-vert`"):
        load_products_csv(path)
