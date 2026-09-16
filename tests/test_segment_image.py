"""Tests for scripts/segment.py — the geometry and naming, not the model.

Everything here runs without torch: loading BiRefNet to check that a slug is
lowercase would be absurd.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from product_catalog import ProductRow

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_segment_module():
    """scripts/ is not a package, so load the script by path."""
    spec = importlib.util.spec_from_file_location("segment", PROJECT_ROOT / "scripts" / "segment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


segment_script = _load_segment_module()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Piscine Bestway 4.6m", "piscine-bestway-4-6m"),
        ("Piscine Gonflée — été", "piscine-gonflee-ete"),
        ("3.96x84-bestway", "3-96x84-bestway"),
        ("  spaced  out  ", "spaced-out"),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert segment_script.slugify(raw) == expected


def test_product_name_prefers_title() -> None:
    row = ProductRow(product_folder=r"C:\photos\IMG_4821", price=4300, title="Piscine Bestway 4.6m")
    assert segment_script.product_name(row) == "piscine-bestway-4-6m"


def test_product_name_falls_back_to_folder_name() -> None:
    """Title is normally blank in the CSV — the vision step fills it in later."""
    row = ProductRow(product_folder=r"C:\photos\3.96x84-bestway", price=1300)
    assert row.title is None
    assert segment_script.product_name(row) == "3-96x84-bestway"


def test_folder_photos_are_sorted_and_filtered(tmp_path: Path) -> None:
    """Numbering must be stable across runs, and non-photos must not reach BiRefNet."""
    for name in ("b.jpeg", "a.png", "notes.txt", "Thumbs.db"):
        (tmp_path / name).touch()
    (tmp_path / "subfolder").mkdir()

    assert segment_script.folder_photos(tmp_path) == [tmp_path / "a.png", tmp_path / "b.jpeg"]


def test_output_path_layout() -> None:
    path = segment_script.output_path("piscine-bestway-4-6m", 2)
    assert path.parent.name == "piscine-bestway-4-6m"
    assert path.name == "product_2_seg.png"
    assert path.parent.parent == segment_script.IMAGES_DIR


def test_load_photo_honours_exif_orientation(tmp_path: Path) -> None:
    """Phone shots carry Orientation=6; ignoring it lists the product on its side."""
    photo = tmp_path / "sideways.jpeg"
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 CW
    Image.new("RGB", (40, 30), "green").save(photo, exif=exif)

    assert Image.open(photo).size == (40, 30)  # what a bare open sees
    assert segment_script.load_photo(photo).size == (30, 40)


def test_load_photo_leaves_an_untagged_photo_alone(tmp_path: Path) -> None:
    photo = tmp_path / "plain.jpeg"
    Image.new("RGB", (40, 30), "green").save(photo)

    assert segment_script.load_photo(photo).size == (40, 30)


def test_firm_up_clips_both_tails_and_keeps_the_ramp() -> None:
    """Faint floor reflections go, the near-opaque body goes solid, edges survive."""
    floor, ceiling = round(segment_script.MATTE_FLOOR * 255), round(segment_script.MATTE_CEILING * 255)
    matte = Image.new("L", (7, 1))
    matte.putdata([0, 20, floor, floor + 1, (floor + ceiling) // 2, ceiling, 253])

    result = list(segment_script.firm_up(matte).get_flattened_data())

    assert result[:3] == [0, 0, 0]  # ghosts erased
    assert result[5:] == [255, 255]  # body fully opaque
    assert result[3] > 0 and 100 < result[4] < 160  # edge ramp preserved


def test_letterbox_pads_to_square_without_squashing() -> None:
    """A 3:4 photo must keep its proportions; stretching walks the matte off it."""
    image = Image.new("RGB", (600, 800), "red")
    padded, (offset_x, offset_y, box_width, box_height) = segment_script.letterbox(image, 1024)

    assert padded.size == (1024, 1024)
    assert (box_width, box_height) == (768, 1024)
    assert box_width / box_height == pytest.approx(image.width / image.height, abs=0.01)
    assert (offset_x, offset_y) == (128, 0)
    assert padded.getpixel((512, 512)) == (255, 0, 0)  # product
    assert padded.getpixel((1, 512)) == (0, 0, 0)  # padding, left bar
    assert padded.getpixel((1022, 512)) == (0, 0, 0)  # padding, right bar


def test_letterbox_box_round_trips_a_matte_to_the_original_size() -> None:
    """Cropping the box out of the square matte lands back on the input photo."""
    image = Image.new("RGB", (1200, 675), "blue")
    _, (offset_x, offset_y, box_width, box_height) = segment_script.letterbox(image, 1024)

    matte = Image.new("L", (1024, 1024), 0)
    restored = matte.crop(
        (offset_x, offset_y, offset_x + box_width, offset_y + box_height)
    ).resize(image.size)

    assert restored.size == image.size
