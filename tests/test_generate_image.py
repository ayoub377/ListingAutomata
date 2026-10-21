"""Tests for scripts/generate.py — the prompts, paths and payloads, not the API.

Nothing here reaches OpenRouter or PixelVault: the two network functions are the
thin ends of the script, and everything worth getting wrong happens before them.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_generate_module():
    """scripts/ is not a package, so load the script by path."""
    spec = importlib.util.spec_from_file_location("generate", PROJECT_ROOT / "scripts" / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_script = _load_generate_module()


@pytest.fixture
def cutouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the script at a temporary images/ and return a maker for cutouts."""
    monkeypatch.setattr(generate_script, "IMAGES_DIR", tmp_path / "images")

    def make(name: str, *filenames: str) -> Path:
        folder = tmp_path / "images" / name
        folder.mkdir(parents=True)
        for filename in filenames:
            (folder / filename).write_bytes(b"\x89PNG\r\n\x1a\n")
        return folder

    return make


def test_reference_cutouts_puts_the_primary_first(cutouts) -> None:
    """The primary is the shot the scenes are built on, so the model sees it first."""
    cutouts("chaise", "product_1_seg.png", "product_primary_seg.png", "product_2_seg.png")

    assert [path.name for path in generate_script.reference_cutouts("chaise")] == [
        "product_primary_seg.png",
        "product_1_seg.png",
        "product_2_seg.png",
    ]


def test_reference_cutouts_sorts_numerically_not_alphabetically(cutouts) -> None:
    """Plain name order would hand the model product_10 before product_2."""
    cutouts("chaise", "product_10_seg.png", "product_2_seg.png", "product_1_seg.png")

    assert [path.name for path in generate_script.reference_cutouts("chaise")] == [
        "product_1_seg.png",
        "product_2_seg.png",
        "product_10_seg.png",
    ]


def test_reference_cutouts_caps_the_count(cutouts) -> None:
    """Every extra reference is billed tokens for the same product from the same day."""
    names = [f"product_{index}_seg.png" for index in range(1, generate_script.MAX_REFERENCES + 4)]
    cutouts("chaise", *names)

    assert len(generate_script.reference_cutouts("chaise")) == generate_script.MAX_REFERENCES


def test_reference_cutouts_ignores_strays(cutouts) -> None:
    cutouts("chaise", "product_1_seg.png", "notes.txt", "chaise_kass1.jpeg")

    assert [path.name for path in generate_script.reference_cutouts("chaise")] == [
        "product_1_seg.png"
    ]


def test_reference_cutouts_is_empty_when_nothing_was_segmented(cutouts) -> None:
    """The batch must report this and move on, not send a reference-less request."""
    assert generate_script.reference_cutouts("never-segmented") == []


def test_build_prompt_stages_the_scenery() -> None:
    prompt = generate_script.build_prompt(
        "Chaise Camping", "forest near a lake, green grass", "Shot from the side."
    )

    assert "Place the product in this setting: forest near a lake, green grass." in prompt
    assert "Subject: Chaise Camping." in prompt
    assert "Shot from the side." in prompt
    assert generate_script.WHITE_BACKDROP not in prompt


def test_build_prompt_insists_the_scene_is_visible() -> None:
    """The first run came back plain white, so the setting is now stated as the point."""
    prompt = generate_script.build_prompt("Chaise", "a forest near a lake", "Angle.")

    assert "must be clearly visible" in prompt
    assert "fill the background" in prompt


def test_build_prompt_falls_back_to_white() -> None:
    """A blank scenery cell means no scene was asked for — that is not an error."""
    prompt = generate_script.build_prompt("Piscine Bestway", None, "Shot from the side.")

    assert generate_script.WHITE_BACKDROP in prompt
    assert "Place the product in this setting" not in prompt


def test_build_prompt_requires_whole_people_rather_than_banning_them() -> None:
    """The first run produced a man with a cropped face; the Bestway row wants children."""
    prompt = generate_script.build_prompt("Piscine", "backyard, two children running", "Angle.")

    assert "never cut through the face or body" in prompt
    assert "no person" not in prompt.lower()
    assert "without people" not in prompt.lower()


def test_build_prompt_asks_for_a_stunning_realistic_photograph() -> None:
    prompt = generate_script.build_prompt("Chaise", "a forest", "Angle.")

    assert "Ultra-realistic, visually stunning" in prompt
    assert "no watermark" in prompt


def test_build_prompt_does_not_double_the_full_stop() -> None:
    prompt = generate_script.build_prompt("Chaise", "a sunny lawn.", "Angle.")

    assert "setting: a sunny lawn." in prompt
    assert ".." not in prompt


def test_shot_instructions_do_not_summon_a_user() -> None:
    """`in use` is what put an unrequested man in the first context render."""
    for _, instruction in generate_script.SHOTS:
        assert "in use" not in instruction.lower()


def _cutout_file(path: Path, size: tuple[int, int] = (3024, 4032)) -> Path:
    """A phone-sized RGBA cutout: opaque product, transparent surround."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.paste((200, 40, 40, 255), (0, 0, size[0] // 2, size[1] // 2))
    image.save(path)
    return path


def test_encode_reference_fits_within_the_edge_budget(tmp_path: Path) -> None:
    """A raw 3024x4032 cutout is what built the 47 MB request that timed out."""
    encoded = generate_script.encode_reference(_cutout_file(tmp_path / "product_1_seg.png"))

    with Image.open(io.BytesIO(encoded)) as shrunk:
        assert max(shrunk.size) == generate_script.REFERENCE_MAX_EDGE
        assert shrunk.size == (1152, 1536)  # 3:4 kept, not squashed
    assert len(encoded) < 1_000_000


def test_encode_reference_flattens_transparency_onto_white(tmp_path: Path) -> None:
    """Alpha the provider may not read would otherwise render as a black surround."""
    encoded = generate_script.encode_reference(_cutout_file(tmp_path / "product_1_seg.png"))

    with Image.open(io.BytesIO(encoded)) as shrunk:
        assert shrunk.mode == "RGB"
        corner = shrunk.getpixel((shrunk.width - 5, shrunk.height - 5))  # was transparent
        assert all(channel > 245 for channel in corner)
        product = shrunk.getpixel((5, 5))  # was opaque red
        assert product[0] > 150 and product[1] < 90


def test_data_url_is_an_inline_jpeg(tmp_path: Path) -> None:
    cutout = _cutout_file(tmp_path / "product_1_seg.png", size=(80, 60))

    url = generate_script.data_url(cutout)

    assert url.startswith("data:image/jpeg;base64,")
    payload = base64.b64decode(url.removeprefix("data:image/jpeg;base64,"))
    assert payload == generate_script.encode_reference(cutout)


def test_build_payload_matches_the_documented_shape(tmp_path: Path) -> None:
    """input_references is the field name OpenRouter documents; a typo fails silently."""
    cutout = _cutout_file(tmp_path / "product_1_seg.png", size=(80, 60))

    payload = generate_script.build_payload("a chaise in a forest", [cutout])

    assert payload["model"] == "bytedance-seed/seedream-4.5"
    assert payload["prompt"] == "a chaise in a forest"
    assert payload["output_format"] == "jpeg"
    assert payload["input_references"] == [
        {"type": "image_url", "image_url": {"url": generate_script.data_url(cutout)}}
    ]


def test_decode_image_round_trips() -> None:
    payload = {"data": [{"b64_json": base64.b64encode(b"jpeg-bytes").decode()}]}

    assert generate_script.decode_image(payload) == b"jpeg-bytes"


def test_decode_image_surfaces_an_api_error() -> None:
    payload = {"error": {"message": "insufficient credits"}}

    with pytest.raises(ValueError, match="insufficient credits"):
        generate_script.decode_image(payload)


def test_decode_image_rejects_an_empty_response() -> None:
    """A 200 with no image would otherwise write a zero-byte jpg and call it done."""
    with pytest.raises(ValueError, match="no image"):
        generate_script.decode_image({"data": []})


def test_output_path_layout() -> None:
    path = generate_script.output_path("chaise-camping", "angle")

    assert path.name == "product_angle.jpg"
    assert path.parent.name == "chaise-camping"
    assert path.parent.parent == generate_script.OUTPUT_DIR


def test_plan_generations_covers_every_shot_of_every_product() -> None:
    plan = generate_script.plan_generations(
        [
            generate_script.Product("chaise", "Chaise Camping", "a forest"),
            generate_script.Product("piscine", "Piscine", None),
        ]
    )

    assert [(job.name, job.shot) for job in plan] == [
        ("chaise", "angle"),
        ("chaise", "context"),
        ("piscine", "angle"),
        ("piscine", "context"),
    ]
    assert "a forest" in plan[0].prompt
    assert generate_script.WHITE_BACKDROP in plan[2].prompt


def test_cutout_name_falls_back_to_the_photo_folder_alias(cutouts) -> None:
    """Batch mode looked for images/chaise-camping/ and skipped images/chaise/."""
    cutouts("chaise", "product_1_seg.png")
    product = generate_script.Product("chaise-camping", "Chaise Camping", "a forest", ("chaise",))

    assert generate_script.cutout_name(product) == "chaise"


def test_cutout_name_prefers_the_canonical_name(cutouts) -> None:
    cutouts("chaise-camping", "product_1_seg.png")
    cutouts("chaise", "product_1_seg.png")
    product = generate_script.Product("chaise-camping", "Chaise Camping", "a forest", ("chaise",))

    assert generate_script.cutout_name(product) == "chaise-camping"


def test_cutout_name_is_none_when_nothing_was_segmented(cutouts) -> None:
    product = generate_script.Product("chaise-camping", "Chaise Camping", None, ("chaise",))

    assert generate_script.cutout_name(product) is None


def test_a_product_answers_to_its_photo_folder_name() -> None:
    """images/chaise/ and the CSV row `Chaise Camping` are the same product."""
    product = generate_script.Product("chaise-camping", "Chaise Camping", "a forest", ("chaise",))

    assert product.answers_to("chaise-camping")
    assert product.answers_to("chaise")
    assert not product.answers_to("piscine")


def test_products_from_csv_aliases_the_photo_folder(tmp_path: Path) -> None:
    """Without this alias, --name chaise found no row and lost the scenery to white."""
    folder = tmp_path / "chaise"
    folder.mkdir()
    (folder / "chaise_kass1.jpeg").touch()
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        f"product_folder,title,price,scenery\n{folder},Chaise Camping,90,a forest near a lake\n",
        encoding="utf-8",
    )

    product = generate_script.products_from_csv(csv_path)[0]

    assert product.name == "chaise-camping"
    assert product.answers_to("chaise")
    assert product.scenery == "a forest near a lake"


def test_log_event_appends_one_json_object_per_line(tmp_path: Path) -> None:
    log = tmp_path / "generation-log.jsonl"

    generate_script.log_event({"event": "generate", "product": "chaise"}, log)
    generate_script.log_event({"event": "host", "product": "chaise"}, log)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["generate", "host"]
    assert json.loads(lines[0])["time"]  # every line is timestamped


def test_summarize_response_keeps_the_cost_and_drops_the_pixels() -> None:
    """A log line carrying 600 KB of base64 is not a log line anybody reads."""
    payload = {
        "data": [{"b64_json": "A" * 5000, "media_type": "image/jpeg"}],
        "usage": {"cost": 0.04},
    }

    summary = generate_script.summarize_response(payload, b"x" * 120)

    assert summary == {"usage": {"cost": 0.04}, "media_type": "image/jpeg", "image_bytes": 120}
    assert "b64_json" not in json.dumps(summary)


def test_merge_manifest_keeps_the_other_shot() -> None:
    """Hosting one shot on its own must not drop the URL the other one already has."""
    manifest = {"product": "chaise", "images": [
        {"shot": "angle", "file": "product_angle.jpg", "url": "https://img/old-angle.jpg"}
    ]}

    merged = generate_script.merge_manifest(
        manifest, "context", "product_context.jpg", "https://img/context.jpg"
    )

    assert [entry["shot"] for entry in merged["images"]] == ["angle", "context"]
    assert merged["images"][0]["url"] == "https://img/old-angle.jpg"


def test_merge_manifest_replaces_a_reshot_rather_than_appending() -> None:
    manifest = {"product": "chaise", "images": [
        {"shot": "angle", "file": "product_angle.jpg", "url": "https://img/old.jpg"}
    ]}

    merged = generate_script.merge_manifest(
        manifest, "angle", "product_angle.jpg", "https://img/new.jpg"
    )

    assert merged["images"] == [
        {"shot": "angle", "file": "product_angle.jpg", "url": "https://img/new.jpg"}
    ]


def test_read_manifest_is_empty_for_a_product_never_hosted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generate_script, "OUTPUT_DIR", tmp_path / "output")

    assert generate_script.read_manifest("chaise") == {"product": "chaise", "images": []}
