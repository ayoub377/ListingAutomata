"""Tests for scripts/publish.py - the graph, the hosting node and the batch."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

from product_catalog import MetaListing, ProductRow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_publish_module():
    spec = importlib.util.spec_from_file_location("publish", SCRIPTS / "publish.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["publish"] = module
    spec.loader.exec_module(module)
    return module


publish = _load_publish_module()
generate = publish.generate


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(generate, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(generate, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(generate, "LOG_PATH", tmp_path / "output" / "log.jsonl")
    return tmp_path


def _row(tmp_path: Path, **fields: object) -> ProductRow:
    values = {"product_folder": tmp_path / "chaise", "title": "Chaise Camping", "price": 90}
    return ProductRow(**{**values, **fields})


def _generated(name: str = "chaise-camping", shots: tuple[str, ...] = ("angle", "context")) -> None:
    """What scripts/workflow.py leaves behind for one product."""
    for shot in shots:
        path = generate.output_path(name, shot)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), (10, 120, 200)).save(path, "JPEG")
    publish.description_path(name).parent.mkdir(parents=True, exist_ok=True)
    publish.description_path(name).write_text(
        json.dumps({"title": "Chaise Camping", "fr": "Chaise pliante.", "ar": "كرسي.", "images": []}),
        encoding="utf-8",
    )


def _state(tmp_path: Path, **fields: object) -> dict:
    return {
        "name": "chaise-camping",
        "folder_name": "chaise-camping",
        "row": _row(tmp_path),
        "lang": "fr",
        "force": False,
        **fields,
    }


# --- whatsapp link ----------------------------------------------------------


def test_whatsapp_link_carries_the_product_name() -> None:
    link = publish.whatsapp_link("+212 600-000000", "Chaise Camping")

    assert link.startswith("https://wa.me/212600000000?text=")
    assert "Chaise%20Camping" in link


def test_whatsapp_link_needs_a_number() -> None:
    with pytest.raises(ValueError, match="WHATSAPP_NUMBER"):
        publish.whatsapp_link(None, "Chaise")


# --- load_product -----------------------------------------------------------


def test_load_product_reads_the_text_in_the_chosen_language(folders: Path) -> None:
    _generated()

    fr = asyncio.run(publish.load_product(_state(folders)))
    ar = asyncio.run(publish.load_product(_state(folders, lang="ar")))

    assert fr["description"] == "Chaise pliante."
    assert ar["description"] == "كرسي."
    assert [p.name for p in fr["images"]] == ["product_angle.jpg", "product_context.jpg"]


def test_load_product_without_generated_output_says_what_to_run(folders: Path) -> None:
    with pytest.raises(ValueError, match="run scripts/workflow.py"):
        asyncio.run(publish.load_product(_state(folders)))


def test_load_product_without_images_is_an_error(folders: Path) -> None:
    _generated(shots=())

    with pytest.raises(ValueError, match="no staged images"):
        asyncio.run(publish.load_product(_state(folders)))


# --- upload_images ----------------------------------------------------------


def test_upload_hosts_each_shot_and_records_the_manifest(folders: Path, monkeypatch) -> None:
    _generated()
    hosted: list[str] = []

    async def fake_host(client, path, folder, api_key, log):
        hosted.append(path.name)
        return f"https://cdn/{folder}/{path.name}"

    monkeypatch.setattr(generate, "host", fake_host)
    images = [generate.output_path("chaise-camping", s) for s in ("angle", "context")]

    result = asyncio.run(publish.make_upload_images("key")(_state(folders, images=images)))

    assert hosted == ["product_angle.jpg", "product_context.jpg"]
    assert result["image_urls"] == [
        "https://cdn/chaise-camping/product_angle.jpg",
        "https://cdn/chaise-camping/product_context.jpg",
    ]
    manifest = generate.read_manifest("chaise-camping")
    assert [entry["shot"] for entry in manifest["images"]] == ["angle", "context"]


def test_upload_skips_shots_already_hosted(folders: Path, monkeypatch) -> None:
    _generated()
    publish.write_manifest(
        "chaise-camping",
        {"product": "chaise-camping", "images": [
            {"shot": "angle", "file": "product_angle.jpg", "url": "https://cdn/old-angle.jpg"}
        ]},
    )
    hosted: list[str] = []

    async def fake_host(client, path, folder, api_key, log):
        hosted.append(path.name)
        return "https://cdn/new.jpg"

    monkeypatch.setattr(generate, "host", fake_host)
    images = [generate.output_path("chaise-camping", s) for s in ("angle", "context")]

    result = asyncio.run(publish.make_upload_images("key")(_state(folders, images=images)))

    assert hosted == ["product_context.jpg"]
    assert result["image_urls"] == ["https://cdn/old-angle.jpg", "https://cdn/new.jpg"]


def test_force_uploads_everything_again(folders: Path, monkeypatch) -> None:
    _generated()
    publish.write_manifest(
        "chaise-camping",
        {"product": "chaise-camping", "images": [
            {"shot": "angle", "file": "product_angle.jpg", "url": "https://cdn/old.jpg"}
        ]},
    )
    hosted: list[str] = []

    async def fake_host(client, path, folder, api_key, log):
        hosted.append(path.name)
        return "https://cdn/new.jpg"

    monkeypatch.setattr(generate, "host", fake_host)
    images = [generate.output_path("chaise-camping", s) for s in ("angle", "context")]

    asyncio.run(publish.make_upload_images("key")(_state(folders, images=images, force=True)))

    assert hosted == ["product_angle.jpg", "product_context.jpg"]


def test_a_failed_upload_keeps_the_shots_already_hosted(folders: Path, monkeypatch) -> None:
    _generated()

    async def fake_host(client, path, folder, api_key, log):
        if path.name == "product_context.jpg":
            raise ValueError("pixelvault 500")
        return "https://cdn/angle.jpg"

    monkeypatch.setattr(generate, "host", fake_host)
    images = [generate.output_path("chaise-camping", s) for s in ("angle", "context")]

    with pytest.raises(ValueError, match="pixelvault 500"):
        asyncio.run(publish.make_upload_images("key")(_state(folders, images=images)))

    assert [e["shot"] for e in generate.read_manifest("chaise-camping")["images"]] == ["angle"]


# --- build_listing ----------------------------------------------------------


def _built(folders: Path, row: ProductRow | None = None, number: str | None = "212600000000"):
    state = _state(
        folders,
        title="Chaise Camping",
        description="Chaise pliante.",
        image_urls=["https://cdn/1.jpg", "https://cdn/2.jpg"],
    )
    if row:
        state["row"] = row
    return asyncio.run(publish.make_build_listing(number)(state))["listing"]


def test_build_listing_uses_the_row_the_text_and_the_hosted_urls(folders: Path) -> None:
    listing = _built(folders)

    assert isinstance(listing, MetaListing)
    assert (listing.id, listing.price, listing.currency) == ("chaise-camping", Decimal(90), "MAD")
    assert listing.image_urls == ["https://cdn/1.jpg", "https://cdn/2.jpg"]
    assert listing.link.startswith("https://wa.me/212600000000")


def test_a_link_in_the_csv_beats_the_whatsapp_one(folders: Path) -> None:
    listing = _built(folders, _row(folders, link="https://shop.example/chaise"), number=None)

    assert listing.link == "https://shop.example/chaise"


def test_build_listing_without_link_or_number_fails(folders: Path) -> None:
    with pytest.raises(ValueError, match="WHATSAPP_NUMBER"):
        _built(folders, number=None)


# --- graph and batch --------------------------------------------------------


def test_graph_runs_load_upload_build_in_order() -> None:
    order: list[str] = []

    async def load(state):
        order.append("load")
        return {"title": "t", "description": "d", "images": [Path("a.jpg")]}

    async def upload(state):
        order.append("upload")
        assert state["images"] == [Path("a.jpg")]
        return {"image_urls": ["https://cdn/a.jpg"]}

    async def build(state):
        order.append("build")
        assert state["image_urls"] == ["https://cdn/a.jpg"]
        return {"listing": "built"}

    result = asyncio.run(publish.build_graph(load, upload, build).ainvoke({"name": "x"}))

    assert order == ["load", "upload", "build"]
    assert result["listing"] == "built"


def test_plan_inputs_finds_output_under_the_name_or_the_folder_alias(folders: Path) -> None:
    _generated("chaise")  # workflow.py was run from the photo folder `chaise`
    rows = [_row(folders), _row(folders, title="Table", product_folder=folders / "table")]

    inputs, missing = publish.plan_inputs(rows, "fr", False)

    assert [(i["name"], i["folder_name"]) for i in inputs] == [("chaise-camping", "chaise")]
    assert missing == ["table"]


def test_plan_inputs_can_be_narrowed_to_named_products(folders: Path) -> None:
    _generated("chaise-camping")
    _generated("table")
    rows = [_row(folders), _row(folders, title="Table", product_folder=folders / "table")]

    inputs, _ = publish.plan_inputs(rows, "fr", False, only=["table"])

    assert [i["name"] for i in inputs] == ["table"]


def test_one_failing_product_does_not_stop_the_rest(folders: Path, capsys) -> None:
    async def load(state):
        if state["name"] == "bad":
            raise ValueError("boom")
        return {}

    async def upload(state):
        return {"image_urls": ["https://cdn/1.jpg"]}

    async def build(state):
        return {"listing": MetaListing(
            id=state["name"], title="T", description="D", price=1,
            link="https://x", image_urls=["https://cdn/1.jpg"],
        )}

    graph = publish.build_graph(load, upload, build)
    inputs = [{"name": "bad"}, {"name": "good"}]

    listings, failed = asyncio.run(publish.run_batch(graph, inputs, 2))

    assert [listing.id for listing in listings] == ["good"]
    assert failed == ["bad"]
    assert "failed bad: boom" in capsys.readouterr().err


def test_write_files_produces_the_batch_body_and_the_feed(folders: Path) -> None:
    listing = MetaListing(
        id="chaise-camping", title="Chaise", description="D", price=Decimal(90),
        link="https://x", image_urls=["https://cdn/1.jpg"],
    )

    batch, feed = publish.write_files([listing])

    body = json.loads(batch.read_text(encoding="utf-8"))
    assert body["allow_upsert"] is True and body["requests"][0]["data"]["id"] == "chaise-camping"
    rows = list(csv.DictReader(feed.open(encoding="utf-8")))
    assert rows[0]["id"] == "chaise-camping" and rows[0]["price"] == "90.00 MAD"


def test_report_status_flags_errors(capsys) -> None:
    clean = publish.report_status("h1", {"status": "finished", "errors": [], "errors_total_count": 0})
    dirty = publish.report_status(
        "h2", {"status": "finished", "errors": [{"message": "bad price"}], "errors_total_count": 1}
    )

    assert clean is True and dirty is False
    assert "bad price" in capsys.readouterr().err


def test_dry_run_builds_listings_without_uploading_or_pushing(folders: Path, monkeypatch, capsys) -> None:
    _generated()
    csv_path = folders / "products.csv"
    csv_path.write_text(
        f"product_folder,title,price\n{folders / 'chaise'},Chaise Camping,90\n", encoding="utf-8"
    )

    async def boom(*args, **kwargs):
        raise AssertionError("a dry run must not touch the network")

    monkeypatch.setattr(generate, "host", boom)
    monkeypatch.setattr(publish, "push_to_meta", boom)
    monkeypatch.setenv("WHATSAPP_NUMBER", "212600000000")
    monkeypatch.setattr(generate, "load_keys", lambda: (None, None))

    code = publish.main(["--csv", str(csv_path), "--dry-run"])

    assert code == 0
    out = capsys.readouterr().out
    assert "UPLOAD://chaise-camping/product_angle.jpg" in out
    assert not (generate.OUTPUT_DIR / "meta-batch.json").exists()


def test_main_stops_without_meta_credentials(folders: Path, monkeypatch, capsys) -> None:
    _generated()
    csv_path = folders / "products.csv"
    csv_path.write_text(
        f"product_folder,title,price\n{folders / 'chaise'},Chaise Camping,90\n", encoding="utf-8"
    )
    for key in ("META_CATALOG_ID", "META_ACCESS_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(generate, "load_keys", lambda: (None, "pv-key"))

    assert publish.main(["--csv", str(csv_path)]) == 1
    assert "META_CATALOG_ID, META_ACCESS_TOKEN not set" in capsys.readouterr().err
