"""Tests for scripts/workflow.py — graph wiring, node behaviour and the CSV batch, no network.

The image node's HTTP call and the chat model are replaced with fakes, so what is
checked is what the workflow itself decides: order, state passing, what each node
sends to its model, and that one failing row does not stop the batch.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_workflow_module():
    """scripts/ is not a package, so load the script by path (it imports its sibling)."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("workflow", SCRIPTS / "workflow.py")
    module = importlib.util.module_from_spec(spec)
    # LangGraph resolves the state's string annotations through sys.modules.
    sys.modules["workflow"] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_workflow_module()
generate = workflow.generate


class FakeWriter:
    def __init__(self, result) -> None:
        self.result = result
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.result


class FakeLLM:
    def __init__(self, result) -> None:
        self.writer = FakeWriter(result)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.writer


def _jpeg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), (10, 120, 200)).save(path, "JPEG")
    return path


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's .env may switch LangSmith on; tests must never upload."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generate, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(generate, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(generate, "LOG_PATH", tmp_path / "output" / "log.jsonl")
    generate.data_url.cache_clear()
    return tmp_path


def _cutout(root: Path, name: str, filename: str = "product_1_seg.png") -> Path:
    path = root / "images" / name / filename
    path.parent.mkdir(parents=True)
    Image.new("RGBA", (40, 30), (200, 40, 40, 255)).save(path)
    return path


def test_graph_runs_the_image_node_then_the_description_node() -> None:
    order: list[str] = []

    async def image(state):
        order.append("image")
        return {"references": [Path("ref.png")], "generated_images": [Path("out.jpg")]}

    async def text(state):
        order.append("text")
        assert state["generated_images"] == [Path("out.jpg")]  # the image node's output arrives
        return {"description_fr": "fr", "description_ar": "ar"}

    result = asyncio.run(workflow.build_graph(image, text).ainvoke({"name": "chaise", "title": "Chaise"}))

    assert order == ["image", "text"]
    assert (result["description_fr"], result["description_ar"]) == ("fr", "ar")


def test_generate_image_stages_the_references_and_writes_the_shot(folders, monkeypatch) -> None:
    reference = _cutout(folders, "chaise")
    seen = {}

    async def fake_generate(client, job, references, api_key, log):
        seen.update(job=job, references=references, api_key=api_key)
        return b"jpeg-bytes"

    monkeypatch.setattr(generate, "generate", fake_generate)

    state = asyncio.run(
        workflow.make_generate_image("key")(
            {"name": "chaise", "title": "Chaise Camping", "scenery": "a forest", "shots": ["angle"]}
        )
    )

    destination = generate.output_path("chaise", "angle")
    assert destination.read_bytes() == b"jpeg-bytes"
    assert state == {"references": [reference], "generated_images": [destination]}
    assert seen["references"] == [reference]
    assert "a forest" in seen["job"].prompt and "Chaise Camping" in seen["job"].prompt
    assert seen["api_key"] == "key"


def test_generate_image_reuses_a_shot_already_on_disk(folders, monkeypatch) -> None:
    """Every generation is billed, so an existing file is not paid for twice."""
    _cutout(folders, "chaise")
    existing = _jpeg(generate.output_path("chaise", "angle"))

    async def boom(*args, **kwargs):
        raise AssertionError("must not call the API for a shot that exists")

    monkeypatch.setattr(generate, "generate", boom)

    state = asyncio.run(workflow.make_generate_image("key")({"name": "chaise", "title": "Chaise", "shots": ["angle"]}))

    assert state["generated_images"] == [existing]


def test_generate_image_force_regenerates(folders, monkeypatch) -> None:
    _cutout(folders, "chaise")
    destination = _jpeg(generate.output_path("chaise", "angle"))

    async def fake_generate(*args, **kwargs):
        return b"new-bytes"

    monkeypatch.setattr(generate, "generate", fake_generate)

    asyncio.run(
        workflow.make_generate_image("key")({"name": "chaise", "title": "Chaise", "force": True})
    )

    assert destination.read_bytes() == b"new-bytes"


def test_generate_image_fails_loudly_without_cutouts_or_photos(folders) -> None:
    """No reference means no request - never spend money on a reference-less render."""
    with pytest.raises(ValueError, match="no product folder"):
        asyncio.run(workflow.make_generate_image("key")({"name": "never-segmented", "title": "X"}))


def _fake_segmentation(monkeypatch) -> list[Path]:
    """Stand in for BiRefNet: every photo 'segments' to an opaque red square."""
    seen: list[Path] = []
    monkeypatch.setattr(workflow, "load_segmenter", lambda: (object(), "cpu"))

    def fake_segment(image, model, device, resolution):
        return Image.new("RGBA", (20, 20), (200, 40, 40, 255))

    monkeypatch.setattr(workflow.segment, "segment", fake_segment)
    original = workflow.segment.load_photo
    monkeypatch.setattr(
        workflow.segment, "load_photo", lambda path: (seen.append(path), original(path))[1]
    )
    return seen


@pytest.fixture
def photo_folder(folders, monkeypatch):
    monkeypatch.setattr(workflow.segment, "IMAGES_DIR", folders / "images")
    folder = folders / "photos" / "pool"
    _jpeg(folder / "pool_a.jpg")
    _jpeg(folder / "pool_b.jpg")
    (folder / "notes.txt").write_text("stray")
    return folder


def test_segment_photos_writes_numbered_cutouts_like_segment_py(photo_folder, folders, monkeypatch) -> None:
    seen = _fake_segmentation(monkeypatch)

    written = workflow.segment_photos("pool", photo_folder)

    assert [path.name for path in written] == ["product_1_seg.png", "product_2_seg.png"]
    assert all(path.parent == folders / "images" / "pool" and path.is_file() for path in written)
    assert [path.name for path in seen] == ["pool_a.jpg", "pool_b.jpg"]  # the .txt never reaches the model


def test_segment_photos_leaves_existing_cutouts_alone(photo_folder, folders, monkeypatch) -> None:
    """Rerunning a batch must not pay the model again for photos already cut out."""
    seen = _fake_segmentation(monkeypatch)
    _cutout(folders, "pool", "product_1_seg.png")

    workflow.segment_photos("pool", photo_folder)

    assert [path.name for path in seen] == ["pool_b.jpg"]


def test_segment_photos_rejects_a_missing_or_empty_folder(folders, tmp_path) -> None:
    with pytest.raises(ValueError, match="not found"):
        workflow.segment_photos("pool", tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no photos"):
        workflow.segment_photos("pool", empty)


def test_generate_image_segments_the_folder_first_then_generates(photo_folder, folders, monkeypatch) -> None:
    """The point of the tool: hand over the photo folder, get a staged image."""
    _fake_segmentation(monkeypatch)
    order: list[str] = []
    seen = {}

    async def fake_generate(client, job, references, api_key, log):
        order.append("generate")
        seen["references"] = references
        return b"jpeg-bytes"

    monkeypatch.setattr(generate, "generate", fake_generate)
    real_segment_photos = workflow.segment_photos
    monkeypatch.setattr(
        workflow, "segment_photos",
        lambda *args, **kwargs: (order.append("segment"), real_segment_photos(*args, **kwargs))[1],
    )

    state = asyncio.run(
        workflow.make_generate_image("key")(
            {"name": "pool", "title": "Piscine", "folder": photo_folder, "shots": ["angle"]}
        )
    )

    assert order == ["segment", "generate"]
    assert [path.name for path in seen["references"]] == ["product_1_seg.png", "product_2_seg.png"]
    assert state["references"] == seen["references"]


def test_generate_image_does_not_segment_when_cutouts_exist(photo_folder, folders, monkeypatch) -> None:
    _cutout(folders, "pool")

    def boom(*args, **kwargs):
        raise AssertionError("must not segment when cutouts are already on disk")

    monkeypatch.setattr(workflow, "segment_photos", boom)

    async def fake_generate(*args, **kwargs):
        return b"jpeg-bytes"

    monkeypatch.setattr(generate, "generate", fake_generate)

    asyncio.run(
        workflow.make_generate_image("key")(
            {"name": "pool", "title": "Piscine", "folder": photo_folder}
        )
    )


def test_description_node_sends_title_reference_and_generated_image(tmp_path: Path) -> None:
    reference = _jpeg(tmp_path / "reference.jpg")
    generated = _jpeg(tmp_path / "generated.jpg")
    llm = FakeLLM(workflow.Description(french="Une chaise.", arabic="كرسي."))

    state = asyncio.run(
        workflow.make_write_description(llm)(
            {"title": "Chaise Camping", "references": [reference], "generated_images": [generated]}
        )
    )

    assert llm.schema is workflow.Description
    system, human = llm.writer.messages
    text, first, second = human.content
    assert "Chaise Camping" in text["text"] and "Arabic" in text["text"]
    assert "The images after it" in text["text"]
    assert first["image_url"]["url"] == generate.data_url(reference)
    assert second["image_url"]["url"] == generate.data_url(generated)
    assert state == {"description_fr": "Une chaise.", "description_ar": "كرسي."}


def test_system_prompt_is_international_and_asks_for_benefits_and_seo() -> None:
    """The first draft was Moroccan-only; sellers are worldwide and buyers want benefits."""
    prompt = workflow.DESCRIPTION_SYSTEM.lower()

    assert "moroccan" not in prompt and "morocco" not in prompt
    assert "anywhere in the world" in prompt
    assert "what the buyer gets" in prompt
    assert "search" in prompt and "keyword stuffing" in prompt
    assert "40 to 70 words" in prompt
    assert "never invent" in prompt


def test_description_rejects_an_empty_language() -> None:
    with pytest.raises(ValueError):
        workflow.Description(french="Une chaise.", arabic="")


def test_plan_inputs_covers_every_product_with_cutouts(folders) -> None:
    _cutout(folders, "chaise")
    _cutout(folders, "piscine")
    products = [
        generate.Product("chaise", "Chaise Camping", "a forest"),
        generate.Product("piscine", "Piscine", None),
        generate.Product("tente", "Tente", None),
    ]

    inputs, missing = workflow.plan_inputs(products, ["angle"], force=True)

    assert [state["name"] for state in inputs] == ["chaise", "piscine"]
    assert inputs[0] == {
        "name": "chaise", "title": "Chaise Camping", "scenery": "a forest",
        "shots": ["angle"], "force": True,
    }
    assert missing == ["tente"]


def test_plan_inputs_queues_a_product_that_only_has_a_photo_folder(folders, tmp_path) -> None:
    """The two Bestway pools: photos on disk, nothing segmented yet - not a skip."""
    photos = tmp_path / "pool-photos"
    photos.mkdir()
    products = [
        generate.Product("pool", "Piscine", None, folder=photos),
        generate.Product("ghost", "Ghost", None, folder=tmp_path / "nowhere"),
    ]

    inputs, missing = workflow.plan_inputs(products, ["angle"], force=False)

    assert [state["name"] for state in inputs] == ["pool"]
    assert inputs[0]["folder"] == photos
    assert missing == ["ghost"]


def test_plan_inputs_files_a_product_under_its_alias_when_that_has_the_cutouts(folders) -> None:
    _cutout(folders, "chaise")
    product = generate.Product("chaise-camping", "Chaise Camping", None, ("chaise",))

    inputs, missing = workflow.plan_inputs([product], ["angle"], force=False)

    assert [state["name"] for state in inputs] == ["chaise"]
    assert missing == []


def test_run_batch_loops_every_row_and_survives_a_failure(folders, capsys) -> None:
    """Row 2 blowing up must not stop row 3, and must be named in the report."""
    async def image(state):
        if state["name"] == "bad":
            raise ValueError("seedream said no")
        return {"references": [Path("ref.png")], "generated_images": [Path(f"{state['name']}.jpg")]}

    async def text(state):
        return {"description_fr": f"fr {state['name']}", "description_ar": f"ar {state['name']}"}

    graph = workflow.build_graph(image, text)
    inputs = [{"name": name, "title": name.title()} for name in ("one", "bad", "three")]

    succeeded, failed = asyncio.run(workflow.run_batch(graph, inputs, concurrency=2))

    assert (succeeded, failed) == (2, 1)
    assert workflow.description_path("one").is_file()
    assert workflow.description_path("three").is_file()
    assert not workflow.description_path("bad").exists()
    assert "failed bad: seedream said no" in capsys.readouterr().err


def _stub_graph(fail: bool = False):
    async def image(state):
        if fail:
            raise ValueError("seedream said no")
        return {"references": [Path("ref.png")], "generated_images": [Path("product_angle.jpg")]}

    async def text(state):
        return {"description_fr": "fr", "description_ar": "ar"}

    return workflow.build_graph(image, text)


def test_run_product_returns_the_state_and_saves_the_description(folders) -> None:
    result = asyncio.run(
        workflow.run_product(_stub_graph(), {"name": "chaise", "title": "Chaise"})
    )

    assert result["description_fr"] == "fr"
    saved = workflow.description_path("chaise").read_text(encoding="utf-8")
    assert '"fr": "fr"' in saved and '"ar": "ar"' in saved


def test_run_product_propagates_a_node_failure(folders) -> None:
    """Retrying is the caller's call (a queue task may, the CLI just reports)."""
    with pytest.raises(ValueError, match="seedream said no"):
        asyncio.run(
            workflow.run_product(_stub_graph(fail=True), {"name": "bad", "title": "Bad"})
        )

    assert not workflow.description_path("bad").exists()


def test_run_product_passes_a_named_traceable_config(folders) -> None:
    seen = {}

    class SpyGraph:
        async def ainvoke(self, state, config=None):
            seen["config"] = config
            return {**state, "description_fr": "fr", "description_ar": "ar",
                    "generated_images": [Path("product_angle.jpg")]}

    asyncio.run(workflow.run_product(SpyGraph(), {"name": "chaise", "title": "Chaise"}, "b7"))

    assert seen["config"]["run_name"] == "product:chaise"
    assert seen["config"]["metadata"]["batch_id"] == "b7"


def test_graph_nodes_carry_agent_metadata_for_filtering() -> None:
    async def noop(state):
        return {}

    nodes = workflow.build_graph(noop, noop).nodes

    assert nodes["generate_image"].metadata == {"agent": "image"}
    assert nodes["write_description"].metadata == {"agent": "writer"}


def test_save_description_keeps_arabic_readable(folders) -> None:
    """ensure_ascii would turn the Arabic into \\u escapes nobody can proofread."""
    path = workflow.save_description(
        {
            "name": "chaise", "title": "Chaise", "description_fr": "Une chaise.",
            "description_ar": "كرسي قابل للطي.", "generated_images": [Path("product_angle.jpg")],
        }
    )

    text = path.read_text(encoding="utf-8")
    assert "كرسي قابل للطي." in text
    assert '"images": [' in text and '"product_angle.jpg"' in text

def test_generate_image_stages_every_shot_by_default(folders, monkeypatch) -> None:
    """One image per product was the bug: a product gets its angle and its context shot."""
    _cutout(folders, "chaise")
    prompts = []

    async def fake_generate(client, job, references, api_key, log):
        prompts.append((job.shot, job.prompt))
        return b"jpeg-" + job.shot.encode()

    monkeypatch.setattr(generate, "generate", fake_generate)

    state = asyncio.run(
        workflow.make_generate_image("key")({"name": "chaise", "title": "Chaise", "scenery": "a forest"})
    )

    assert sorted(shot for shot, _ in prompts) == ["angle", "context"]
    assert [path.name for path in state["generated_images"]] == ["product_angle.jpg", "product_context.jpg"]
    assert all(path.is_file() for path in state["generated_images"])


def test_generate_image_only_pays_for_the_missing_shot(folders, monkeypatch) -> None:
    """The chaise already has its angle: rerunning must bill the context shot alone."""
    _cutout(folders, "chaise")
    _jpeg(generate.output_path("chaise", "angle"))
    shots = []

    async def fake_generate(client, job, references, api_key, log):
        shots.append(job.shot)
        return b"jpeg"

    monkeypatch.setattr(generate, "generate", fake_generate)

    asyncio.run(workflow.make_generate_image("key")({"name": "chaise", "title": "Chaise"}))

    assert shots == ["context"]


def test_generate_image_honours_a_single_requested_shot(folders, monkeypatch) -> None:
    _cutout(folders, "chaise")
    shots = []

    async def fake_generate(client, job, references, api_key, log):
        shots.append(job.shot)
        return b"jpeg"

    monkeypatch.setattr(generate, "generate", fake_generate)

    state = asyncio.run(
        workflow.make_generate_image("key")({"name": "chaise", "title": "Chaise", "shots": ["context"]})
    )

    assert shots == ["context"]
    assert [path.name for path in state["generated_images"]] == ["product_context.jpg"]


def test_description_node_sends_every_generated_image(tmp_path: Path) -> None:
    reference = _jpeg(tmp_path / "reference.jpg")
    angle, context = _jpeg(tmp_path / "angle.jpg"), _jpeg(tmp_path / "context.jpg")

    messages = workflow.build_description_messages("Chaise", reference, [angle, context])

    urls = [part["image_url"]["url"] for part in messages[1].content[1:]]
    assert urls == [generate.data_url(path) for path in (reference, angle, context)]


def test_context_shot_shows_the_product_not_its_retail_box() -> None:
    """The pool's only reference is its box, and the model faithfully drew the box."""
    instruction = dict(generate.SHOTS)["context"].lower()

    assert "retail box" in instruction
    assert "leave the box out" in instruction
