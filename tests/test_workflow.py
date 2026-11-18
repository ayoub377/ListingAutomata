"""Tests for scripts/workflow.py — graph wiring, node behaviour and the CSV batch, no network.

The image node's HTTP call and the chat model are replaced with fakes, so what is
checked is what the workflow itself decides: order, state passing, what each node
sends to its model, and that one failing row does not stop the batch.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
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


def test_every_shot_shows_the_product_not_its_retail_box() -> None:
    """The pool's only reference is its box. Only the context shot used to say so, and
    the angle shot drew the family printed on the box into the pool."""
    for shot, instruction in generate.SHOTS:
        prompt = generate.build_prompt("Piscine", None, instruction).lower()

        assert "retail box" in prompt, shot
        assert "leave out the box" in prompt, shot


# --------------------------------------------------------------------------
# Regenerating one part (the review app)
# --------------------------------------------------------------------------


def _fake_generate(monkeypatch, payload: bytes = b"new-image", seen: dict | None = None) -> None:
    async def fake(client, job, references, api_key, log):
        if seen is not None:
            seen["prompt"] = job.prompt
        return payload

    monkeypatch.setattr(generate, "generate", fake)


def _shot_state(**extra) -> dict:
    return {"name": "chaise", "title": "Chaise Camping", "scenery": "a forest", "shots": ["angle"], **extra}


def test_regenerate_shot_replaces_one_shot_and_keeps_the_old_one_in_prev(folders, monkeypatch) -> None:
    _cutout(folders, "chaise")
    angle, context = generate.output_path("chaise", "angle"), generate.output_path("chaise", "context")
    angle.parent.mkdir(parents=True)
    angle.write_bytes(b"old-angle")
    context.write_bytes(b"context-untouched")
    _fake_generate(monkeypatch)

    asyncio.run(workflow.regenerate_shot(_shot_state(), "key"))

    assert angle.read_bytes() == b"new-image"
    assert workflow.previous_path("chaise", "angle").read_bytes() == b"old-angle"
    assert context.read_bytes() == b"context-untouched"  # the other shot is not regenerated or billed


def test_regenerate_shot_sends_the_reviewers_note_in_the_prompt(folders, monkeypatch) -> None:
    _cutout(folders, "chaise")
    seen: dict = {}
    _fake_generate(monkeypatch, seen=seen)

    asyncio.run(workflow.regenerate_shot(_shot_state(note="The cup holder is missing."), "key"))

    assert seen["prompt"].endswith("change nothing else: The cup holder is missing.")


def test_a_failed_regeneration_leaves_the_shot_and_prev_alone(folders, monkeypatch) -> None:
    _cutout(folders, "chaise")
    angle = generate.output_path("chaise", "angle")
    angle.parent.mkdir(parents=True)
    angle.write_bytes(b"old-angle")

    async def down(*args, **kwargs):
        raise ValueError("openrouter 500")

    monkeypatch.setattr(generate, "generate", down)

    with pytest.raises(ValueError, match="openrouter 500"):
        asyncio.run(workflow.regenerate_shot(_shot_state(), "key"))

    assert angle.read_bytes() == b"old-angle"
    assert not workflow.previous_path("chaise", "angle").exists()


def test_regenerate_shot_refuses_anything_but_one_shot(folders) -> None:
    for shots in ([], ["angle", "context"]):
        with pytest.raises(ValueError, match="exactly one shot"):
            asyncio.run(workflow.regenerate_shot(_shot_state(shots=shots), "key"))


def test_a_first_regeneration_has_nothing_to_back_up(folders, monkeypatch) -> None:
    _cutout(folders, "chaise")
    _fake_generate(monkeypatch)

    asyncio.run(workflow.regenerate_shot(_shot_state(), "key"))

    assert not workflow.previous_path("chaise", "angle").exists()


def test_restore_previous_swaps_so_it_can_be_undone_too(folders) -> None:
    current, backup = generate.output_path("chaise", "angle"), workflow.previous_path("chaise", "angle")
    backup.parent.mkdir(parents=True)
    current.write_bytes(b"new")
    backup.write_bytes(b"old")

    assert workflow.restore_previous("chaise", "angle")
    assert (current.read_bytes(), backup.read_bytes()) == (b"old", b"new")

    assert workflow.restore_previous("chaise", "angle")
    assert (current.read_bytes(), backup.read_bytes()) == (b"new", b"old")


def test_restore_previous_without_a_backup_does_nothing(folders) -> None:
    generate.output_path("chaise", "angle").parent.mkdir(parents=True)
    generate.output_path("chaise", "angle").write_bytes(b"only")

    assert not workflow.restore_previous("chaise", "angle")
    assert generate.output_path("chaise", "angle").read_bytes() == b"only"


def test_description_note_follows_the_standing_instruction(tmp_path: Path) -> None:
    reference, shot = _jpeg(tmp_path / "ref.jpg"), _jpeg(tmp_path / "shot.jpg")

    plain = workflow.build_description_messages("Chaise", reference, [shot])
    noted = workflow.build_description_messages("Chaise", reference, [shot], note=" Name the brand exactly. ")
    blank = workflow.build_description_messages("Chaise", reference, [shot], note="  ")

    assert blank[1].content[0] == plain[1].content[0]
    text = noted[1].content[0]["text"]
    assert text.startswith(plain[1].content[0]["text"])
    assert text.endswith("keep every other rule above: Name the brand exactly.")


def test_regenerate_description_rewrites_the_text_and_touches_no_image(folders) -> None:
    _cutout(folders, "chaise")
    angle = _jpeg(generate.output_path("chaise", "angle"))
    before = angle.read_bytes()
    path = workflow.description_path("chaise")
    path.write_text(
        '{"title": "Chaise", "fr": "ancien", "ar": "قديم", "images": ["product_angle.jpg"], "extra": 1}',
        encoding="utf-8",
    )
    llm = FakeLLM(workflow.Description(french="Nouveau texte.", arabic="نص جديد."))

    result = asyncio.run(workflow.regenerate_description("chaise", "Chaise Camping", "Say folding.", llm))

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == result
    assert (saved["fr"], saved["ar"]) == ("Nouveau texte.", "نص جديد.")
    assert saved["title"] == "Chaise" and saved["extra"] == 1  # merged, not replaced
    assert angle.read_bytes() == before
    assert not path.with_name(path.name + ".tmp").exists()
    assert "Say folding." in llm.writer.messages[1].content[0]["text"]


def test_regenerate_description_needs_a_reference_and_an_image(folders) -> None:
    llm = FakeLLM(workflow.Description(french="x", arabic="y"))
    with pytest.raises(ValueError, match="no cutouts"):
        asyncio.run(workflow.regenerate_description("chaise", "Chaise", None, llm))

    _cutout(folders, "chaise")
    with pytest.raises(ValueError, match="no staged images"):
        asyncio.run(workflow.regenerate_description("chaise", "Chaise", None, llm))


# --------------------------------------------------------------------------
# Colour variants
# --------------------------------------------------------------------------


def _variant_product(folders: Path, color: str, photo_color: str | None = "vert") -> generate.Product:
    photos = folders / "photos"
    photos.mkdir(exist_ok=True)
    return generate.Product(
        name=f"chaise-camping-{color}",
        title="Chaise Camping",
        scenery="a forest",
        aliases=("chaise-camping", "chaise"),
        folder=photos,
        color=color,
        photo_color=photo_color,
        group="chaise-camping",
    )


def test_plan_inputs_keeps_each_colour_in_its_own_folder_but_shares_the_cutouts(folders) -> None:
    _cutout(folders, "chaise-camping")  # images/chaise-camping/: made once for the whole product
    products = [_variant_product(folders, "vert"), _variant_product(folders, "bleu")]

    inputs, missing = workflow.plan_inputs(products, ["angle"], False)

    assert missing == []
    assert [state["name"] for state in inputs] == ["chaise-camping-vert", "chaise-camping-bleu"]
    assert {state["cutouts"] for state in inputs} == {"chaise-camping"}
    assert [(state["color"], state["photo_color"]) for state in inputs] == [("vert", "vert"), ("bleu", "vert")]


def test_a_variants_cutouts_default_to_its_products_name_when_none_exist_yet(folders) -> None:
    inputs, _ = workflow.plan_inputs([_variant_product(folders, "vert")], ["angle"], False)

    assert inputs[0]["cutouts"] == "chaise-camping"  # segmented into the product's folder, not its own
    assert inputs[0]["name"] == "chaise-camping-vert"


def test_a_plain_product_state_has_none_of_the_variant_keys(folders) -> None:
    _cutout(folders, "chaise-camping")
    photos = folders / "photos"
    photos.mkdir()
    plain = generate.Product(name="chaise-camping", title="Chaise Camping", scenery=None, folder=photos)

    (state,), _ = workflow.plan_inputs([plain], ["angle"], False)

    assert not {"cutouts", "color", "photo_color"} & set(state)


def test_variants_stage_from_the_shared_cutouts_and_only_recolour_the_ones_that_need_it(folders, monkeypatch) -> None:
    reference = _cutout(folders, "chaise-camping")
    seen: dict[str, dict] = {}

    async def fake_generate(client, job, references, api_key, log):
        seen[job.name] = {"prompt": job.prompt, "references": references}
        return b"image-" + job.name.encode()

    monkeypatch.setattr(generate, "generate", fake_generate)

    for color in ("vert", "bleu"):
        asyncio.run(
            workflow.make_generate_image("key")(
                {
                    "name": f"chaise-camping-{color}",
                    "cutouts": "chaise-camping",
                    "title": "Chaise Camping",
                    "shots": ["angle"],
                    "color": color,
                    "photo_color": "vert",
                }
            )
        )

    assert generate.output_path("chaise-camping-vert", "angle").read_bytes() == b"image-chaise-camping-vert"
    assert generate.output_path("chaise-camping-bleu", "angle").read_bytes() == b"image-chaise-camping-bleu"
    assert not generate.output_path("chaise-camping", "angle").exists()  # nothing filed under the product itself
    assert all(call["references"] == [reference] for call in seen.values())
    assert "version of the product" not in seen["chaise-camping-vert"]["prompt"]  # the photo is already green
    assert "this is the bleu version" in seen["chaise-camping-bleu"]["prompt"]


def test_a_variant_with_no_cutouts_anywhere_is_segmented_into_its_products_folder(folders, monkeypatch) -> None:
    segmented: list[str] = []

    def fake_segment_photos(name: str, folder: Path, resolution: int = 1024) -> list[Path]:
        segmented.append(name)
        return [_cutout(folders, name)]

    async def fake_generate(client, job, references, api_key, log):
        return b"image"

    monkeypatch.setattr(workflow, "segment_photos", fake_segment_photos)
    monkeypatch.setattr(generate, "generate", fake_generate)

    asyncio.run(
        workflow.make_generate_image("key")(
            {
                "name": "chaise-camping-bleu",
                "cutouts": "chaise-camping",
                "title": "Chaise Camping",
                "shots": ["angle"],
                "folder": folders / "photos",
                "color": "bleu",
            }
        )
    )

    assert segmented == ["chaise-camping"]  # the cutouts are the product's, not the variant's


def test_the_description_request_names_the_colour_of_the_variant(tmp_path: Path) -> None:
    reference, shot = _jpeg(tmp_path / "ref.jpg"), _jpeg(tmp_path / "shot.jpg")

    plain = workflow.build_description_messages("Chaise", reference, [shot])[1].content[0]["text"]
    coloured = workflow.build_description_messages("Chaise", reference, [shot], color=" bleu ")[1].content[0]["text"]

    assert "version" not in plain
    assert coloured.startswith(plain)
    assert "This listing is the bleu version" in coloured
    assert "may show the product in another colour" in coloured


def test_find_cutouts_returns_the_first_name_that_has_them(folders) -> None:
    _cutout(folders, "chaise-camping")

    assert workflow.find_cutouts("chaise-camping-bleu", None, "chaise-camping", "chaise") == "chaise-camping"
    assert workflow.find_cutouts("nothing", None) is None


def test_regenerating_a_variants_description_reads_the_shared_cutouts_and_names_the_colour(folders) -> None:
    _cutout(folders, "chaise-camping")
    _jpeg(generate.output_path("chaise-camping-bleu", "angle"))
    llm = FakeLLM(workflow.Description(french="Chaise bleue.", arabic="كرسي أزرق."))

    asyncio.run(
        workflow.regenerate_description(
            "chaise-camping-bleu", "Chaise Camping", None, llm, cutouts="chaise-camping", color="bleu"
        )
    )

    assert "This listing is the bleu version" in llm.writer.messages[1].content[0]["text"]
    saved = json.loads(workflow.description_path("chaise-camping-bleu").read_text(encoding="utf-8"))
    assert saved["fr"] == "Chaise bleue."
