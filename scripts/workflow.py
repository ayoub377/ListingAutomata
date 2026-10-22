"""LangGraph workflow: every row of products.csv in, a staged image and a FR/AR description out.

    python scripts/workflow.py                      # every row in products.csv
    python scripts/workflow.py --csv batch.csv
    python scripts/workflow.py --force              # regenerate images already on disk
    python scripts/workflow.py --dry-run            # print the plan, call nothing

Each product runs through a two-node graph, each node with one job:

    generate_image    - takes the product's reference cutouts and has Seedream stage
                        them into a catalogue photo (output/<name>/product_<shot>.jpg)
    write_description - takes the title, the generated image and the reference, and
                        writes a buyer-focused, SEO-friendly description in French
                        and Arabic (output/<name>/description.json)

The CSV is walked row by row and the graph runs once per product, a few at a time.
One product failing is reported and does not stop the others. The image call,
prompt and paths are scripts/generate.py's own, imported rather than copied, so
the two entry points cannot drift apart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from product_catalog import flush, new_batch_id, run_config

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # noqa: E402  (sibling scripts; need the path line above)
import segment  # noqa: E402

DESCRIPTION_MODEL = "google/gemini-2.5-flash"
ALL_SHOTS = [shot for shot, _ in generate.SHOTS]

DESCRIPTION_SYSTEM = (
    "You write product descriptions for online marketplaces and shop catalogues "
    "that sell to buyers anywhere in the world. You are given the product title, a "
    "reference photo of the real product and a staged photo of it.\n"
    "Write for a buyer deciding in seconds, and for the search engine that sends "
    "them there:\n"
    "- Open with what the product is, using the words a buyer would type to find "
    "it (take them from the title), then its main benefit.\n"
    "- Then say what the buyer gets: the concrete benefits of what is visible - "
    "material, shape, colour, size cues, how it is used - not adjectives. Prefer "
    "'folds flat to fit in a car boot' over 'convenient and practical'.\n"
    "- Be concise: 40 to 70 words, one short paragraph, plain sentences.\n"
    "- Use the main search terms naturally, once or twice. No keyword stuffing, no "
    "repetition, no lists of tags.\n"
    "- Plain text only: no emojis, no markdown, no ALL CAPS, no exclamation marks.\n"
    "- Never invent specifications, dimensions, capacities, brands, certifications "
    "or features that are not visible in the photos or stated in the title. Make no "
    "superlative or unprovable claims (best, number one, guaranteed), and say "
    "nothing about price, shipping, stock or warranty.\n"
    "- The staged scene is marketing backdrop. Do not describe it as part of the "
    "product, and do not mention people or props that are not sold with it."
)
DESCRIPTION_INSTRUCTION = (
    "Title: {title}\n\n"
    "Image 1 is the reference photo of the product. The images after it are "
    "staged photos of the same product.\n"
    "Write the description in French and again in Modern Standard Arabic. Each is "
    "written natively for readers of that language and for search in that language, "
    "not translated word for word; both say the same true things."
)


class Description(BaseModel):
    """What the description node must return, in both languages."""

    french: str = Field(min_length=1, description="Description en français.")
    arabic: str = Field(min_length=1, description="الوصف بالعربية الفصحى.")


class CatalogState(TypedDict, total=False):
    """Everything the two nodes pass along. Inputs first, then each node's output."""

    name: str
    title: str
    scenery: str | None
    folder: Path  # the product's photos; segmented into cutouts when none exist yet
    shots: list[str]  # which shots to stage; every shot in generate.SHOTS when omitted
    force: bool
    references: list[Path]
    generated_images: list[Path]  # in shot order, hero first
    description_fr: str
    description_ar: str


# --------------------------------------------------------------------------
# Segmentation tool
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_segmenter(device: str = "auto"):
    """BiRefNet, loaded once per process - the weights are ~900 MB.

    Returns (model, device). Cached so a batch of fifty products pays the load once.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return segment.load_model(device), device


def segment_photos(name: str, folder: Path, resolution: int = 1024) -> list[Path]:
    """Cut out every photo in `folder` into images/<name>/, like scripts/segment.py.

    Same naming and the same rule: a photo whose cutout already exists is left alone.
    Blocking (PyTorch), so callers run it in a thread.
    """
    if not folder.is_dir():
        raise ValueError(f"{name}: photo folder not found: {folder}")
    photos = segment.folder_photos(folder)
    if not photos:
        raise ValueError(f"{name}: no photos in {folder}")

    plan = segment.plan_destinations([(name, photo) for photo in photos])
    pending = [(photo, destination) for photo, destination in plan if not destination.exists()]
    if pending:
        model, device = load_segmenter()
        for photo, destination in pending:
            cutout = segment.segment(segment.load_photo(photo), model, device, resolution)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cutout.save(destination)
    return [destination for _, destination in plan]


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def make_generate_image(api_key: str, log: Path | None = None):
    """The image node: segment the product photos if needed, then stage the image.

    Segmentation is a tool this node calls itself, so the graph only needs the
    product's photo folder. It shows up in the trace as its own `segment_cutouts` run.
    """
    # BiRefNet is one model on one device: products take turns instead of racing it.
    segmenting = asyncio.Lock()

    @traceable(
        run_type="tool",
        name="segment_cutouts",
        process_outputs=lambda cutouts: {"cutouts": [path.name for path in cutouts]},
    )
    async def ensure_cutouts(name: str, folder: str) -> list[Path]:
        async with segmenting:
            await asyncio.to_thread(segment_photos, name, Path(folder))
        return generate.reference_cutouts(name)

    # Seedream is a plain HTTP call, so LangChain never sees it. Tracing it as an
    # LLM run is what puts it in the trace with its own cost; the bytes it returns
    # are summarised rather than uploaded.
    @traceable(
        run_type="llm",
        name="seedream_generate",
        metadata={"model": generate.MODEL_ID},
        process_outputs=lambda image: {"image_bytes": len(image)},
    )
    async def render(name: str, shot: str, prompt: str, references: list[str]) -> bytes:
        import httpx

        job = generate.Job(name, shot, prompt, generate.output_path(name, shot))
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            image = await generate.generate(
                client, job, [Path(path) for path in references], api_key, log or generate.LOG_PATH
            )
        # Set after the call: a request that raised was not billed.
        if run := get_current_run_tree():
            run.set(usage_metadata={"total_cost": generate.COST_PER_IMAGE})
        return image

    async def stage(state: CatalogState, shot: str, references: list[Path]) -> Path:
        """One shot of one product: reuse it if it is on disk, else render and save it."""
        destination = generate.output_path(state["name"], shot)
        # Generation is billed per image, so a shot already on disk is reused.
        if destination.is_file() and not state.get("force"):
            return destination
        instruction = dict(generate.SHOTS)[shot]
        prompt = generate.build_prompt(state["title"], state.get("scenery"), instruction)
        image = await render(state["name"], shot, prompt, [str(path) for path in references])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image)
        return destination

    async def generate_image(state: CatalogState) -> CatalogState:
        name = state["name"]
        shots = state.get("shots") or ALL_SHOTS
        references = generate.reference_cutouts(name)
        if not references:
            if not state.get("folder"):
                raise ValueError(
                    f"no cutouts in {generate.cutout_dir(name)} and no product folder "
                    f"to segment them from"
                )
            references = await ensure_cutouts(name, str(state["folder"]))

        # Every shot of the product from the same references, so they stay one product.
        images = await asyncio.gather(*(stage(state, shot, references) for shot in shots))
        return {"references": references, "generated_images": list(images)}

    return generate_image


def build_description_messages(
    title: str, reference: Path, generated: list[Path]
) -> list[SystemMessage | HumanMessage]:
    """The vision request: title text, then the reference, then every staged image."""
    return [
        SystemMessage(DESCRIPTION_SYSTEM),
        HumanMessage(
            content=[
                {"type": "text", "text": DESCRIPTION_INSTRUCTION.format(title=title)},
                {"type": "image_url", "image_url": {"url": generate.data_url(reference)}},
                *(
                    {"type": "image_url", "image_url": {"url": generate.data_url(image)}}
                    for image in generated
                ),
            ]
        ),
    ]


def make_write_description(llm):
    """The description node: title + reference + generated image in, FR and AR out.

    `llm` is any LangChain chat model that takes images; structured output pins the
    reply to the two fields instead of parsing prose.
    """
    writer = llm.with_structured_output(Description)

    async def write_description(state: CatalogState) -> CatalogState:
        messages = build_description_messages(
            state["title"], state["references"][0], state["generated_images"]
        )
        result = await writer.ainvoke(messages)
        return {"description_fr": result.french, "description_ar": result.arabic}

    return write_description


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def build_graph(generate_image, write_description):
    """START -> generate_image -> write_description -> END."""
    graph = StateGraph(CatalogState)
    # `agent` metadata is how each node's runs are filtered on their own in LangSmith.
    graph.add_node("generate_image", generate_image, metadata={"agent": "image"})
    graph.add_node("write_description", write_description, metadata={"agent": "writer"})
    graph.add_edge(START, "generate_image")
    graph.add_edge("generate_image", "write_description")
    graph.add_edge("write_description", END)
    return graph.compile()


def default_graph(openrouter_key: str, log: Path | None = None):
    """The production wiring: Seedream for the image, a vision model for the text."""
    from langchain_openrouter import ChatOpenRouter

    llm = ChatOpenRouter(model=DESCRIPTION_MODEL, api_key=openrouter_key, temperature=0.3)
    return build_graph(make_generate_image(openrouter_key, log), make_write_description(llm))


# --------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------


def description_path(name: str) -> Path:
    return generate.OUTPUT_DIR / name / "description.json"


def save_description(state: CatalogState) -> Path:
    """Write one finished product's FR/AR text next to its image."""
    destination = description_path(state["name"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "title": state["title"],
                "fr": state["description_fr"],
                "ar": state["description_ar"],
                "images": [image.name for image in state["generated_images"]],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def plan_inputs(
    products: list[generate.Product], shots: list[str], force: bool
) -> tuple[list[CatalogState], list[str]]:
    """Graph inputs for every product we can start from, and the names we cannot.

    A product starts from cutouts already on disk (filed under whichever name they
    have, as generate.py does) or, failing that, from its photo folder, which the
    image node segments. Only a product with neither is reported missing.
    """
    inputs: list[CatalogState] = []
    missing: list[str] = []
    for product in products:
        source = generate.cutout_name(product)
        has_photos = product.folder is not None and product.folder.is_dir()
        if source is None and not has_photos:
            missing.append(product.name)
            continue
        state: CatalogState = {
            "name": source or product.name,
            "title": product.title,
            "scenery": product.scenery,
            "shots": shots,
            "force": force,
        }
        if has_photos:
            state["folder"] = product.folder
        inputs.append(state)
    return inputs, missing


async def run_product(graph, state: CatalogState, batch_id: str | None = None) -> CatalogState:
    """One product through the whole graph, its description saved beside its image.

    The single entry point for a unit of work: the CLI batch calls it per row, and
    a queue task (Celery behind FastAPI) can call it the same way. Both nodes stay
    inside one run because the writer needs the file the image node just wrote.
    Failures propagate; whoever calls decides whether to retry.

    Each call is one LangSmith trace named `product:<name>`; `batch_id` groups the
    traces of a CSV run.
    """
    result = await graph.ainvoke(state, config=run_config(state, batch_id or new_batch_id()))
    save_description(result)
    return result


async def run_batch(graph, inputs: list[CatalogState], concurrency: int) -> tuple[int, int]:
    """Run every input through `run_product`. Returns (succeeded, failed).

    `return_exceptions` keeps one bad row from cancelling the rest: a product that
    fails is reported by name and the loop carries on.
    """
    limit = asyncio.Semaphore(concurrency)
    batch_id = new_batch_id()

    async def one(state: CatalogState) -> CatalogState:
        async with limit:
            return await run_product(graph, state, batch_id)

    results = await asyncio.gather(*(one(state) for state in inputs), return_exceptions=True)
    succeeded = failed = 0
    for state, result in zip(inputs, results):
        if isinstance(result, Exception):
            failed += 1
            print(f"failed {state['name']}: {result}", file=sys.stderr)
            continue
        succeeded += 1
        print(f"done   {state['name']}  {generate.relative(description_path(state['name']))}")
    return succeeded, failed


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=generate.PROJECT_ROOT / "products.csv")
    parser.add_argument(
        "--shot", action="append", choices=ALL_SHOTS, dest="shots",
        help="Stage only this shot (repeatable). Default: every shot.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate images already on disk.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only.")
    parser.add_argument(
        "--concurrency", type=int, default=generate.DEFAULT_CONCURRENCY,
        help="Products in flight at once.",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    try:
        products = generate.products_from_csv(args.csv)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    shots = args.shots or ALL_SHOTS
    inputs, missing = plan_inputs(products, shots, args.force)
    for name in missing:
        print(f"skip   {name} (no cutouts and its photo folder was not found)", file=sys.stderr)
    if not inputs:
        print("nothing to do", file=sys.stderr)
        return 1

    new_images = sum(
        1
        for state in inputs
        for shot in shots
        if args.force or not generate.output_path(state["name"], shot).is_file()
    )
    print(
        f"{len(inputs)} product(s), {new_images} new image(s) via {generate.MODEL_ID}, "
        f"~${new_images * generate.COST_PER_IMAGE:.2f}"
    )
    if args.dry_run:
        for state in inputs:
            note = "" if generate.reference_cutouts(state["name"]) else "  [will segment first]"
            print(f"  {state['name']}  ({state['title']}){note}")
        return 0

    openrouter_key, _ = generate.load_keys()
    if not openrouter_key:
        print("OPEN_ROUTER_API_KEY is not set (.env or environment)", file=sys.stderr)
        return 1

    graph = default_graph(openrouter_key)
    try:
        succeeded, failed = asyncio.run(run_batch(graph, inputs, args.concurrency))
    finally:
        flush()  # traces upload in the background; don't exit before the last ones go
    print(f"{succeeded} done, {failed} failed, {len(missing)} skipped")
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())