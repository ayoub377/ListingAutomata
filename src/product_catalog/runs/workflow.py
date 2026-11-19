"""What the runs layer needs from the AI workflow, and the real implementation of it.

`RunWorkflow` is the seam: the runner and the routes only know this protocol, so tests
swap in a fake and never import torch or call a model. `ScriptWorkflow` is the real
one and wraps scripts/workflow.py the same way the review app's regenerator does,
importing the scripts folder by path because it is not a package.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

# Called as on_stage(node_name, seconds_since_the_previous_stage) when a graph node finishes.
StageCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class PlanItem:
    name: str
    title: str
    new_images: int
    note: str = ""
    state: Any = field(default=None, compare=False, repr=False)  # the graph input, opaque here


@dataclass(frozen=True)
class RunPlan:
    items: list[PlanItem]
    missing: list[str]  # products with neither cutouts nor a photo folder
    shots: list[str]
    cost_per_image: float
    model: str

    @property
    def new_images(self) -> int:
        return sum(item.new_images for item in self.items)

    @property
    def est_cost(self) -> float:
        return round(self.new_images * self.cost_per_image, 2)


class RunWorkflow(Protocol):
    all_shots: list[str]

    def missing(self) -> list[str]:
        """Names of the .env settings that must be filled in before a run can start."""

    def plan(self, csv_path: Path, shots: list[str], force: bool) -> RunPlan:
        """What a run would do. Raises ValueError (every bad row listed) for an invalid CSV."""

    def new_batch_id(self) -> str: ...

    async def run_item(self, item: PlanItem, batch_id: str, on_stage: StageCallback) -> None:
        """One product through the whole graph. Failures propagate."""

    def flush(self) -> None:
        """Block until queued traces are uploaded."""


class ScriptWorkflow:
    def __init__(self) -> None:
        self._workflow: Any = None
        self._graph: Any = None

    def _module(self) -> Any:
        if self._workflow is None:
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_DIR))
            import workflow

            self._workflow = workflow
        return self._workflow

    @property
    def all_shots(self) -> list[str]:
        return list(self._module().ALL_SHOTS)

    def missing(self) -> list[str]:
        key = self._module().generate.load_keys()[0]
        return [] if key else ["OPEN_ROUTER_API_KEY"]

    def plan(self, csv_path: Path, shots: list[str], force: bool) -> RunPlan:
        workflow = self._module()
        generate = workflow.generate
        products = generate.products_from_csv(csv_path)
        inputs, missing = workflow.plan_inputs(products, shots, force)
        items = []
        for state in inputs:
            name = state["name"]
            fresh = sum(1 for shot in shots if force or not generate.output_path(name, shot).is_file())
            note = "" if generate.reference_cutouts(name) else "will segment the photos first"
            items.append(PlanItem(name, state["title"], fresh, note, state))
        return RunPlan(items, missing, shots, generate.COST_PER_IMAGE, generate.MODEL_ID)

    def new_batch_id(self) -> str:
        from product_catalog import new_batch_id

        return new_batch_id()

    async def run_item(self, item: PlanItem, batch_id: str, on_stage: StageCallback) -> None:
        """Same steps as `workflow.run_product`, but streamed so each node reports when it ends.

        `run_product` only returns at the very end, which would leave a 90 second product
        silent. The state has no reducers, so folding each node's update into a dict gives
        the same final state `ainvoke` returns.
        """
        from product_catalog import run_config

        workflow = self._module()
        if self._graph is None:
            key = workflow.generate.load_keys()[0]
            if not key:
                raise ValueError("OPEN_ROUTER_API_KEY is not set in .env")
            self._graph = workflow.default_graph(key)
        result = dict(item.state)
        mark = time.monotonic()
        config = run_config(item.state, batch_id)
        async for update in self._graph.astream(item.state, config=config, stream_mode="updates"):
            for node, delta in update.items():
                result.update(delta or {})
                on_stage(node, time.monotonic() - mark)
                mark = time.monotonic()
        workflow.save_description(result)

    def flush(self) -> None:
        from product_catalog import flush

        flush()
