"""Regenerating one part of a product from the review page.

`WorkflowRegenerator` calls scripts/workflow.py, so the review app and the CLI use the
same prompt and the same nodes. Like `MetaPusher`, it imports the scripts folder by
path because that folder is not a package.

Two parts can be regenerated on their own: a single photo (about $0.04 and up to a
minute) and the description (a cheap text call). Neither touches the other.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

from product_catalog.review.catalog import ReviewProduct

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


class Regenerator(Protocol):
    """What the app needs from a regenerator. Tests swap in a fake."""

    def missing(self) -> list[str]:
        """Names of the settings that must be filled in before anything can be regenerated."""

    async def image(self, product: ReviewProduct, shot: str, note: str, scenery: str) -> None: ...

    async def description(self, product: ReviewProduct, note: str) -> None: ...

    def restore(self, product: ReviewProduct, shot: str) -> bool:
        """Swap a shot with the version a regeneration replaced. False if there is none."""


class WorkflowRegenerator:
    def __init__(self) -> None:
        self._workflow: Any = None

    def _module(self) -> Any:
        if self._workflow is None:
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_DIR))
            import workflow

            self._workflow = workflow
        return self._workflow

    def missing(self) -> list[str]:
        key = self._module().generate.load_keys()[0]
        return [] if key else ["OPEN_ROUTER_API_KEY"]

    async def image(self, product: ReviewProduct, shot: str, note: str, scenery: str) -> None:
        workflow = self._module()
        key = workflow.generate.load_keys()[0]
        if not key:
            raise ValueError("OPEN_ROUTER_API_KEY is not set in .env")
        await workflow.regenerate_shot(
            {
                "name": product.folder_name,
                "title": product.title,
                "scenery": scenery or None,
                "folder": product.row.product_folder,
                "shots": [shot],
                "note": note or None,
            },
            key,
        )

    async def description(self, product: ReviewProduct, note: str) -> None:
        workflow = self._module()
        key = workflow.generate.load_keys()[0]
        if not key:
            raise ValueError("OPEN_ROUTER_API_KEY is not set in .env")
        from langchain_openrouter import ChatOpenRouter

        llm = ChatOpenRouter(model=workflow.DESCRIPTION_MODEL, api_key=key, temperature=0.3)
        await workflow.regenerate_description(product.folder_name, product.title, note or None, llm)

    def restore(self, product: ReviewProduct, shot: str) -> bool:
        return self._module().restore_previous(product.folder_name, shot)
