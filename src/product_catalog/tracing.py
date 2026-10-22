"""LangSmith wiring: what a trace stores, and how a run is named and grouped.

Tracing itself is switched on by the environment (`LANGSMITH_TRACING`,
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`); nothing here sends anything when it is
off. This module only decides how runs are labelled and keeps the pixels out of them.
"""

from __future__ import annotations

import re
import uuid
from functools import lru_cache
from typing import Any

from langsmith.utils import tracing_is_enabled

_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=]+)")


def redact_images(value: Any) -> Any:
    """Replace inline base64 images with a size placeholder, at any depth.

    The writer node is sent two data URLs of hundreds of KB each. Left in, every
    trace would carry them to LangSmith: slow uploads, and the product photos
    leaving the machine. `<image 312 KB>` keeps the shape of the request readable.
    """
    if isinstance(value, str):
        return _DATA_URL.sub(lambda match: f"<image {len(match.group(1)) * 3 // 4 // 1024} KB>", value)
    if isinstance(value, dict):
        return {key: redact_images(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_images(item) for item in value]
    if hasattr(value, "model_dump"):  # a LangChain message or other pydantic object
        return redact_images(value.model_dump())
    return value


@lru_cache(maxsize=1)
def client():
    """The one LangSmith client, built on first use so a disabled run never makes it."""
    from langsmith import Client

    return Client(hide_inputs=redact_images, hide_outputs=redact_images)


def new_batch_id() -> str:
    """Groups every product of one CSV run in LangSmith."""
    return uuid.uuid4().hex[:12]


def run_config(state: dict, batch_id: str) -> dict:
    """Graph config for one product: a named, tagged, filterable trace.

    Each node's run nests under it automatically; `metadata` is what lets you
    filter by product or by batch in the dashboard.
    """
    config: dict[str, Any] = {
        "run_name": f"product:{state['name']}",
        "tags": ["product-catalog"],
        "metadata": {
            "product": state["name"],
            "title": state.get("title"),
            "shots": state.get("shots"),
            "batch_id": batch_id,
        },
    }
    if tracing_is_enabled():
        from langchain_core.tracers.langchain import LangChainTracer

        # Passing the tracer is how the redacting client gets used for every run
        # in the graph, including the LLM call that holds the images.
        config["callbacks"] = [LangChainTracer(client=client())]
    return config


def flush() -> None:
    """Block until queued traces are uploaded.

    Traces go out on a background thread, so a short-lived process (the CLI, a
    queue task) can exit with the last products' runs still unsent.
    """
    if not tracing_is_enabled():
        return
    from langchain_core.tracers.langchain import wait_for_all_tracers

    wait_for_all_tracers()
    client().flush()
