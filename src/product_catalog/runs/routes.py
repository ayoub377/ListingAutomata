"""The /runs pages: upload, preview, start, monitor, and the log/trace routes.

Registered onto the review app by `register_run_routes`, so the whole tool is one
localhost server with one set of safety rules: every POST needs the HTMX header
(see `require_htmx` in review/app.py), and nothing here pushes to Meta. A finished run
links to the review board, which stays the only place a product is approved and pushed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from product_catalog.runs.logs import LEVELS, MAX_SHOWN, tail_text
from product_catalog.runs.runner import RunManager, RunRejected
from product_catalog.runs.store import Run
from product_catalog.runs.tabular import MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")  # also what keeps `..` out of file paths
STAGES = ("product", "segment", "generate_image", "write_description", "generate", "host")
BASE_KEYS = {"ts", "level", "run_id", "product", "stage", "event", "msg", "traceback", "source", "time"}


def view_events(events: list[dict]) -> list[dict]:
    """Events shaped for the template: a short time, and the extra fields as one JSON string."""
    shown = []
    for event in events:
        extras = {key: value for key, value in event.items() if key not in BASE_KEYS}
        shown.append(
            {
                "time": str(event.get("ts", ""))[11:23],
                "level": str(event.get("level", "INFO")),
                "product": event.get("product") or "",
                "stage": event.get("stage") or "",
                "event": event.get("event", ""),
                "msg": event.get("msg", ""),
                "traceback": event.get("traceback", ""),
                "source": event.get("source", ""),
                "extras": json.dumps(extras, ensure_ascii=False, default=str) if extras else "",
            }
        )
    return shown


def register_run_routes(
    app: FastAPI,
    manager: RunManager,
    templates: Jinja2Templates,
    require_htmx: Callable[..., None],
    app_log: Path | None = None,
) -> None:
    store, logs = manager.store, manager.logs
    guard = [Depends(require_htmx)]

    def get_run(run_id: str) -> Run:
        run = store.get(run_id) if RUN_ID.match(run_id) else None
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return run

    def render(request: Request, name: str, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    def runs_context(error: str | None = None) -> dict[str, Any]:
        return {
            "runs": store.recent(),
            "active": manager.active,
            "error": error,
            "shots": manager.workflow.all_shots,
        }

    def panel_context(run: Run, notice: str | None = None) -> dict[str, Any]:
        items = store.items(run.id)
        return {
            "run": run,
            "items": items,
            "notice": notice,
            "busy": manager.active if manager.active != run.id else None,
            "has_log": logs.exists(run.id),
        }

    # ---- history and upload -------------------------------------------

    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(request: Request):
        return render(request, "runs.html", **runs_context())

    @app.get("/runs/table", response_class=HTMLResponse)
    def runs_table(request: Request):
        return render(request, "_run_table.html", **runs_context())

    @app.post("/runs", response_class=HTMLResponse, dependencies=guard)
    async def upload(
        request: Request,
        file: Annotated[UploadFile, File()],
        shots: Annotated[list[str] | None, Form()] = None,
        force: Annotated[bool, Form()] = False,
    ):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            # Off the event loop: validating a batch reads the disk and, the first time, imports the workflow.
            run = await asyncio.to_thread(manager.upload, file.filename or "upload", data, shots or [], force)
        except RunRejected as error:
            logger.warning("upload of %r rejected: %s", file.filename, error)
            return render(request, "_run_form.html", **runs_context(str(error)))
        logger.info("upload of %r accepted as run %s", file.filename, run.id)
        return Response(status_code=204, headers={"HX-Redirect": f"/runs/{run.id}"})

    # ---- one run -------------------------------------------------------

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        return render(request, "run_detail.html", **panel_context(get_run(run_id)))

    @app.get("/runs/{run_id}/panel", response_class=HTMLResponse)
    def run_panel(request: Request, run_id: str):
        return render(request, "_run_panel.html", **panel_context(get_run(run_id)))

    @app.post("/runs/{run_id}/start", response_class=HTMLResponse, dependencies=guard)
    async def start(request: Request, run_id: str):
        run = get_run(run_id)
        notice = None
        try:
            run = manager.start(run.id)
        except RunRejected as error:
            notice = str(error)
        return render(request, "_run_panel.html", **panel_context(run, notice))

    # ---- logs and traces -----------------------------------------------

    @app.get("/runs/{run_id}/logs.jsonl")
    def run_log_file(run_id: str):
        run = get_run(run_id)
        path = logs.path(run.id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no log for this run (it may have been pruned)")
        return FileResponse(path, media_type="application/x-ndjson", filename=f"{run.id}.jsonl")

    @app.get("/runs/{run_id}/logs")
    def run_logs(
        request: Request,
        run_id: str,
        after: int = 0,
        level: str = "",
        product: str = "",
        stage: str = "",
        q: str = "",
        format: str = "",
    ):
        run = get_run(run_id)
        level = level.upper() if level.upper() in LEVELS else ""
        filters = {"level": level or None, "product": product or None, "stage": stage or None, "q": q or None}
        if after:  # a tail poll: only what was written since, run events only
            page = logs.read(run.id, after=after, limit=MAX_SHOWN, **filters)
        else:  # a full load: add the provider requests (Seedream, PixelVault) from the run's time window
            names = [item.product for item in store.items(run.id)]
            page = logs.merged(run.id, names, run.started_at or run.created_at, run.finished_at, **filters)

        if format == "json":
            return JSONResponse(
                {"run_id": run.id, "status": run.status, "events": page.events,
                 "next_offset": page.next_offset, "hidden": page.hidden}
            )

        qs = urlencode({"level": level, "product": product, "stage": stage, "q": q})
        context = {
            "run": run, "events": view_events(page.events), "next_offset": page.next_offset,
            "hidden": page.hidden, "live": run.active, "qs": qs, "has_log": logs.exists(run.id),
            "filters": {"level": level, "product": product, "stage": stage, "q": q},
            "levels": LEVELS, "stages": STAGES, "products": [i.product for i in store.items(run.id)],
        }
        if request.headers.get("hx-request") == "true":
            return render(request, "_log_rows.html", **context)
        return render(request, "run_logs.html", **context)

    @app.get("/logs", response_class=HTMLResponse)
    def app_logs(request: Request, lines: int = 300):
        lines = min(max(lines, 20), 2000)
        text = tail_text(app_log, lines) if app_log else []
        return render(request, "app_log.html", lines=text, count=lines, configured=app_log is not None)
