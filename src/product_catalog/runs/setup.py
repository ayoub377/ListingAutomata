"""Wiring for the real app: where the files live and which handlers are attached."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from product_catalog.runs.logs import RedactingFormatter, RunLogs, install_run_handler
from product_catalog.runs.runner import RunManager
from product_catalog.runs.store import RunStore
from product_catalog.runs.workflow import RunWorkflow, ScriptWorkflow

APP_LOG_BYTES = 1_000_000
APP_LOG_BACKUPS = 3
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_app_log(path: Path) -> None:
    """Write the app's own log (startup, uploads, recovery) to `path`, rotating, secrets masked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for old in [h for h in root.handlers if getattr(h, "_product_catalog_app_log", False)]:
        root.removeHandler(old)
        old.close()
    handler = RotatingFileHandler(path, maxBytes=APP_LOG_BYTES, backupCount=APP_LOG_BACKUPS, encoding="utf-8")
    handler.setFormatter(RedactingFormatter(FORMAT))
    handler.setLevel(logging.INFO)
    handler._product_catalog_app_log = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    package_logger = logging.getLogger("product_catalog")
    if package_logger.level == logging.NOTSET:
        package_logger.setLevel(logging.INFO)


def build(root: Path, workflow: RunWorkflow | None = None) -> tuple[RunManager, Path]:
    """The run manager for a project folder, and the app log it writes to."""
    output = root / "output"
    app_log = output / "app.log"
    configure_app_log(app_log)
    logs = RunLogs(output / "runs", generation_log=output / "generation-log.jsonl")
    install_run_handler(logs)
    manager = RunManager(RunStore(output / "runs.db"), logs, workflow or ScriptWorkflow())
    return manager, app_log
