"""Per-run event logs: one JSONL file per run, written as it happens, read back for the UI.

Every line is a self-contained JSON object:
    ts, level, run_id, product, stage, event, msg, plus event-specific fields
    (duration_ms, error_type, traceback, ...).

Why a file per run and not one big log: a run is the unit you investigate, the file
can be tailed by byte offset while the run is live, and deleting old runs is just
deleting folders. `output/generation-log.jsonl` (provider requests) is global and
keyed by product only, so it is merged in at read time by product and time window.
"""

from __future__ import annotations

import contextvars
import json
import logging
import shutil
import threading
import traceback as traceback_module
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from product_catalog.runs.redact import redact, redact_deep

MAX_BYTES = 5 * 1024 * 1024  # per run; past this, one marker line and then silence
KEEP_RUNS = 50  # newest run folders kept on disk
MAX_SHOWN = 1000  # events one page load returns
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
EVENTS_FILE = "events.jsonl"

current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_run", default=None)
current_product: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_product", default=None)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class LogPage:
    events: list[dict]
    next_offset: int  # pass back as `after` to get only what was written since
    hidden: int = 0  # matching events older than the page limit, not returned


class RunLogs:
    def __init__(
        self, runs_dir: Path, generation_log: Path | None = None, max_bytes: int = MAX_BYTES
    ) -> None:
        self.runs_dir = runs_dir
        self.generation_log = generation_log
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._truncated: set[str] = set()

    # ---- writing -------------------------------------------------------

    def folder(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def path(self, run_id: str) -> Path:
        return self.folder(run_id) / EVENTS_FILE

    def write(
        self,
        run_id: str,
        level: str,
        event: str,
        msg: str = "",
        *,
        product: str | None = None,
        stage: str | None = None,
        **extra: Any,
    ) -> None:
        """Append one event. Never raises: a log that breaks a run is worse than a gap."""
        record = {
            "ts": utc_now(),
            "level": level.upper(),
            "run_id": run_id,
            "product": product if product is not None else current_product.get(),
            "stage": stage,
            "event": event,
            "msg": msg,
            **extra,
        }
        try:
            line = json.dumps(redact_deep(record), ensure_ascii=False, default=str) + "\n"
            with self._lock:
                self.folder(run_id).mkdir(parents=True, exist_ok=True)
                path = self.path(run_id)
                size = path.stat().st_size if path.exists() else 0
                if size + len(line.encode("utf-8")) > self.max_bytes:
                    if run_id not in self._truncated:
                        self._truncated.add(run_id)
                        marker = {
                            "ts": utc_now(), "level": "WARNING", "run_id": run_id, "product": None,
                            "stage": None, "event": "truncated",
                            "msg": f"log reached {self.max_bytes} bytes; later events were dropped",
                        }
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(marker) + "\n")
                    return
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).exception("could not write a run log event")

    # ---- reading -------------------------------------------------------

    def exists(self, run_id: str) -> bool:
        return self.path(run_id).is_file()

    def read(
        self,
        run_id: str,
        *,
        after: int = 0,
        level: str | None = None,
        product: str | None = None,
        stage: str | None = None,
        q: str | None = None,
        limit: int = MAX_SHOWN,
    ) -> LogPage:
        """Events written since byte offset `after`, filtered, newest `limit` kept.

        Only complete lines are returned and counted in `next_offset`, so a line the
        writer is halfway through is picked up whole on the next call.
        """
        path = self.path(run_id)
        if not path.is_file():
            return LogPage([], after)
        with path.open("rb") as handle:
            handle.seek(max(after, 0))
            chunk = handle.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return LogPage([], after)
        complete = chunk[: end + 1]
        events = [event for raw in complete.splitlines() if (event := _parse(raw))]
        events = [e for e in events if matches(e, level=level, product=product, stage=stage, q=q)]
        hidden = max(len(events) - limit, 0) if limit else 0  # limit=0 means "all of it"
        return LogPage(events[-limit:] if limit else events, after + len(complete), hidden)

    def generation_events(
        self, products: Collection[str], start: str | None, end: str | None
    ) -> list[dict]:
        """Provider request records (Seedream, PixelVault) for these products in a time window."""
        if not self.generation_log or not self.generation_log.is_file():
            return []
        low = parse_time(start) if start else None
        high = parse_time(end) if end else None
        found: list[dict] = []
        with self.generation_log.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                entry = _parse(raw.encode("utf-8"))
                if not entry or entry.get("product") not in products:
                    continue
                moment = parse_time(entry.get("time", ""))
                if moment is None or (low and moment < low) or (high and moment > high):
                    continue
                found.append(_as_event(entry))
        return found

    def merged(
        self,
        run_id: str,
        products: Collection[str],
        start: str | None,
        end: str | None,
        **filters: Any,
    ) -> LogPage:
        """The run's own events plus the provider records from the same window, in time order."""
        page = self.read(run_id, limit=0, **filters)
        extra = [
            e for e in self.generation_events(products, start, end)
            if matches(e, **{k: v for k, v in filters.items() if k in ("level", "product", "stage", "q")})
        ]
        everything = sorted([*page.events, *extra], key=lambda e: e.get("ts", ""))
        hidden = max(len(everything) - MAX_SHOWN, 0)
        return LogPage(everything[-MAX_SHOWN:], page.next_offset, hidden)

    # ---- housekeeping --------------------------------------------------

    def prune(self, keep: int = KEEP_RUNS, protect: Iterable[str] = ()) -> list[str]:
        """Delete the oldest run folders beyond `keep`. Run ids sort by start time."""
        if not self.runs_dir.is_dir():
            return []
        folders = sorted(d for d in self.runs_dir.iterdir() if d.is_dir())
        protected = set(protect)
        removed = []
        for folder in folders[: max(len(folders) - keep, 0)]:
            if folder.name in protected:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            self._truncated.discard(folder.name)
            removed.append(folder.name)
        return removed


def _parse(raw: bytes) -> dict | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _as_event(entry: dict) -> dict:
    """A generation-log line in the run-event shape, so the viewer renders both alike."""
    failed = bool(entry.get("error"))
    detail = [
        f"{key}={entry[key]}"
        for key in ("shot", "file", "attempt", "status", "elapsed_s")
        if entry.get(key) is not None
    ]
    if failed:
        detail.append(str(entry["error"]))
    return {
        "ts": entry.get("time", ""),
        "level": "ERROR" if failed else "INFO",
        "product": entry.get("product"),
        "stage": entry.get("event"),
        "event": "provider_request",
        "msg": " ".join(detail),
        "source": "generation-log",
        "detail": {k: v for k, v in entry.items() if k not in ("time", "product")},
    }


def matches(
    event: dict,
    *,
    level: str | None = None,
    product: str | None = None,
    stage: str | None = None,
    q: str | None = None,
) -> bool:
    if level and level.upper() in LEVELS:
        if LEVELS.index(str(event.get("level", "INFO")).upper()) < LEVELS.index(level.upper()):
            return False
    if product and event.get("product") != product:
        return False
    if stage and event.get("stage") != stage:
        return False
    if q:
        haystack = " ".join(
            str(event.get(key) or "") for key in ("msg", "event", "error_type", "traceback", "stage")
        )
        if q.lower() not in haystack.lower():
            return False
    return True


class RunLogHandler(logging.Handler):
    """Sends standard `logging` records to the run that is current in this context.

    Whatever the code under a run logs (a warning from a library, `logger.exception`
    in a node) lands in that run's file with its product, without the code knowing
    runs exist. Records made outside any run are ignored here; the app log has them.
    """

    def __init__(self, logs: RunLogs) -> None:
        super().__init__(level=logging.INFO)
        self.logs = logs

    def emit(self, record: logging.LogRecord) -> None:
        run_id = current_run.get()
        if not run_id or record.name == __name__:
            return  # not in a run, or our own "could not write" complaint: logging it here would loop
        extra: dict[str, Any] = {"logger": record.name}
        if record.exc_info and record.exc_info[0] is not None:
            extra["error_type"] = record.exc_info[0].__name__
            extra["traceback"] = "".join(traceback_module.format_exception(*record.exc_info))
        self.logs.write(run_id, record.levelname, "log", record.getMessage(), **extra)


def install_run_handler(logs: RunLogs) -> RunLogHandler:
    """Attach one `RunLogHandler` to the root logger, replacing any earlier one."""
    root = logging.getLogger()
    for existing in [h for h in root.handlers if isinstance(h, RunLogHandler)]:
        root.removeHandler(existing)
    handler = RunLogHandler(logs)
    root.addHandler(handler)
    package_logger = logging.getLogger("product_catalog")
    if package_logger.level == logging.NOTSET:
        package_logger.setLevel(logging.INFO)  # records below the root's WARNING never reach handlers
    return handler


class RedactingFormatter(logging.Formatter):
    """For the app log file: format as usual, then mask secrets in the final text."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def tail_text(path: Path, max_lines: int = 500) -> list[str]:
    """The last `max_lines` lines of a text file, secrets masked. Empty if it is missing."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()[-max_lines:]
    return [redact(line.rstrip("\n")) for line in lines]
