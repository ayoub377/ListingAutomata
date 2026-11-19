"""Run history in SQLite: one row per run, one per product in it.

Stdlib only. The file lives at `output/runs.db` beside the rest of the output, and a
run that was in flight when the process died is marked `interrupted` on the next start
instead of staying "running" forever.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Run statuses. `preview` = uploaded and costed, waiting for the user to confirm.
PREVIEW, QUEUED, RUNNING = "preview", "queued", "running"
SUCCEEDED, PARTIAL, FAILED, INTERRUPTED = "succeeded", "partial", "failed", "interrupted"
ACTIVE = (QUEUED, RUNNING)

# Item statuses.
PLANNED, DONE, SKIPPED = "planned", "done", "skipped"  # plus RUNNING, FAILED, INTERRUPTED

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    shots TEXT NOT NULL,
    force INTEGER NOT NULL DEFAULT 0,
    batch_id TEXT,
    new_images INTEGER NOT NULL DEFAULT 0,
    est_cost REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS run_items (
    run_id TEXT NOT NULL REFERENCES runs(id),
    product TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error TEXT,
    new_images INTEGER NOT NULL DEFAULT 0,
    est_cost REAL NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    PRIMARY KEY (run_id, product)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunItem:
    run_id: str
    product: str
    title: str
    status: str
    error: str | None
    new_images: int
    est_cost: float
    started_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class Run:
    id: str
    filename: str
    status: str
    shots: list[str]
    force: bool
    batch_id: str | None
    new_images: int
    est_cost: float
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    counts: dict[str, int]  # item status -> how many

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()  # one connection shared by request threads and the runner task
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def _run(self, sql: str, args: tuple = ()) -> None:
        with self._lock, self._db:
            self._db.execute(sql, args)

    # ---- writes --------------------------------------------------------

    def create(self, run_id: str, filename: str, shots: list[str], force: bool) -> None:
        self._run(
            "INSERT INTO runs (id, filename, status, shots, force, created_at) VALUES (?,?,?,?,?,?)",
            (run_id, filename, PREVIEW, json.dumps(shots), int(force), now()),
        )

    def set_plan(self, run_id: str, items: list[tuple[str, str, str, int, float, str | None]]) -> None:
        """Replace a run's items with (product, title, status, new_images, est_cost, error) rows."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM run_items WHERE run_id = ?", (run_id,))
            self._db.executemany(
                "INSERT INTO run_items (run_id, product, title, status, new_images, est_cost, error)"
                " VALUES (?,?,?,?,?,?,?)",
                [(run_id, *item[:5], item[5]) for item in items],
            )
            planned = [item for item in items if item[2] == PLANNED]
            self._db.execute(
                "UPDATE runs SET new_images = ?, est_cost = ? WHERE id = ?",
                (sum(item[3] for item in planned), sum(item[4] for item in planned), run_id),
            )

    def mark_queued(self, run_id: str) -> None:
        self._run("UPDATE runs SET status = ?, error = NULL WHERE id = ?", (QUEUED, run_id))

    def mark_running(self, run_id: str, batch_id: str) -> None:
        self._run(
            "UPDATE runs SET status = ?, batch_id = ?, started_at = ? WHERE id = ?",
            (RUNNING, batch_id, now(), run_id),
        )

    def update_item(self, run_id: str, product: str, status: str, error: str | None = None) -> None:
        stamp = now()
        if status == RUNNING:
            self._run(
                "UPDATE run_items SET status = ?, started_at = ? WHERE run_id = ? AND product = ?",
                (status, stamp, run_id, product),
            )
        else:
            self._run(
                "UPDATE run_items SET status = ?, error = ?, finished_at = ? WHERE run_id = ? AND product = ?",
                (status, error, stamp, run_id, product),
            )

    def finish(self, run_id: str, status: str, error: str | None = None) -> None:
        self._run(
            "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, error, now(), run_id),
        )

    def recover_interrupted(self) -> int:
        """Mark runs left queued or running by a dead process. Returns how many."""
        stamp = now()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE run_items SET status = ?, finished_at = ? WHERE status IN (?, ?) AND run_id IN"
                " (SELECT id FROM runs WHERE status IN (?, ?))",
                (INTERRUPTED, stamp, RUNNING, PLANNED, QUEUED, RUNNING),
            )
            cursor = self._db.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE status IN (?, ?)",
                (INTERRUPTED, stamp, "The app stopped while this run was in progress.", QUEUED, RUNNING),
            )
            return cursor.rowcount

    # ---- reads ---------------------------------------------------------

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return self._build(row) if row else None

    def recent(self, limit: int = 50) -> list[Run]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [self._build(row) for row in rows]

    def items(self, run_id: str) -> list[RunItem]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM run_items WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [RunItem(**{key: row[key] for key in row.keys()}) for row in rows]

    def _build(self, row: sqlite3.Row) -> Run:
        counts = {
            status: count
            for status, count in self._db.execute(
                "SELECT status, COUNT(*) FROM run_items WHERE run_id = ? GROUP BY status", (row["id"],)
            )
        }
        return Run(
            id=row["id"], filename=row["filename"], status=row["status"],
            shots=json.loads(row["shots"]), force=bool(row["force"]), batch_id=row["batch_id"],
            new_images=row["new_images"], est_cost=row["est_cost"], error=row["error"],
            created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"],
            counts=counts,
        )
