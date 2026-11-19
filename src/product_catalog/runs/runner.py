"""Starting a run and carrying it through: upload, cost preview, confirm, execute.

One run is active at a time. The workflow writes to one shared `output/` namespace
(product names come from titles), so two runs at once could overwrite each other's files.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

from product_catalog.models import load_products_csv
from product_catalog.runs import store as states
from product_catalog.runs.logs import RunLogs, current_product, current_run
from product_catalog.runs.redact import redact
from product_catalog.runs.store import Run, RunStore
from product_catalog.runs.tabular import TabularError, normalize
from product_catalog.runs.workflow import PlanItem, RunPlan, RunWorkflow

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4
CSV_NAME = "products.csv"
MAX_ERROR = 500  # characters of an error kept on the run record; the log has the full traceback


class RunRejected(ValueError):
    """The request cannot be accepted. The message is shown to the user as is."""


class RunBusy(RunRejected):
    """Another run is in progress."""


def new_run_id() -> str:
    """Sorts by start time, which is what log pruning and the history list rely on.

    The last four hex digits are the sub-second part (16 microsecond ticks, at most
    62500 of 65536), so two runs in the same second still sort in the order they began.
    """
    now = datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{now.microsecond // 16:04x}"


def plan_rows(plan: RunPlan, cost_per_image: float | None = None) -> list[tuple]:
    """A plan as the rows `RunStore.set_plan` takes: planned items, then skipped ones."""
    price = plan.cost_per_image if cost_per_image is None else cost_per_image
    rows: list[tuple] = [
        (item.name, item.title, states.PLANNED, item.new_images, round(item.new_images * price, 2), item.note or None)
        for item in plan.items
    ]
    rows += [
        (name, "", states.SKIPPED, 0, 0.0, "no cutouts and its photo folder was not found")
        for name in plan.missing
    ]
    return rows


class RunManager:
    def __init__(
        self,
        store: RunStore,
        logs: RunLogs,
        workflow: RunWorkflow,
        concurrency: int = DEFAULT_CONCURRENCY,
        uploads_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.logs = logs
        self.workflow = workflow
        self.concurrency = concurrency
        # Uploaded files live apart from the logs: pruning old logs must never delete a CSV
        # the review board still needs to price a generated product.
        self.uploads_dir = uploads_dir or logs.runs_dir.parent / "uploads"
        self.active: str | None = None
        self._tasks: set[asyncio.Task] = set()
        if recovered := store.recover_interrupted():
            logger.warning("marked %d run(s) interrupted: the app stopped while they were running", recovered)

    def csv_path(self, run_id: str) -> Path:
        return self.uploads_dir / run_id / CSV_NAME

    def board_csvs(self) -> list[Path]:
        """Uploaded CSVs of runs that produced output, oldest first, for the review board.

        The board prices a generated product from its CSV row, so it has to read the
        files that runs were made from, not only the project's own products.csv.
        """
        runs = [r for r in self.store.recent(limit=1000) if r.status in (states.SUCCEEDED, states.PARTIAL)]
        paths = [self.csv_path(run.id) for run in reversed(runs)]
        return [path for path in paths if path.is_file()]

    # ---- upload and preview -------------------------------------------

    def upload(self, filename: str, data: bytes, shots: list[str], force: bool) -> Run:
        """Validate an uploaded CSV/Excel file and create a run waiting for confirmation."""
        shots = shots or list(self.workflow.all_shots)
        if unknown := [shot for shot in shots if shot not in self.workflow.all_shots]:
            raise RunRejected(f"Unknown shot: {', '.join(unknown)}.")
        try:
            csv_bytes = normalize(filename, data)
        except TabularError as error:
            raise RunRejected(str(error)) from error

        run_id = new_run_id()
        folder = self.uploads_dir / run_id
        folder.mkdir(parents=True, exist_ok=True)
        try:
            (folder / ("source" + Path(filename).suffix.lower())).write_bytes(data)
            csv_path = self.csv_path(run_id)
            csv_path.write_bytes(csv_bytes)
            try:
                load_products_csv(csv_path)
            except ValueError as error:  # every bad row, with its line number
                raise RunRejected(str(error)) from error
            try:
                plan = self.workflow.plan(csv_path, shots, force)
            except ValueError as error:
                raise RunRejected(str(error)) from error
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

        display = Path(filename).name[:100] or "upload"
        self.store.create(run_id, display, shots, force)
        self.store.set_plan(run_id, plan_rows(plan))
        self.logs.write(
            run_id, "INFO", "uploaded", f"{display}: {len(plan.items)} product(s) to run, "
            f"{len(plan.missing)} skipped, ~${plan.est_cost:.2f} for {plan.new_images} new image(s)",
            products=len(plan.items), skipped=plan.missing, new_images=plan.new_images,
        )
        run = self.store.get(run_id)
        assert run is not None
        return run

    # ---- start ---------------------------------------------------------

    def start(self, run_id: str) -> Run:
        """Confirm a previewed run and begin executing it in the background."""
        run = self.store.get(run_id)
        if run is None:
            raise RunRejected("Unknown run.")
        if run.status != states.PREVIEW:
            raise RunRejected(f"This run is already {run.status}. Upload the file again to run it again.")
        if self.active:
            raise RunBusy(f"Run {self.active} is still going. Wait for it to finish.")
        if missing := self.workflow.missing():
            raise RunRejected(f"Set {', '.join(missing)} in .env, then restart the app.")
        csv_path = self.csv_path(run_id)
        if not csv_path.is_file():
            raise RunRejected("The uploaded file was cleaned up. Upload it again.")
        try:
            plan = self.workflow.plan(csv_path, run.shots, run.force)  # re-plan: files may have changed
        except ValueError as error:
            raise RunRejected(str(error)) from error
        if not plan.items:
            raise RunRejected("Nothing to run: no product has photos or cutouts.")

        self.store.set_plan(run_id, plan_rows(plan))
        self.store.mark_queued(run_id)
        self.active = run_id  # set before the task exists, so a second click cannot slip in
        task = asyncio.create_task(self._execute(run_id, plan))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        refreshed = self.store.get(run_id)
        assert refreshed is not None
        return refreshed

    async def wait(self) -> None:
        """Block until the running task, if any, is done. For tests and shutdown."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ---- execution -----------------------------------------------------

    async def _execute(self, run_id: str, plan: RunPlan) -> None:
        current_run.set(run_id)  # child tasks inherit it, so stdlib logging lands in this run's file
        write = self.logs.write
        batch_id = self.workflow.new_batch_id()
        self.store.mark_running(run_id, batch_id)
        write(
            run_id, "INFO", "run_start",
            f"{len(plan.items)} product(s), {plan.new_images} new image(s), ~${plan.est_cost:.2f} via {plan.model}",
            batch_id=batch_id, shots=plan.shots, concurrency=self.concurrency,
        )
        for name in plan.missing:
            write(run_id, "WARNING", "product_skipped", "no cutouts and its photo folder was not found", product=name)

        limit = asyncio.Semaphore(self.concurrency)

        async def one(item: PlanItem) -> bool:
            async with limit:
                current_product.set(item.name)
                started = time.monotonic()
                self.store.update_item(run_id, item.name, states.RUNNING)
                write(run_id, "INFO", "product_start", item.title, stage="product")

                def on_stage(node: str, seconds: float) -> None:
                    write(run_id, "INFO", "stage_done", f"{node} finished", stage=node,
                          duration_ms=int(seconds * 1000))

                try:
                    await self.workflow.run_item(item, batch_id, on_stage)
                except Exception as error:  # noqa: BLE001 - one bad product must not stop the rest
                    # Masked here too: this text goes into the database and onto the page, not only the log.
                    message = redact(str(error) or type(error).__name__)
                    write(run_id, "ERROR", "product_failed", message, stage="product",
                          error_type=type(error).__name__, traceback=traceback.format_exc(),
                          duration_ms=int((time.monotonic() - started) * 1000))
                    self.store.update_item(run_id, item.name, states.FAILED, message[:MAX_ERROR])
                    return False
                write(run_id, "INFO", "product_done", item.title, stage="product",
                      duration_ms=int((time.monotonic() - started) * 1000))
                self.store.update_item(run_id, item.name, states.DONE)
                return True

        try:
            results = await asyncio.gather(*(one(item) for item in plan.items))
        except asyncio.CancelledError:
            write(run_id, "ERROR", "run_cancelled", "the app is shutting down")
            self.store.recover_interrupted()
            self._finish_up(run_id)
            raise
        except Exception as error:  # noqa: BLE001 - a bug in the runner itself
            write(run_id, "ERROR", "run_crashed", str(error), error_type=type(error).__name__,
                  traceback=traceback.format_exc())
            self.store.finish(run_id, states.FAILED, redact(f"{type(error).__name__}: {error}")[:MAX_ERROR])
            self._finish_up(run_id)
            return

        succeeded = sum(results)
        failed = len(results) - succeeded
        status = states.SUCCEEDED if not failed else states.FAILED if not succeeded else states.PARTIAL
        write(run_id, "INFO" if not failed else "WARNING", "run_end",
              f"{succeeded} done, {failed} failed, {len(plan.missing)} skipped",
              succeeded=succeeded, failed=failed, skipped=len(plan.missing), status=status)
        self.store.finish(run_id, status)
        try:
            await asyncio.to_thread(self.workflow.flush)  # traces upload in the background
        except Exception:  # noqa: BLE001
            logger.exception("flushing traces after %s failed", run_id)
        self._finish_up(run_id)

    def _finish_up(self, run_id: str) -> None:
        self.active = None
        self.logs.prune(protect=[run_id])
