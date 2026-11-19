"""Upload, run history, the runner and the log routes.

The workflow is a fake `RunWorkflow`, so none of this loads torch, calls a model, or
touches Meta or .env. Async code is driven with `asyncio.run` and, for HTTP, a
`TestClient` used as a context manager so background tasks outlive a single request.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from product_catalog import load_products_csv, product_name
from product_catalog.review.app import create_app
from product_catalog.review.catalog import ReviewSettings, discover
from product_catalog.review.publisher import PushResult
from product_catalog.runs import store as states
from product_catalog.runs.logs import (
    RunLogs,
    current_product,
    current_run,
    install_run_handler,
    tail_text,
)
from product_catalog.runs.redact import MASK, redact
from product_catalog.runs.runner import RunBusy, RunManager, RunRejected
from product_catalog.runs.store import RunStore
from product_catalog.runs.tabular import TabularError, normalize
from product_catalog.runs.workflow import PlanItem, RunPlan

HTMX = {"HX-Request": "true"}
SECRET = "pv-super-secret-1234"
GOOD_CSV = "product_folder,title,price\nphotos/chaise,Chaise Camping,90\nphotos/table,Table Pliante,250\n"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeWorkflow:
    all_shots = ["angle", "context"]

    def __init__(self, fail=(), unplanned=(), missing_keys=(), delay: float = 0) -> None:
        self.fail = set(fail)
        self.unplanned = set(unplanned)
        self.missing_keys = list(missing_keys)
        self.delay = delay
        self.ran: list[str] = []
        self.flushed = 0

    def missing(self) -> list[str]:
        return self.missing_keys

    def plan(self, csv_path: Path, shots: list[str], force: bool) -> RunPlan:
        rows = load_products_csv(csv_path).listings()
        items, missing = [], []
        for row in rows:
            name = product_name(row)
            if name in self.unplanned:
                missing.append(name)
            else:
                items.append(PlanItem(name, row.title or name, len(shots)))
        return RunPlan(items, missing, shots, 0.04, "fake-model")

    def new_batch_id(self) -> str:
        return "batch123"

    async def run_item(self, item: PlanItem, batch_id: str, on_stage) -> None:
        await asyncio.sleep(self.delay)
        on_stage("generate_image", 0.25)
        logging.getLogger("product_catalog.fake").warning("fake warning for %s", item.name)
        if item.name in self.fail:
            raise ValueError(f"provider rejected key {SECRET} for {item.name}")
        self.ran.append(item.name)

    def flush(self) -> None:
        self.flushed += 1


class FakePusher:
    def missing(self) -> list[str]:
        return []

    async def push(self, product) -> PushResult:
        return PushResult(True, "ok")


@pytest.fixture(autouse=True)
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXELVAULT_API_KEY", SECRET)


@pytest.fixture
def logs(tmp_path: Path):
    runs_logs = RunLogs(tmp_path / "output" / "runs", tmp_path / "output" / "generation-log.jsonl")
    handler = install_run_handler(runs_logs)
    package_logger = logging.getLogger("product_catalog")
    before = package_logger.level
    yield runs_logs
    logging.getLogger().removeHandler(handler)
    package_logger.setLevel(before)


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "output" / "runs.db")


def make_manager(store, logs, workflow=None, **kwargs) -> RunManager:
    return RunManager(store, logs, workflow or FakeWorkflow(), **kwargs)


def go(manager: RunManager, run_id: str):
    """Start a run and wait for it, inside one event loop."""

    async def main():
        manager.start(run_id)
        await manager.wait()

    asyncio.run(main())
    return manager.store.get(run_id)


# --------------------------------------------------------------------------
# tabular
# --------------------------------------------------------------------------


def test_semicolon_cp1252_csv_is_normalised_to_utf8_commas() -> None:
    raw = "product_folder;title;price\nphotos/chaise;Chaise Pliée;90\n".encode("cp1252")
    out = normalize("produits.csv", raw).decode("utf-8")
    assert out.splitlines() == ["product_folder,title,price", "photos/chaise,Chaise Pliée,90"]


def test_quoted_commas_survive_normalisation() -> None:
    out = normalize("p.csv", b'product_folder,title,price\nx,"Chaise, pliable",90\n').decode()
    assert '"Chaise, pliable"' in out


def test_xlsx_floats_become_ints_and_blank_columns_are_dropped() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["product_folder", "title", "price", None])
    sheet.append(["photos/chaise", "Chaise", 4300.0, None])
    sheet.append([None, None, None, None])  # blank rows are ignored
    sheet.append(["photos/table", "Table", 12.5, "stray"])
    buffer = io.BytesIO()
    book.save(buffer)

    out = normalize("produits.xlsx", buffer.getvalue()).decode().splitlines()

    assert out == ["product_folder,title,price", "photos/chaise,Chaise,4300", "photos/table,Table,12.5"]


@pytest.mark.parametrize(
    "name, data, fragment",
    [
        ("a.xls", b"x", "Save it as .xlsx"),
        ("a.txt", b"x", "Upload a .csv or .xlsx"),
        ("a.csv", b"", "empty"),
        ("a.csv", b"product_folder,price\n", "header row and at least one"),
        ("a.csv", b"x" * (5 * 1024 * 1024 + 1), "over 5 MB"),
        ("a.xlsx", b"not a workbook", "Could not open"),
    ],
    ids=["xls", "txt", "empty", "header-only", "too-big", "bad-xlsx"],  # a 5 MB id overflows Windows' env var limit
)
def test_bad_uploads_are_rejected_with_a_message(name: str, data: bytes, fragment: str) -> None:
    with pytest.raises(TabularError, match=fragment):
        normalize(name, data)


# --------------------------------------------------------------------------
# redaction
# --------------------------------------------------------------------------


def test_redact_masks_env_values_and_secret_shapes() -> None:
    text = f"key {SECRET}; Authorization: Bearer abcdef123456; GET /x?access_token=EAAB12345&v=1; sk-or-v1-0123456789abcdef0123"
    out = redact(text)
    assert SECRET not in out
    assert "abcdef123456" not in out and "EAAB12345" not in out and "0123456789abcdef" not in out
    assert out.count(MASK) == 4
    assert "v=1" in out  # only the secret part of a URL is masked


def test_redact_ignores_short_env_values() -> None:
    assert redact("a token of 123", {"PIXELVAULT_API_KEY": "123"}) == "a token of 123"


# --------------------------------------------------------------------------
# run logs
# --------------------------------------------------------------------------


def test_write_then_read_by_offset_and_filters(logs: RunLogs) -> None:
    logs.write("r1", "INFO", "product_start", "begin", product="chaise", stage="product")
    logs.write("r1", "ERROR", "product_failed", "boom", product="table", stage="product", traceback="Traceback...")
    first = logs.read("r1")
    assert [e["event"] for e in first.events] == ["product_start", "product_failed"]

    logs.write("r1", "INFO", "stage_done", "image done", product="chaise", stage="generate_image")
    later = logs.read("r1", after=first.next_offset)
    assert [e["event"] for e in later.events] == ["stage_done"]

    assert [e["product"] for e in logs.read("r1", level="error").events] == ["table"]
    assert [e["event"] for e in logs.read("r1", product="chaise").events] == ["product_start", "stage_done"]
    assert [e["event"] for e in logs.read("r1", stage="generate_image").events] == ["stage_done"]
    assert [e["event"] for e in logs.read("r1", q="TRACEBACK").events] == ["product_failed"]


def test_read_skips_a_line_the_writer_has_not_finished(logs: RunLogs) -> None:
    logs.write("r1", "INFO", "a", "one")
    with logs.path("r1").open("a", encoding="utf-8") as handle:
        handle.write('{"event": "half')  # no newline yet
    page = logs.read("r1")
    assert [e["event"] for e in page.events] == ["a"]
    with logs.path("r1").open("a", encoding="utf-8") as handle:
        handle.write('-written", "level": "INFO"}\n')
    assert [e["event"] for e in logs.read("r1", after=page.next_offset).events] == ["half-written"]


def test_write_redacts_secrets_in_every_field(logs: RunLogs) -> None:
    logs.write("r1", "ERROR", "x", f"bad {SECRET}", traceback=f"File ... {SECRET}", extra={"k": SECRET})
    assert SECRET not in logs.path("r1").read_text(encoding="utf-8")


def test_log_is_capped_with_one_marker(tmp_path: Path) -> None:
    small = RunLogs(tmp_path / "runs", max_bytes=600)
    for index in range(30):
        small.write("r1", "INFO", "tick", f"event {index}")
    events = small.read("r1").events
    assert sum(e["event"] == "truncated" for e in events) == 1
    assert events[-1]["event"] == "truncated"
    assert small.path("r1").stat().st_size < 1200


def test_prune_keeps_the_newest_and_protected_runs(logs: RunLogs) -> None:
    for run_id in ("20260101-000000-aaaa", "20260102-000000-bbbb", "20260103-000000-cccc", "20260104-000000-dddd"):
        logs.write(run_id, "INFO", "x")
    removed = logs.prune(keep=2, protect=["20260101-000000-aaaa"])
    assert removed == ["20260102-000000-bbbb"]
    assert sorted(p.name for p in logs.runs_dir.iterdir()) == [
        "20260101-000000-aaaa", "20260103-000000-cccc", "20260104-000000-dddd",
    ]


def test_generation_records_are_merged_by_product_and_time_window(logs: RunLogs) -> None:
    def line(when: str, product: str, **extra) -> str:
        return json.dumps({"time": when, "event": "generate", "product": product, "shot": "angle", **extra})

    logs.generation_log.parent.mkdir(parents=True, exist_ok=True)
    logs.generation_log.write_text(
        "\n".join(
            [
                line("2026-11-19T10:00:05+00:00", "chaise", status=200, elapsed_s=12.5),
                line("2026-11-19T10:00:06+00:00", "chaise", error="openrouter 500: oops"),
                line("2026-11-19T10:00:07+00:00", "other-product"),
                line("2026-11-19T09:00:00+00:00", "chaise"),  # before the run
                "not json",
            ]
        ),
        encoding="utf-8",
    )
    page = logs.merged("r1", ["chaise"], "2026-11-19T10:00:00+00:00", "2026-11-19T10:05:00+00:00")
    assert [e["level"] for e in page.events] == ["INFO", "ERROR"]
    assert all(e["source"] == "generation-log" for e in page.events)
    assert "openrouter 500" in page.events[1]["msg"]
    assert [e["event"] for e in logs.merged("r1", ["chaise"], None, None, level="ERROR").events] == ["provider_request"]


def test_handler_routes_records_to_the_current_run_with_traceback(logs: RunLogs) -> None:
    logger = logging.getLogger("product_catalog.somewhere")
    logger.info("outside any run")  # ignored: no run is current

    async def main() -> None:
        current_run.set("r1")
        current_product.set("chaise")
        try:
            raise RuntimeError(f"nope {SECRET}")
        except RuntimeError:
            logger.exception("node failed")

    asyncio.run(main())

    events = logs.read("r1").events
    assert [e["msg"] for e in events] == ["node failed"]
    assert events[0]["product"] == "chaise"
    assert events[0]["error_type"] == "RuntimeError"
    assert "Traceback" in events[0]["traceback"] and SECRET not in events[0]["traceback"]


def test_tail_text_masks_secrets(tmp_path: Path) -> None:
    path = tmp_path / "app.log"
    path.write_text(f"one\nkey={SECRET}\nthree\n", encoding="utf-8")
    assert tail_text(path, 2) == [f"key={MASK}", "three"]
    assert tail_text(tmp_path / "missing.log") == []


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


def test_store_lifecycle_and_interrupted_recovery(store: RunStore) -> None:
    store.create("r1", "p.csv", ["angle"], force=True)
    store.set_plan("r1", [("a", "A", states.PLANNED, 2, 0.08, None), ("b", "", states.SKIPPED, 0, 0.0, "no photos")])
    run = store.get("r1")
    assert (run.status, run.new_images, run.est_cost, run.force, run.shots) == (states.PREVIEW, 2, 0.08, True, ["angle"])
    assert run.counts == {states.PLANNED: 1, states.SKIPPED: 1}

    store.mark_running("r1", "batch")
    store.update_item("r1", "a", states.RUNNING)
    assert store.recover_interrupted() == 1

    run = store.get("r1")
    assert run.status == states.INTERRUPTED and "stopped" in run.error
    by_product = {i.product: i.status for i in store.items("r1")}
    assert by_product == {"a": states.INTERRUPTED, "b": states.SKIPPED}  # a skipped item stays skipped


def test_recovery_leaves_finished_runs_alone(store: RunStore) -> None:
    store.create("r1", "p.csv", ["angle"], force=False)
    store.set_plan("r1", [("a", "A", states.PLANNED, 1, 0.04, None)])
    store.update_item("r1", "a", states.DONE)
    store.finish("r1", states.SUCCEEDED)
    assert store.recover_interrupted() == 0
    assert store.get("r1").status == states.SUCCEEDED


def test_recent_lists_newest_first(store: RunStore) -> None:
    for run_id in ("20260101-000000-aaaa", "20260102-000000-bbbb"):
        store.create(run_id, "p.csv", [], force=False)
    assert [r.id for r in store.recent()] == ["20260102-000000-bbbb", "20260101-000000-aaaa"]


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def test_upload_creates_a_costed_preview_and_starts_nothing(store: RunStore, logs: RunLogs) -> None:
    workflow = FakeWorkflow(unplanned={"table-pliante"})
    manager = make_manager(store, logs, workflow)

    run = manager.upload("produits.csv", GOOD_CSV.encode(), [], force=False)

    assert run.status == states.PREVIEW and manager.active is None
    assert run.shots == ["angle", "context"]
    assert (run.new_images, run.est_cost) == (2, 0.08)  # one planned product x two shots
    assert {i.product: i.status for i in store.items(run.id)} == {
        "chaise-camping": states.PLANNED, "table-pliante": states.SKIPPED,
    }
    assert manager.csv_path(run.id).is_file() and workflow.ran == []


def test_invalid_csv_is_rejected_with_line_numbers_and_leaves_nothing_behind(store: RunStore, logs: RunLogs) -> None:
    manager = make_manager(store, logs)
    bad = b"product_folder,title,price\nphotos/a,A,90\nphotos/b,B,not-a-price\n"

    with pytest.raises(RunRejected, match=r"\.csv:3: price"):
        manager.upload("bad.csv", bad, [], force=False)

    assert store.recent() == []
    assert not manager.uploads_dir.exists() or not any(manager.uploads_dir.iterdir())


def test_unknown_shot_and_bad_extension_are_rejected(store: RunStore, logs: RunLogs) -> None:
    manager = make_manager(store, logs)
    with pytest.raises(RunRejected, match="Unknown shot: banner"):
        manager.upload("p.csv", GOOD_CSV.encode(), ["banner"], force=False)
    with pytest.raises(RunRejected, match=r"\.csv or \.xlsx"):
        manager.upload("p.pdf", b"x", [], force=False)


def test_a_run_succeeds_and_logs_every_step(store: RunStore, logs: RunLogs) -> None:
    workflow = FakeWorkflow()
    manager = make_manager(store, logs, workflow)
    run = manager.upload("produits.csv", GOOD_CSV.encode(), ["angle"], force=False)

    done = go(manager, run.id)

    assert done.status == states.SUCCEEDED and done.batch_id == "batch123" and manager.active is None
    assert sorted(workflow.ran) == ["chaise-camping", "table-pliante"] and workflow.flushed == 1
    assert {i.status for i in store.items(run.id)} == {states.DONE}
    events = logs.read(run.id).events
    names = [e["event"] for e in events]
    assert names[:2] == ["uploaded", "run_start"] and names[-1] == "run_end"
    assert names.count("product_start") == names.count("product_done") == 2
    stage = next(e for e in events if e["event"] == "stage_done")
    assert (stage["stage"], stage["duration_ms"]) == ("generate_image", 250)
    # stdlib logging inside the workflow lands in the same file, tagged with the product
    warnings = [e for e in events if e["event"] == "log" and e["level"] == "WARNING"]
    assert sorted(e["product"] for e in warnings) == ["chaise-camping", "table-pliante"]


def test_one_failing_product_does_not_stop_the_rest_and_keeps_a_redacted_traceback(store: RunStore, logs: RunLogs) -> None:
    manager = make_manager(store, logs, FakeWorkflow(fail={"table-pliante"}, unplanned=set()))
    run = manager.upload("produits.csv", GOOD_CSV.encode(), [], force=False)

    done = go(manager, run.id)

    assert done.status == states.PARTIAL
    items = {i.product: i for i in store.items(run.id)}
    assert items["chaise-camping"].status == states.DONE
    assert items["table-pliante"].status == states.FAILED
    assert SECRET not in (items["table-pliante"].error or "")
    failure = next(e for e in logs.read(run.id).events if e["event"] == "product_failed")
    assert failure["product"] == "table-pliante" and failure["error_type"] == "ValueError"
    assert "Traceback" in failure["traceback"] and SECRET not in failure["traceback"]
    assert SECRET not in logs.path(run.id).read_text(encoding="utf-8")


def test_when_every_product_fails_the_run_failed(store: RunStore, logs: RunLogs) -> None:
    manager = make_manager(store, logs, FakeWorkflow(fail={"chaise-camping", "table-pliante"}))
    run = manager.upload("produits.csv", GOOD_CSV.encode(), [], force=False)
    assert go(manager, run.id).status == states.FAILED


def test_start_rules(store: RunStore, logs: RunLogs) -> None:
    manager = make_manager(store, logs, FakeWorkflow(delay=0.05))
    first = manager.upload("a.csv", GOOD_CSV.encode(), [], force=False)
    second = manager.upload("b.csv", GOOD_CSV.encode(), [], force=False)

    async def main() -> None:
        manager.start(first.id)
        assert manager.active == first.id
        with pytest.raises(RunBusy, match=first.id):
            manager.start(second.id)  # only one run at a time
        with pytest.raises(RunRejected, match="already"):
            manager.start(first.id)  # and a run starts once
        await manager.wait()

    asyncio.run(main())
    assert store.get(first.id).status == states.SUCCEEDED and store.get(second.id).status == states.PREVIEW
    assert go(manager, second.id).status == states.SUCCEEDED  # free again once the first ended


def test_start_refuses_without_keys_and_without_runnable_products(store: RunStore, logs: RunLogs) -> None:
    run = make_manager(store, logs).upload("a.csv", GOOD_CSV.encode(), [], force=False)
    no_key = make_manager(store, logs, FakeWorkflow(missing_keys=["OPEN_ROUTER_API_KEY"]))

    async def attempt(manager, run_id):
        manager.start(run_id)

    with pytest.raises(RunRejected, match="OPEN_ROUTER_API_KEY"):
        asyncio.run(attempt(no_key, run.id))
    assert store.get(run.id).status == states.PREVIEW

    nothing = make_manager(store, logs, FakeWorkflow(unplanned={"chaise-camping", "table-pliante"}))
    with pytest.raises(RunRejected, match="Nothing to run"):
        asyncio.run(attempt(nothing, run.id))


def test_old_logs_are_pruned_but_uploaded_csvs_are_kept(store: RunStore, logs: RunLogs) -> None:
    manager = make_manager(store, logs)
    run = manager.upload("a.csv", GOOD_CSV.encode(), [], force=False)
    go(manager, run.id)
    logs.prune(keep=0)
    assert not logs.exists(run.id)
    assert manager.csv_path(run.id).is_file()  # the board still needs it


def test_board_csvs_lists_only_runs_that_produced_output_oldest_first(store: RunStore, logs: RunLogs) -> None:
    manager = make_manager(store, logs, FakeWorkflow(fail={"chaise-camping", "table-pliante"}))
    failed = manager.upload("a.csv", GOOD_CSV.encode(), [], force=False)
    go(manager, failed.id)
    manager.workflow = FakeWorkflow()
    ok_one = manager.upload("b.csv", GOOD_CSV.encode(), [], force=False)
    go(manager, ok_one.id)
    ok_two = manager.upload("c.csv", GOOD_CSV.encode(), [], force=False)
    go(manager, ok_two.id)
    pending = manager.upload("d.csv", GOOD_CSV.encode(), [], force=False)

    assert manager.board_csvs() == [manager.csv_path(ok_one.id), manager.csv_path(ok_two.id)]
    assert pending.status == states.PREVIEW


def test_a_new_manager_marks_runs_the_dead_process_left_running(store: RunStore, logs: RunLogs) -> None:
    store.create("20260101-000000-aaaa", "p.csv", [], force=False)
    store.mark_running("20260101-000000-aaaa", "b")
    make_manager(store, logs)
    assert store.get("20260101-000000-aaaa").status == states.INTERRUPTED


# --------------------------------------------------------------------------
# the review board reads uploaded CSVs too
# --------------------------------------------------------------------------


def test_board_discovers_products_priced_by_an_uploaded_csv(tmp_path: Path) -> None:
    uploaded = tmp_path / "uploads" / "r1.csv"
    uploaded.parent.mkdir()
    uploaded.write_text("product_folder,title,price\nphotos/chaise,Chaise Camping,120\n", encoding="utf-8")
    folder = tmp_path / "output" / "chaise-camping"
    folder.mkdir(parents=True)
    (folder / "product_angle.jpg").write_bytes(b"\xff\xd8\xff x")
    (folder / "description.json").write_text(
        json.dumps({"title": "Chaise", "fr": "Une chaise.", "images": ["product_angle.jpg"]}), encoding="utf-8"
    )

    # No products.csv at all: a project that only works from uploads is fine.
    settings = ReviewSettings.from_project(tmp_path, extra_csvs=lambda: [uploaded])
    found = discover(settings)
    assert [p.name for p in found.products] == ["chaise-camping"] and found.problems == []
    assert str(found.products[0].price) == "120"

    # Without the extra file there is nothing to price it with, exactly as before.
    plain = discover(ReviewSettings.from_project(tmp_path))
    assert plain.products == [] and plain.problems


def test_a_later_csv_replaces_an_earlier_row_of_the_same_product(tmp_path: Path) -> None:
    (tmp_path / "products.csv").write_text("product_folder,title,price\nphotos/chaise,Chaise Camping,90\n", encoding="utf-8")
    newer = tmp_path / "new.csv"
    newer.write_text("product_folder,title,price\nphotos/chaise,Chaise Camping,150\n", encoding="utf-8")
    folder = tmp_path / "output" / "chaise-camping"
    folder.mkdir(parents=True)
    (folder / "product_angle.jpg").write_bytes(b"\xff\xd8\xff x")
    (folder / "description.json").write_text(
        json.dumps({"fr": "x", "images": ["product_angle.jpg"]}), encoding="utf-8"
    )
    found = discover(ReviewSettings.from_project(tmp_path, extra_csvs=lambda: [newer]))
    assert [str(p.price) for p in found.products] == ["150"]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


@pytest.fixture
def app_parts(tmp_path: Path, store: RunStore, logs: RunLogs):
    manager = make_manager(store, logs)
    app_log = tmp_path / "output" / "app.log"
    app_log.write_text(f"startup ok\nkey={SECRET}\n", encoding="utf-8")
    settings = ReviewSettings.from_project(tmp_path, extra_csvs=manager.board_csvs)
    return create_app(settings, FakePusher(), None, manager, app_log), manager


def upload(client: TestClient, content: str = GOOD_CSV, name: str = "produits.csv", **data):
    return client.post("/runs", files={"file": (name, content.encode(), "text/csv")}, data=data, headers=HTMX)


def wait_until_finished(client: TestClient, run_id: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}/panel").text
        if 'hx-trigger="every 2s"' not in body:  # the panel stops polling once the run is over
            return body
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_runs_pages_render_and_posts_need_the_htmx_header(app_parts) -> None:
    app, _ = app_parts
    with TestClient(app) as client:
        page = client.get("/runs")
        assert page.status_code == 200 and "Upload and preview" in page.text and "No runs yet" in page.text
        assert client.post("/runs", files={"file": ("p.csv", GOOD_CSV.encode())}).status_code == 403
        assert client.post("/runs/20260101-000000-aaaa/start").status_code == 403


def test_upload_redirects_to_the_preview_and_a_bad_file_explains_itself(app_parts) -> None:
    app, manager = app_parts
    with TestClient(app) as client:
        good = upload(client)
        assert good.status_code == 204
        run_id = good.headers["HX-Redirect"].rsplit("/", 1)[-1]
        detail = client.get(f"/runs/{run_id}")
        assert "Start run" in detail.text and "$0.16" in detail.text and "chaise-camping" in detail.text

        bad = upload(client, "product_folder,title,price\nphotos/a,A,nope\n")
        assert bad.status_code == 200 and "price" in bad.text and "alert" in bad.text
        assert upload(client, "x", name="p.pdf").text.count("Upload a .csv or .xlsx") == 1
        assert len(manager.store.recent()) == 1  # rejected uploads leave no run behind


def test_start_runs_in_the_background_and_the_panel_reports_progress(app_parts) -> None:
    app, manager = app_parts
    with TestClient(app) as client:
        run_id = upload(client).headers["HX-Redirect"].rsplit("/", 1)[-1]
        started = client.post(f"/runs/{run_id}/start", headers=HTMX)
        assert started.status_code == 200
        body = wait_until_finished(client, run_id)
        assert "succeeded" in body and "Review and push on the board" in body and "batch123" in body
        table = client.get("/runs/table").text
        assert "produits.csv" in table and "succeeded" in table
        again = client.post(f"/runs/{run_id}/start", headers=HTMX)
        assert "already succeeded" in again.text


def test_log_routes_viewer_json_tail_and_download(app_parts) -> None:
    app, manager = app_parts
    manager.workflow = FakeWorkflow(fail={"table-pliante"})
    with TestClient(app) as client:
        run_id = upload(client).headers["HX-Redirect"].rsplit("/", 1)[-1]
        client.post(f"/runs/{run_id}/start", headers=HTMX)
        wait_until_finished(client, run_id)

        page = client.get(f"/runs/{run_id}/logs")
        assert page.status_code == 200 and "product_failed" in page.text and "traceback" in page.text
        assert SECRET not in page.text

        data = client.get(f"/runs/{run_id}/logs", params={"format": "json", "level": "error"}).json()
        assert data["status"] == states.PARTIAL
        assert {e["event"] for e in data["events"]} == {"product_failed"}

        only_table = client.get(f"/runs/{run_id}/logs", params={"format": "json", "product": "table-pliante"}).json()
        assert {e["product"] for e in only_table["events"]} == {"table-pliante"}

        offset = client.get(f"/runs/{run_id}/logs", params={"format": "json"}).json()["next_offset"]
        tail = client.get(f"/runs/{run_id}/logs", params={"after": offset}, headers=HTMX)
        assert tail.status_code == 200 and "product_failed" not in tail.text  # nothing new since that offset

        download = client.get(f"/runs/{run_id}/logs.jsonl")
        assert download.status_code == 200 and download.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(line) for line in download.text.splitlines()]
        assert lines[0]["event"] == "uploaded" and lines[-1]["event"] == "run_end"
        assert SECRET not in download.text


def test_tail_of_a_live_run_keeps_polling_and_a_finished_one_stops(app_parts) -> None:
    app, manager = app_parts
    manager.workflow = FakeWorkflow(delay=0.3)
    with TestClient(app) as client:
        run_id = upload(client).headers["HX-Redirect"].rsplit("/", 1)[-1]
        client.post(f"/runs/{run_id}/start", headers=HTMX)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not manager.logs.exists(run_id):
            time.sleep(0.01)
        live = client.get(f"/runs/{run_id}/logs")
        assert 'id="log-tail"' in live.text and "after=" in live.text
        wait_until_finished(client, run_id)
        assert 'id="log-tail"' not in client.get(f"/runs/{run_id}/logs").text


def test_unknown_or_malformed_run_ids_are_404(app_parts) -> None:
    app, _ = app_parts
    with TestClient(app) as client:
        for path in (
            "/runs/20260101-000000-aaaa",
            "/runs/not-a-run",
            "/runs/..%2f..%2fsecrets/logs.jsonl",
            "/runs/20260101-000000-aaaa/logs",
            "/runs/20260101-000000-aaaa/logs.jsonl",
        ):
            assert client.get(path).status_code == 404, path


def test_app_log_route_masks_secrets(app_parts) -> None:
    app, _ = app_parts
    with TestClient(app) as client:
        page = client.get("/logs")
        assert page.status_code == 200 and "startup ok" in page.text
        assert SECRET not in page.text and MASK in page.text


def test_the_board_still_works_without_runs_wired_in(tmp_path: Path) -> None:
    (tmp_path / "products.csv").write_text(GOOD_CSV, encoding="utf-8")
    with TestClient(create_app(ReviewSettings.from_project(tmp_path), FakePusher())) as client:
        assert client.get("/").status_code == 200
        assert client.get("/runs").status_code == 404
