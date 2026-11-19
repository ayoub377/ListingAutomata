"""The review board: look at what was generated, fix it, approve it, push it to Meta.

One page, rendered server-side and updated by HTMX. Every action swaps the whole
`#board` section, so the counts, filters and push button never go stale.

Rules that keep the human in the loop:
  * only an approved product can be pushed, and the server checks that itself;
  * an approval belongs to one version of the content (see store.py), so editing or
    regenerating a product sends it back to pending;
  * a product that is being regenerated cannot be approved, edited or pushed until the
    new version has landed, because nobody has seen it yet.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from product_catalog.listings import MetaListing
from product_catalog.review.catalog import ReviewProduct, ReviewSettings, discover
from product_catalog.review.problems import DESCRIPTION_PROBLEMS, IMAGE_PROBLEMS, compose_note
from product_catalog.review.publisher import Pusher
from product_catalog.review.regenerator import Regenerator
from product_catalog.review.store import Regeneration, Status, now, save_review
from product_catalog.runs.routes import register_run_routes
from product_catalog.runs.runner import RunManager

logger = logging.getLogger(__name__)

VIEWS = ("all", "pending", "approved", "rejected", "pushed")
TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
MAX_NOTE = 1000  # characters of the reviewer's own words added to a prompt
MAX_SCENERY = 500  # the same limit products.csv enforces
CONCURRENT_REGENERATIONS = 2

# Which statuses each action may start from. A stale tab clicking "approve" on a
# product that was pushed in the meantime gets a no-op, not a state change.
ALLOWED_FROM = {
    "approve": {Status.PENDING, Status.REJECTED},
    "reject": {Status.PENDING, Status.APPROVED},
    "reopen": {Status.APPROVED, Status.REJECTED, Status.PUSHED},
}
RESULT_STATUS = {"approve": Status.APPROVED, "reject": Status.REJECTED, "reopen": Status.PENDING}


@dataclass
class Job:
    """A regeneration in flight, or one that failed and is waiting to be dismissed."""

    part: str
    state: str  # "running" or "failed"
    message: str = ""


def require_htmx(hx_request: Annotated[str | None, Header()] = None) -> None:
    """Refuse requests that did not come from the page's own HTMX calls.

    A web page on another site can make the browser POST to localhost, but it cannot
    add this header. That is what stops a stray tab from pushing to your catalogue.
    """
    if hx_request != "true":
        raise HTTPException(status_code=403, detail="use the review page")


def parse_edit(
    product: ReviewProduct, title: str, description: str, price: str, lang: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Check the edit form. Returns (cleaned values, errors by field)."""
    errors: dict[str, str] = {}
    title, description = title.strip(), description.strip()

    if not title:
        errors["title"] = "Required."
    elif len(title) > MetaListing.max_title:
        errors["title"] = f"Meta allows {MetaListing.max_title} characters."
    if not description:
        errors["description"] = "Required."
    elif len(description) > MetaListing.max_description:
        errors["description"] = f"Meta allows {MetaListing.max_description} characters."
    if lang not in product.langs:
        errors["lang"] = "No generated description in that language."

    try:
        amount = Decimal(price.replace(",", ".").replace(" ", ""))
        if not amount.is_finite() or amount <= 0:
            raise InvalidOperation
    except InvalidOperation:
        errors["price"] = "Enter a number above zero."
        amount = None

    cleaned = {"title": title, "description": description, "price": price, "lang": lang}
    if amount is not None:
        cleaned["price"] = str(amount)
    return cleaned, errors


def create_app(
    settings: ReviewSettings,
    pusher: Pusher,
    regenerator: Regenerator | None = None,
    runs: RunManager | None = None,
    app_log: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Product review", docs_url=None, redoc_url=None, openapi_url=None)
    push_lock = asyncio.Lock()  # one push at a time: no double-clicks, no overlapping Meta batches
    slots = asyncio.Semaphore(CONCURRENT_REGENERATIONS)
    review_lock = threading.Lock()  # review.json is read, changed and rewritten by threads and tasks
    jobs: dict[str, Job] = {}
    tasks: set[asyncio.Task] = set()  # keeps background tasks from being garbage collected

    def find(name: str) -> ReviewProduct:
        product = next((p for p in discover(settings).products if p.name == name), None)
        if product is None:
            raise HTTPException(status_code=404, detail="unknown product")
        return product

    def running(name: str) -> bool:
        job = jobs.get(name)
        return job is not None and job.state == "running"

    def board(
        request: Request,
        *,
        view: str = "all",
        editing: str | None = None,
        fixing: str | None = None,
        notes: dict[str, str] | None = None,
        form: dict[str, str] | None = None,
        errors: dict[str, str] | None = None,
        summary: str | None = None,
        full_page: bool = False,
    ) -> HTMLResponse:
        discovery = discover(settings)
        view = view if view in VIEWS else "all"
        counts = Counter(p.status.value for p in discovery.products)
        counts["all"] = len(discovery.products)
        shown = [p for p in discovery.products if view == "all" or p.status.value == view]
        return TEMPLATES.TemplateResponse(
            request,
            "index.html" if full_page else "_board.html",
            {
                "products": shown,
                "counts": counts,
                "views": VIEWS,
                "view": view,
                "problems": discovery.problems,
                "missing": pusher.missing(),
                "can_regenerate": regenerator is not None,
                "regen_missing": regenerator.missing() if regenerator else [],
                "image_problems": IMAGE_PROBLEMS,
                "description_problems": DESCRIPTION_PROBLEMS,
                "jobs": jobs,
                "poll": 3 if any(job.state == "running" for job in jobs.values()) else 6,
                "editing": editing,
                "fixing": fixing,
                "notes": notes or {},
                "form": form or {},
                "errors": errors or {},
                "summary": summary,
            },
        )

    async def push_product(name: str) -> tuple[bool, str]:
        """Push one approved product. Returns (ok, message for the card)."""
        async with push_lock:
            product = find(name)  # reload: the status may have changed while we waited
            if running(name):
                return False, "Not pushed: it is being regenerated, review the new version first."
            if product.status is not Status.APPROVED:
                return False, f"Not pushed: status is {product.status.value}, approve it first."
            if missing := pusher.missing():
                return False, f"Not pushed: set {', '.join(missing)} in .env."
            result = await pusher.push(product)
            if result.ok:
                update = {"status": Status.PUSHED, "error": None, "pushed_at": now()}
            else:
                update = {"error": result.message}
            with review_lock:
                save_review(product.folder, find(name).review.model_copy(update=update))
            return result.ok, result.message

    def finish_regeneration(name: str, part: str, note: str, ok: bool) -> None:
        """Log the attempt and, when it worked, send the product back for another look."""
        with review_lock:
            try:
                product = find(name)
            except HTTPException:
                return  # the product left products.csv while it was regenerating
            update: dict = {"log": [*product.review.log, Regeneration(at=now(), part=part, note=note, ok=ok)]}
            if ok:
                update |= {"status": Status.PENDING, "approved_fingerprint": None, "error": None}
                if part == "description":
                    update["description"] = None  # the regenerated text replaces a hand-edited one
            save_review(product.folder, product.review.model_copy(update=update))

    async def run_regeneration(name: str, part: str, note: str, scenery: str) -> None:
        # A background task has nobody to raise to, so whatever goes wrong has to end up
        # on the card, not in an unread "task exception was never retrieved" log line.
        try:
            async with slots:
                product = find(name)
                if part == "description":
                    await regenerator.description(product, note)
                else:
                    await regenerator.image(product, part, note, scenery)
        except Exception as error:  # noqa: BLE001
            logger.exception("regenerating %s of %s failed", part, name)
            finish_regeneration(name, part, note, ok=False)
            jobs[name] = Job(part, "failed", (str(error) or type(error).__name__)[:300])
            return
        finish_regeneration(name, part, note, ok=True)
        jobs.pop(name, None)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, view: str = "all"):
        return board(request, view=view, full_page=True)

    @app.get("/board", response_class=HTMLResponse)
    def refresh(request: Request, view: str = "all"):
        return board(request, view=view)

    @app.get("/media/{name}/{filename}")
    def media(name: str, filename: str):
        product = find(name)
        if filename not in product.images:  # only the shots description.json lists
            raise HTTPException(status_code=404, detail="unknown image")
        return FileResponse(product.folder / filename, headers={"Cache-Control": "no-cache"})

    @app.get("/products/{name}/edit", response_class=HTMLResponse)
    def edit(
        request: Request,
        name: str,
        view: str = "all",
        lang: str | None = None,
        title: str | None = None,
        price: str | None = None,
    ):
        product = find(name)
        lang = lang if lang in product.langs else product.lang
        # Switching language keeps the typed title and price but swaps in that
        # language's generated text (or the saved edit, if it is the saved language).
        description = (
            product.description
            if lang == product.lang
            else product.generated_description(lang)
        )
        form = {
            "title": title if title is not None else product.title,
            "price": price if price is not None else str(product.price),
            "description": description,
            "lang": lang,
        }
        return board(request, view=view, editing=name, form=form)

    @app.get("/products/{name}/fix", response_class=HTMLResponse)
    def fix(request: Request, name: str, view: str = "all"):
        find(name)
        return board(request, view=view, fixing=name)

    @app.post("/products/{name}/save", response_class=HTMLResponse, dependencies=[Depends(require_htmx)])
    def save(
        request: Request,
        name: str,
        # Defaults of "" so a cleared field reaches parse_edit and gets a message on the
        # card; a missing required field would be a bare 422 that HTMX does not swap in.
        title: Annotated[str, Form()] = "",
        description: Annotated[str, Form()] = "",
        price: Annotated[str, Form()] = "",
        lang: Annotated[str, Form()] = "",
        view: Annotated[str, Form()] = "all",
    ):
        if running(name):
            return board(request, view=view, notes={name: "Wait for the regeneration to finish before editing."})
        with review_lock:
            product = find(name)
            cleaned, errors = parse_edit(product, title, description, price, lang)
            if errors:
                return board(
                    request, view=view, editing=name, form=cleaned | {"description": description}, errors=errors
                )

            amount = Decimal(cleaned["price"])
            unchanged = (
                cleaned["title"] == product.title
                and cleaned["description"] == product.description
                and amount == product.price
                and lang == product.lang
            )
            if unchanged:
                return board(request, view=view, notes={name: "No changes."})

            # Store only what differs from the generated/CSV value, so those stay in charge elsewhere.
            review = product.review.model_copy(
                update={
                    "status": Status.PENDING,  # edited content has not been approved
                    "approved_fingerprint": None,
                    "lang": lang,
                    "title": cleaned["title"] if cleaned["title"] != product.base_title else None,
                    "description": (
                        cleaned["description"]
                        if cleaned["description"] != product.generated_description(lang)
                        else None
                    ),
                    "price": amount if amount != product.base_price else None,
                    "error": None,
                }
            )
            save_review(product.folder, review)
        return board(request, view=view, notes={name: "Saved. It is pending again: approve it to push."})

    @app.post(
        "/products/{name}/regenerate/{part}", response_class=HTMLResponse, dependencies=[Depends(require_htmx)]
    )
    async def regenerate(
        request: Request,
        name: str,
        part: str,
        view: Annotated[str, Form()] = "all",
        ticked: Annotated[list[str] | None, Form()] = None,
        note: Annotated[str, Form()] = "",
        scenery: Annotated[str, Form()] = "",
    ):
        product = find(name)
        is_description = part == "description"
        if not is_description and part not in {product.shot_of(image) for image in product.images}:
            raise HTTPException(status_code=404, detail="unknown part")

        def refuse(message: str) -> HTMLResponse:
            return board(request, view=view, fixing=name, notes={name: message})

        if regenerator is None:
            return refuse("Regenerating is not set up for this app.")
        if running(name):
            return refuse("It is already regenerating.")
        if push_lock.locked():
            return refuse("A push is running. Try again in a moment.")
        if missing := regenerator.missing():
            return refuse(f"Set {', '.join(missing)} in .env, then restart the app.")
        if len(note) > MAX_NOTE:
            return refuse(f"Keep the note under {MAX_NOTE} characters.")
        scenery = scenery.strip()
        if len(scenery) > MAX_SCENERY:
            return refuse(f"Keep the setting under {MAX_SCENERY} characters.")

        composed = compose_note("description" if is_description else "image", ticked or [], note)
        if not is_description and scenery != product.scenery:
            with review_lock:  # a changed scene is remembered for the next regeneration too
                fresh = find(name)
                save_review(
                    fresh.folder,
                    fresh.review.model_copy(
                        update={"scenery": scenery if scenery != (fresh.row.scenery_description or "") else None}
                    ),
                )
        scene = scenery if not is_description else product.scenery

        jobs[name] = Job(part, "running")
        task = asyncio.create_task(run_regeneration(name, part, composed, scene))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        label = "the description" if is_description else f"the {part} photo"
        return board(request, view=view, notes={name: f"Regenerating {label}. It will be pending when it is done."})

    @app.post("/products/{name}/restore/{shot}", response_class=HTMLResponse, dependencies=[Depends(require_htmx)])
    def restore(request: Request, name: str, shot: str, view: Annotated[str, Form()] = "all"):
        if running(name):
            return board(request, view=view, notes={name: "Wait for the regeneration to finish."})
        with review_lock:
            product = find(name)
            if regenerator is None or shot not in {product.shot_of(i) for i in product.images}:
                raise HTTPException(status_code=404, detail="unknown shot")
            if not regenerator.restore(product, shot):
                return board(request, view=view, notes={name: "There is no earlier version of that photo."})
            save_review(
                product.folder,
                product.review.model_copy(update={"status": Status.PENDING, "approved_fingerprint": None}),
            )
        return board(request, view=view, notes={name: f"Brought back the earlier {shot} photo. It is pending again."})

    @app.post("/products/{name}/dismiss", response_class=HTMLResponse, dependencies=[Depends(require_htmx)])
    def dismiss(request: Request, name: str, view: Annotated[str, Form()] = "all"):
        if not running(name):
            jobs.pop(name, None)
        return board(request, view=view)

    @app.post("/products/{name}/push", response_class=HTMLResponse, dependencies=[Depends(require_htmx)])
    async def push_one(request: Request, name: str, view: Annotated[str, Form()] = "all"):
        _, message = await push_product(name)
        return board(request, view=view, notes={name: message})

    # Declared after the fixed /products/{name}/<verb> routes so they match first.
    @app.post("/products/{name}/{action}", response_class=HTMLResponse, dependencies=[Depends(require_htmx)])
    def decide(
        request: Request,
        name: str,
        action: str,
        view: Annotated[str, Form()] = "all",
        fp: Annotated[str, Form()] = "",
    ):
        if action not in ALLOWED_FROM:
            raise HTTPException(status_code=404, detail="unknown action")
        with review_lock:
            product = find(name)
            if running(name):
                note = "It is being regenerated. Review the new version when it is ready."
            elif action == "approve" and fp != product.fingerprint:
                note = "This product was regenerated since you opened the page. Review the new version."
            elif product.status not in ALLOWED_FROM[action]:
                note = f"Nothing changed: it is already {product.status.value}."
            else:
                approving = action == "approve"
                save_review(
                    product.folder,
                    product.review.model_copy(
                        update={
                            "status": RESULT_STATUS[action],
                            "approved_fingerprint": product.fingerprint if approving else None,
                            "error": None,
                        }
                    ),
                )
                note = ""
        return board(request, view=view, notes={name: note} if note else None)

    @app.post("/push-approved", response_class=HTMLResponse, dependencies=[Depends(require_htmx)])
    async def push_all(request: Request, view: Annotated[str, Form()] = "all"):
        names = [p.name for p in discover(settings).products if p.status is Status.APPROVED]
        notes: dict[str, str] = {}
        pushed = 0
        for name in names:
            ok, message = await push_product(name)
            pushed += ok
            if not ok:
                notes[name] = message
        summary = f"Pushed {pushed} of {len(names)} approved product(s)." if names else "Nothing approved to push."
        return board(request, view=view, notes=notes, summary=summary)

    if runs is not None:  # upload, run history and logs; see product_catalog/runs
        register_run_routes(app, runs, TEMPLATES, require_htmx, app_log)

    return app
