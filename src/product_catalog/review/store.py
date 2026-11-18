"""What a human decided about each generated product, kept next to its output.

`output/<folder>/review.json` holds the reviewer's edits and their verdict. The two
have different lifetimes:

  * edits (title, description, price, scene) belong to the reviewer and survive a
    regeneration, because regenerating a photo must not wipe a price they fixed;
  * an approval belongs to one version of the content. It records the fingerprint of
    what was on screen when the reviewer clicked, and reads as pending again as soon
    as the text or any image changes, so nothing nobody looked at can be pushed.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from product_catalog.models import PHOTO_SUFFIXES

REVIEW_FILE = "review.json"
DESCRIPTION_FILE = "description.json"
MAX_LOG = 50


class Status(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUSHED = "pushed"


class Regeneration(BaseModel):
    """One correction the reviewer asked for, kept so they can see which wording worked."""

    at: str
    part: str  # "angle", "context" or "description"
    note: str
    ok: bool = True


class Review(BaseModel):
    """The reviewer's verdict on one product, plus what they changed.

    Edits are stored as overrides: `None` means "keep the generated or CSV
    value", so a later regeneration or CSV fix is never masked by a stale copy.
    """

    model_config = ConfigDict(extra="ignore")

    status: Status = Status.PENDING
    approved_fingerprint: str | None = Field(
        default=None, description="The content version the approval or push applies to."
    )
    lang: str = "fr"
    title: str | None = None
    description: str | None = None
    price: Decimal | None = Field(default=None, gt=0)
    scenery: str | None = Field(default=None, description="Scene override for the image prompt.")
    error: str | None = Field(default=None, description="Why the last push failed.")
    updated_at: str | None = None
    pushed_at: str | None = None
    log: list[Regeneration] = Field(default_factory=list)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def fingerprint(folder: Path) -> str:
    """A hash of everything the reviewer looks at: the text and each image's size and mtime.

    Image bytes are not read; size plus mtime changes whenever a shot is rewritten, and
    the board recomputes this on every refresh. The prev/ backups are not part of it.
    """
    digest = hashlib.sha256()
    description = folder / DESCRIPTION_FILE
    digest.update(description.read_bytes() if description.is_file() else b"")
    for image in sorted(folder.glob("product_*")):
        if image.suffix.lower() in PHOTO_SUFFIXES:
            stat = image.stat()
            digest.update(f"{image.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def load_review(folder: Path, current: str | None = None) -> Review:
    """The saved review, with a stale approval downgraded to pending.

    A missing or unreadable file means "nobody has reviewed this", which is the
    safe answer. Pass `current` when the fingerprint is already known.
    """
    current = current or fingerprint(folder)
    try:
        review = Review.model_validate_json((folder / REVIEW_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pydantic's ValidationError is a ValueError
        return Review()
    if review.status in (Status.APPROVED, Status.PUSHED) and review.approved_fingerprint != current:
        return review.model_copy(update={"status": Status.PENDING})
    return review


def save_review(folder: Path, review: Review) -> Review:
    """Write the review atomically, so a crash never leaves half a file that reads as pending."""
    review = review.model_copy(update={"updated_at": now(), "log": review.log[-MAX_LOG:]})
    temp = folder / f"{REVIEW_FILE}.tmp"
    temp.write_text(review.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temp, folder / REVIEW_FILE)
    return review
