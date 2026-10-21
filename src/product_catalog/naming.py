"""How a product gets its name on disk.

Every stage of the pipeline writes under the same folder name — segment.py puts
cutouts in `images/<name>/`, generate.py puts staged shots in `output/<name>/` —
so the rule that derives that name has to live somewhere both can import.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from product_catalog.models import ProductRow


def slugify(value: str) -> str:
    """`Piscine Bestway 3.96M x 84cm` -> `piscine-bestway-3-96m-x-84cm`."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", normalized.lower())).strip("-")


def product_name(row: ProductRow) -> str:
    """Output folder name for a row: its title, else its photo folder's name."""
    return slugify(row.title) if row.title else slugify(row.product_folder.name)
