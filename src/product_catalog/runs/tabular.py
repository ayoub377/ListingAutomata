"""Turning an uploaded spreadsheet into the CSV that `load_products_csv` already reads.

Excel and French-locale CSV exports differ from what that parser expects (`;` as the
separator, cp1252 text, numbers stored as floats). Everything is normalised to a plain
UTF-8, comma-separated CSV here, so the one validated parser stays the only parser.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from pathlib import Path

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SUPPORTED = (".csv", ".xlsx")
DELIMITERS = ",;\t"


class TabularError(ValueError):
    """The upload cannot be read as a table of products. The message is shown to the user."""


def normalize(filename: str, data: bytes) -> bytes:
    """The upload as UTF-8 comma-separated CSV bytes. Raises TabularError."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        hint = " Save it as .xlsx or .csv first." if suffix in (".xls", ".ods") else ""
        raise TabularError(f"Upload a .csv or .xlsx file (got {suffix or 'no extension'}).{hint}")
    if not data:
        raise TabularError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise TabularError(f"The file is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    rows = _csv_rows(data) if suffix == ".csv" else _xlsx_rows(data)
    return _to_csv(rows)


def _csv_rows(data: bytes) -> list[list[str]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")  # Excel's "CSV" on Windows
    first = text.splitlines()[0] if text.strip() else ""
    delimiter = max(DELIMITERS, key=first.count) if first else ","
    return [list(row) for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)]


def _xlsx_rows(data: bytes) -> list[list[str]]:
    try:
        from openpyxl import load_workbook

        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except ImportError as error:
        raise TabularError("Excel support needs openpyxl (uv add openpyxl).") from error
    except Exception as error:  # noqa: BLE001 - openpyxl raises many types for a bad file
        raise TabularError(f"Could not open the Excel file: {error}") from error
    try:
        sheet = workbook.active
        if sheet is None:
            raise TabularError("The workbook has no sheets.")
        return [[_cell(value) for value in row] for row in sheet.iter_rows(values_only=True)]
    finally:
        workbook.close()


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # 4300.0 would not read as a price like "4300" does
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value).strip()


def _to_csv(rows: list[list[str]]) -> bytes:
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise TabularError("Needs a header row and at least one product row.")
    header = [cell.strip() for cell in rows[0]]
    keep = [index for index, name in enumerate(header) if name]  # drop trailing blank columns
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([header[index] for index in keep])
    for row in rows[1:]:
        writer.writerow([row[index].strip() if index < len(row) else "" for index in keep])
    return out.getvalue().encode("utf-8")
