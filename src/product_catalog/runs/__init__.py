"""Upload a CSV/Excel file, run the workflow on it, and keep a searchable history and log of every run.

Layers, each importable without torch or a network call:
  tabular  uploaded spreadsheet -> normalised CSV
  store    run history in SQLite
  logs     per-run JSONL event logs, redaction, merged provider records
  runner   upload -> preview -> confirm -> execute, one run at a time
  workflow the seam to scripts/workflow.py (a fake in tests)
  routes   the /runs pages, registered onto the review app
"""
