"""`python -m product_catalog.review` - serve the review board and the runs pages on localhost."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from product_catalog.review.app import create_app
from product_catalog.review.catalog import ReviewSettings
from product_catalog.review.publisher import MetaPusher
from product_catalog.review.regenerator import WorkflowRegenerator
from product_catalog.runs.setup import build as build_runs

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    runs, app_log = build_runs(PROJECT_ROOT)
    # The board also reads the CSVs uploaded on /runs, so a run's products show up there.
    settings = ReviewSettings.from_project(PROJECT_ROOT, extra_csvs=runs.board_csvs)
    app = create_app(settings, MetaPusher(), WorkflowRegenerator(), runs, app_log)
    # Localhost only: the page can push to your Meta catalogue, so it is never exposed.
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
