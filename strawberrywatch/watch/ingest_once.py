"""Runs one ingest pass in its own process and prints one line of JSON."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import timedelta

logger = logging.getLogger(__name__)


def main(store=None, argv=None):
    from strawberrywatch.watch import ingest, sources

    parser = argparse.ArgumentParser(prog="python -m strawberrywatch.watch.ingest_once")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="copy data/raw_data and the readings already saved into the raw level, once",
    )
    parser.add_argument(
        "--days", type=float, default=None, help="reach further back than two days this pass"
    )
    parser.add_argument(
        "--local", action="store_true", help="use the local app's store, not the website's"
    )
    args = parser.parse_args(argv)

    began = time.time()
    if store is None and args.local:
        import os

        from strawberrywatch.watch.store import Store

        store, token, mail = Store(), os.getenv("SCMG_API_TOKEN"), ingest.LOCAL_MAIL
    else:
        from strawberrywatch.watch.store_mysql import MysqlStore

        store, token, mail = store or MysqlStore(), None, ingest.WEBSITE_MAIL
        if (store.source() or {}).get("kind") != sources.SITE:
            store.set_source(sources.SITE)

    result = {}
    if args.backfill:
        result["backfill"] = ingest.backfill(store)
    lookback = timedelta(days=args.days) if args.days else ingest.LOOKBACK
    row = ingest.run(store, token=token, lookback=lookback, mail=mail)
    result.update(
        {
            "ok": bool(row and row.get("ok")),
            "message": (row or {}).get("message") or "",
            "rows_new": (row or {}).get("rows_new") or 0,
            "seconds": round(time.time() - began, 1),
        }
    )
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    raise SystemExit(main())
