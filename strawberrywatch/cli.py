"""The strawberrywatch command. Today it has one job: `strawberrywatch dashboard`."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import webbrowser

DEFAULT_PORT = 8790


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="strawberrywatch", description="StrawberryWatch on this computer."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dash = commands.add_parser(
        "dashboard",
        help="Run the StrawberryWatch dashboard on this computer.",
        description="Starts the dashboard on this computer. Models check the data every 15 minutes while it runs.",
    )
    dash.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port on 127.0.0.1, default {DEFAULT_PORT}"
    )
    dash.add_argument(
        "--no-browser", action="store_true", help="Print the link instead of opening a browser tab"
    )
    args = parser.parse_args(argv)
    if args.command == "dashboard":
        return dashboard(args.port, not args.no_browser)
    return 2


def dashboard(port=DEFAULT_PORT, open_browser=True):
    try:
        from dotenv import find_dotenv, load_dotenv
        from werkzeug.serving import make_server

        from strawberrywatch.dashboard import create_app
    except ImportError:
        print(
            'The dashboard needs its extras: pip install "strawberrywatch[dashboard]"',
            file=sys.stderr,
        )
        return 1

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Werkzeug logs every request line, and the first one carries the login token
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # The MYSQL login and SCMG_API_TOKEN, from the .env in the folder it was started from
    load_dotenv(find_dotenv(usecwd=True))
    if not os.getenv("MYSQL_HOST") and not os.getenv("SCMG_API_TOKEN"):
        print(
            "No data source in .env: add the MYSQL_* settings or SCMG_API_TOKEN, "
            "or every check will fail.",
            flush=True,
        )
    print("Loading the models (a few seconds)...", flush=True)
    app = create_app(store=ready())
    context = app.extensions["strawberrywatch"]

    # 127.0.0.1 only. There's no option for another address.
    try:
        server = make_server("127.0.0.1", port, app, threaded=True)
    except OSError:
        server = make_server("127.0.0.1", 0, app, threaded=True)
    url = f"http://127.0.0.1:{server.port}/login?token={context.login.token}"
    print("StrawberryWatch is running. Open this link to sign in:", flush=True)
    print(f"  {url}", flush=True)
    print("The link works once. Press Ctrl+C to stop.", flush=True)

    context.watcher.start()
    context.watcher.check_now()
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    # A plain kill tidies up the same way Ctrl+C does
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. No checks run until StrawberryWatch is started again.", flush=True)
    finally:
        context.watcher.stop()
        server.server_close()
    return 0


def ready():
    """
    Sets up the local store the way setup_tables sets up the website's, on the
    sensor database with default settings. A source left over from the removed
    source picker gets replaced.
    """
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import models as watch_models
    from strawberrywatch.watch import settings as ws
    from strawberrywatch.watch import sources
    from strawberrywatch.watch.store import Store

    store = Store()
    if (store.source() or {}).get("kind") != sources.SITE:
        store.set_source(sources.SITE)
    if store.applied()[1] is None:
        models = watch_models.registered()
        clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
        store.apply(clean)
    return store


if __name__ == "__main__":
    sys.exit(main())
