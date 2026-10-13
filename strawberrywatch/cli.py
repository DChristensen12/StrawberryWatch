"""The strawberrywatch command. Today it has one job: `strawberrywatch dashboard`."""

from __future__ import annotations

import argparse
import logging
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
        help="Watch the creek on this computer and see it in your browser.",
        description="Starts the local dashboard. The models check the creek every 15 minutes while it runs.",
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

    print("Loading the models. That imports torch, so it takes a few seconds...", flush=True)
    app = create_app()
    context = app.extensions["strawberrywatch"]

    # 127.0.0.1 and nothing else, on purpose. There's no option for another address.
    try:
        server = make_server("127.0.0.1", port, app, threaded=True)
    except OSError:
        server = make_server("127.0.0.1", 0, app, threaded=True)
    url = f"http://127.0.0.1:{server.port}/login?token={context.login.token}"
    print("StrawberryWatch is watching the creek. Open this link to see it:", flush=True)
    print(f"  {url}", flush=True)
    print("The link works once. Press Ctrl+C here to stop watching.", flush=True)

    context.watcher.start()
    context.watcher.check_now()
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    # A plain kill tidies up the same way Ctrl+C does
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping. Nothing is watched until StrawberryWatch is started again.", flush=True)
    finally:
        context.watcher.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
