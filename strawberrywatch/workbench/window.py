"""strawberrywatch workbench: the app on a local server thread, shown in its own window."""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from werkzeug.serving import make_server


class Api:
    """What the page can ask the window for. pywebview hands JavaScript these methods."""

    def __init__(self, folder):
        self._folder = folder

    def save_file(self, name, data_base64):
        import webview

        dialog = getattr(webview, "FileDialog", None)
        kind = dialog.SAVE if dialog else webview.SAVE_DIALOG
        chosen = webview.windows[0].create_file_dialog(kind, save_filename=name)
        if not chosen:
            return None
        path = Path(chosen if isinstance(chosen, str) else chosen[0])
        path.write_bytes(base64.b64decode(data_base64))
        return str(path)

    def choose_folder(self):
        import webview

        dialog = getattr(webview, "FileDialog", None)
        kind = dialog.FOLDER if dialog else webview.FOLDER_DIALOG
        chosen = webview.windows[0].create_file_dialog(kind)
        if not chosen:
            return None
        return str(chosen if isinstance(chosen, str) else chosen[0])

    def open_folder(self):
        path = str(self._folder)
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)


def main(browser=False):
    from strawberrywatch.dashboard import security
    from strawberrywatch.workbench.app import create_app

    if not browser:
        try:
            import webview
        except ImportError:
            print(
                'The Model Tuner window needs pywebview: pip install -e ".[workbench]", or use --browser'
            )
            return 1
    # Werkzeug logs every request line, and the first one carries the login token
    logging.getLogger("werkzeug").disabled = True
    login = security.Login()
    app = create_app(browser=browser, login=login)
    folder = app.extensions["strawberrywatch"].folder
    # 127.0.0.1 only, on any free port
    server = make_server("127.0.0.1", 0, app, threaded=True)
    url = f"http://127.0.0.1:{server.port}/login?token={login.token}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if browser:
        print(f"Model Tuner running at {url}", flush=True)
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return 0
    webview.create_window(
        "StrawberryWatch Model Tuner",
        url,
        width=1280,
        height=860,
        min_size=(900, 640),
        js_api=Api(folder),
    )
    webview.start()
    server.shutdown()
    return 0
