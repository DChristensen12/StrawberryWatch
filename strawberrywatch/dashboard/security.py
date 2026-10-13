"""
What keeps a page on someone's laptop safe from other sites open in the same
browser: a Host check, a one-time login link, a strict Content-Security-Policy,
and error pages that never show a traceback. CSRF protection is Flask-WTF's,
set up in create_app.

The Host check and the login link follow Jupyter Server's: check_host in
base/handlers.py and the token login in auth/identity.py (BSD-3).
"""

from __future__ import annotations

import re
import secrets

from flask import abort, g, redirect, render_template, request, session, url_for

# Only these, with any port. Refusing everything else blocks DNS rebinding, where a
# site points its own name at 127.0.0.1 to reach this app from the browser.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)

# Reachable without signing in: the link itself, and the files a sign-in page needs
OPEN_ENDPOINTS = {"main.login", "static"}


def host_ok(header):
    """Whether a Host header names this computer, with the port and IPv6 brackets stripped."""
    match = re.match(r"^(.*?)(:\d+)?$", header or "")
    host = match.group(1) if match else ""
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host.lower() in LOCAL_HOSTS


class Login:
    """
    The token printed at startup. It works once: opening it sets the session
    cookie and uses it up, so a copy left in a terminal's scrollback is no key.
    """

    def __init__(self, token=None):
        self.token = token or secrets.token_urlsafe(32)
        self.used = False

    def accept(self, offered):
        if self.used or not offered:
            return False
        if secrets.compare_digest(offered.encode(), self.token.encode()):
            self.used = True
            return True
        return False


def init(app):
    @app.before_request
    def guard():
        if not host_ok(request.headers.get("Host")):
            abort(403, "StrawberryWatch only answers on this computer.")
        if request.endpoint in OPEN_ENDPOINTS:
            return None
        if not session.get("signed_in"):
            return render_template("signed_out.html"), 403
        return None

    @app.after_request
    def headers(response):
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    @app.errorhandler(500)
    def error(exc):
        # Never the traceback: debug mode stays off and this is all anyone sees
        code = getattr(exc, "code", 500) or 500
        message = {
            400: "The form couldn't be read. Go back and try again.",
            403: getattr(exc, "description", None) or "That isn't allowed.",
            404: "There's no page here.",
            405: "That isn't allowed.",
            413: "That was too big to send.",
        }.get(code, "Something went wrong. It's been logged on this computer.")
        if code == 403 and not host_ok(request.headers.get("Host")):
            return message, 403, {"Content-Type": "text/plain; charset=utf-8"}
        # Whatever failed may be the store, so the page asks it nothing
        g.error_page = True
        return render_template("error.html", code=code, message=message), code


def sign_in(login):
    """The login link: a good token sets the session and is used up, then the token leaves the address bar."""
    if not login.accept(request.args.get("token", "")):
        return render_template("signed_out.html", used=login.used), 403
    session.clear()
    session["signed_in"] = True
    return redirect(url_for("main.overview"))
