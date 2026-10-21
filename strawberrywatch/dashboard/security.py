"""
Keeps the local app safe from other sites open in the same browser: a Host
check, a one-time login link, a strict CSP, and error pages with no traceback.
CSRF is Flask-WTF's, set up in create_app.

The website skips the Host check and the link, since its gate already knows
who's signed in.

The Host check and login link follow Jupyter Server's check_host
(base/handlers.py) and token login (auth/identity.py), BSD-3.
"""

from __future__ import annotations

import re
import secrets

from flask import abort, g, redirect, render_template, request, session, url_for

# Any port is fine. Refusing other names blocks DNS rebinding, where a site
# points its own name at 127.0.0.1 to reach us through the browser.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)

# Reachable without signing in
OPEN_ENDPOINTS = {"main.login", "main.sign_out", "static"}
# The website's one public page: leaving the email list, linked from every alert
PUBLIC_ENDPOINTS = {"main.unsubscribe"}


def host_ok(header):
    """Whether the Host header is this computer. Ignores the port and IPv6 brackets."""
    match = re.match(r"^(.*?)(:\d+)?$", header or "")
    host = match.group(1) if match else ""
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host.lower() in LOCAL_HOSTS


class Login:
    """
    The token printed at startup. It only works once. A copy left in a
    terminal's scrollback is useless.
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


class SiteLogin:
    """
    Who's signed in on the website. gate.py puts a username and an edit flag in
    each request, and turns away anyone not signed in before they get here.
    """

    USER = "strawberrywatch.username"
    CAN_EDIT = "strawberrywatch.can_edit"

    def user(self, environ):
        return environ.get(self.USER)

    def can_edit(self, environ):
        return bool(environ.get(self.CAN_EDIT))


def owner():
    """Whose drafts and notes these are. The username on the website, None locally."""
    login = current_login()
    if isinstance(login, SiteLogin):
        return login.user(request.environ)
    return None


def can_edit():
    login = current_login()
    if isinstance(login, SiteLogin):
        return login.can_edit(request.environ)
    return True


def current_login():
    from flask import current_app

    return current_app.extensions["strawberrywatch"].login


def init(app, profile):
    @app.before_request
    def guard():
        if profile.token_login:
            if not host_ok(request.headers.get("Host")):
                abort(403, "StrawberryWatch only answers on this computer.")
            if request.endpoint in OPEN_ENDPOINTS:
                return None
            if not session.get("signed_in"):
                return render_template("signed_out.html"), 403
            return None
        if request.endpoint == "static" or request.endpoint in PUBLIC_ENDPOINTS:
            return None
        if not owner():
            abort(403, "Sign in to strawberrycreek.org to use StrawberryWatch.")
        if request.method not in ("GET", "HEAD") and not can_edit():
            abort(403, "Your account has view-only access. Ask an admin to make changes.")
        return None

    @app.after_request
    def headers(response):
        response.headers["Content-Security-Policy"] = g.get("csp") or CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        # Flask-WTF wants a same-site Referer on every HTTPS form. The website
        # sends one to itself only
        response.headers["Referrer-Policy"] = (
            "no-referrer" if profile.token_login else "same-origin"
        )
        if not request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    @app.errorhandler(500)
    @app.errorhandler(503)
    def error(exc):
        # No traceback. Debug mode stays off, and this is all anyone sees
        code = getattr(exc, "code", 500) or 500
        message = {
            400: "Something went wrong with the form. Go back and try again.",
            403: getattr(exc, "description", None) or "You don't have access to this page.",
            404: "Page not found",
            405: "This action isn't available.",
            413: "The upload is too large.",
            503: getattr(exc, "description", None) or "StrawberryWatch isn't ready yet.",
        }.get(code, "Something went wrong. The error has been logged.")
        if code == 403 and profile.token_login and not host_ok(request.headers.get("Host")):
            return message, 403, {"Content-Type": "text/plain; charset=utf-8"}
        # The store may be what failed. The page doesn't query it
        g.error_page = True
        return render_template("error.html", code=code, message=message), code


def sign_in(login):
    """Handles the login link, then redirects so the token doesn't stay in the address bar."""
    if not login.accept(request.args.get("token", "")):
        return render_template("signed_out.html", used=login.used), 403
    session.clear()
    session["signed_in"] = True
    return redirect(url_for("main.overview"))
