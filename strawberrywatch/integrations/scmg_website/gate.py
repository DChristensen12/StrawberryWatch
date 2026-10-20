"""
Asks Django whether each visitor is signed in, with get_user, the same way
login_required does. Anyone who isn't gets sent to the site's login page and
back again, except on the public sign-up and unsubscribe pages. The dashboard
only ever sees a username and whether they may edit.
"""

from __future__ import annotations

from strawberrywatch.dashboard.security import SiteLogin

# Open to anyone: signing up for alerts, confirming, unsubscribing, and the
# stylesheet and logo those pages load. The dashboard checks again by page.
PUBLIC_PATHS = ("/signup/", "/unsubscribe/", "/static/")


class Gate:
    """WSGI middleware around the dashboard, run inside the site's own process."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/signup" or path.startswith(PUBLIC_PATHS):
            return self.app(environ, start_response)
        user = visitor(environ)
        if user is None or user.is_anonymous:
            return self._to_login(environ, start_response)
        environ[SiteLogin.USER] = user.get_username()
        environ[SiteLogin.CAN_EDIT] = may_edit(user)
        return self.app(environ, start_response)

    def _to_login(self, environ, start_response):
        from django.contrib.auth.views import redirect_to_login

        back = environ.get("SCRIPT_NAME", "") + environ.get("PATH_INFO", "")
        if environ.get("QUERY_STRING"):
            back += "?" + environ["QUERY_STRING"]
        response = redirect_to_login(back)
        start_response("302 Found", [("Location", response["Location"]), ("Content-Length", "0")])
        return [b""]


def visitor(environ):
    """Django's idea of who sent this request, or None when it can't say."""
    from django.contrib.auth import get_user
    from django.contrib.sessions.middleware import SessionMiddleware
    from django.core.handlers.wsgi import WSGIRequest

    request = WSGIRequest(environ)
    SessionMiddleware(lambda _request: None).process_request(request)
    return get_user(request)


def may_edit(user):
    """
    Anyone who can sign in can edit, unless the site names an editor group in
    STRAWBERRYWATCH_EDITOR_GROUP. Superusers always can.
    """
    from django.conf import settings

    group = getattr(settings, "STRAWBERRYWATCH_EDITOR_GROUP", None)
    if not group or user.is_superuser:
        return True
    return user.groups.filter(name=group).exists()
