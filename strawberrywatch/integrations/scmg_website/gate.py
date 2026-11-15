"""Lets the site's signed-in visitors through and sends the rest to its login page and back."""

from __future__ import annotations

import logging

from strawberrywatch.dashboard.security import SiteLogin

logger = logging.getLogger(__name__)

# Open to anyone, for the unsubscribe link in alert emails and the stylesheet
# and logo that page loads. The dashboard checks again on each page
PUBLIC_PATHS = ("/unsubscribe/", "/static/")
# Read by the site's Alerts link, which sends people back to the dashboard they
# were last on. One sign-in covers both
DASHBOARD_KEY = "dashboard"
DASHBOARD = "strawberrywatch"


class Gate:
    """WSGI middleware around the dashboard, run inside the site's own process."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path.startswith(PUBLIC_PATHS):
            return self.app(environ, start_response)
        request = site_request(environ)
        user = visitor(request)
        if user is None or user.is_anonymous:
            return self._to_login(environ, start_response)
        remember(request)
        # All the dashboard learns about a visitor
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


def site_request(environ):
    """This call as a Django request, with the site's session loaded."""
    from django.contrib.sessions.middleware import SessionMiddleware
    from django.core.handlers.wsgi import WSGIRequest

    request = WSGIRequest(environ)
    SessionMiddleware(lambda _request: None).process_request(request)
    return request


def visitor(request):
    """Django's idea of who sent this request, or None when it can't say."""
    from django.contrib.auth import get_user

    return get_user(request)  # the same check login_required makes


def remember(request):
    """Notes in the session that this browser is on StrawberryWatch."""
    if request.session.get(DASHBOARD_KEY) == DASHBOARD:
        return
    try:
        request.session[DASHBOARD_KEY] = DASHBOARD
        request.session.save()  # same session key, and the site's cookie stays as it is
    except Exception:
        # A line in the journal, not a failed page. The Alerts link keeps
        # pointing at Night Heron
        logger.warning("couldn't note which dashboard is in use", exc_info=True)


def may_edit(user):
    """Anyone signed in, unless the site sets STRAWBERRYWATCH_EDITOR_GROUP. Superusers always."""
    from django.conf import settings

    group = getattr(settings, "STRAWBERRYWATCH_EDITOR_GROUP", None)
    if not group or user.is_superuser:
        return True
    return user.groups.filter(name=group).exists()
