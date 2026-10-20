"""
The dashboard as the website runs it. The site mounts make_dashboard() at
/strawberrywatch/ (see INSTALL.md). It reads the website's sensor database,
uses the site's own sign-in, and keeps its own data in sw_ tables. Night Heron
runs the checks.
"""

from __future__ import annotations

MOUNT = "/strawberrywatch"


class NoWatcher:
    """Night Heron runs the website's checks. There's no watcher to start here."""

    busy = False
    next_at = None

    def check_now(self):
        return None


def make_dashboard(store=None):
    """Returns the dashboard as a WSGI app, behind the gate."""
    from strawberrywatch.dashboard import create_app, profiles
    from strawberrywatch.integrations.scmg_website import night_heron_list
    from strawberrywatch.integrations.scmg_website.gate import Gate
    from strawberrywatch.watch import models, sources
    from strawberrywatch.watch.model_updates import ChildUpdater
    from strawberrywatch.watch.store_mysql import MysqlStore

    store = store or MysqlStore()
    # Always the website's database, whatever a visitor sends
    if (store.source() or {}).get("kind") != sources.SITE:
        store.set_source(sources.SITE)
    app = create_app(
        store=store,
        models=models.registered(),
        watcher=NoWatcher(),
        profile=profiles.WEBSITE,
        signing_key=store.session_key(),
        updater=ChildUpdater(),
        copy_list=night_heron_list.addresses,
    )
    app.config["SESSION_COOKIE_PATH"] = MOUNT
    app.config["SIGN_OUT_URL"] = sign_out_url()
    return Gate(app)


def sign_out_url():
    """The site's own sign-out page."""
    try:
        from django.urls import reverse

        return reverse("logout")
    except Exception:
        return "/logout/"
