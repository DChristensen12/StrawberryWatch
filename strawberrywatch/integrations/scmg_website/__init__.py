"""The dashboard as the website runs it, mounted at /strawberrywatch/ (see INSTALL.md)."""

from __future__ import annotations

MOUNT = "/strawberrywatch"


class NoWatcher:
    """Stands in for the watcher. Night Heron runs the website's checks."""

    busy = False
    next_at = None

    def check_now(self):
        return None


def make_dashboard(store=None):
    """Returns the dashboard as a WSGI app behind the site's own sign-in."""
    from strawberrywatch.dashboard import create_app, profiles
    from strawberrywatch.integrations.scmg_website import night_heron_list
    from strawberrywatch.integrations.scmg_website.gate import Gate
    from strawberrywatch.watch import models, sources
    from strawberrywatch.watch.model_updates import ChildUpdater
    from strawberrywatch.watch.store_mysql import MysqlStore

    store = store or MysqlStore()  # its own data, in the sw_ tables
    # Readings always come from the website's sensor database, whatever is saved
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
    app.config["OTHER_DASHBOARD"] = {"text": "Night Heron", "href": night_heron_url()}
    return Gate(app)


def sign_out_url():
    return site_url("logout", "/logout/")


def night_heron_url():
    """The site's Alerts link, which also notes Night Heron as the dashboard in use."""
    return site_url("alerts", "/alerts/") + "?to=night-heron"


def site_url(name, fallback):
    try:
        from django.urls import reverse

        return reverse(name)
    except Exception:
        return fallback
