"""
The dashboard, a Flask app. The local app runs it for one person on one
computer, and the website mounts the same app behind its own login.
profiles.py has the differences.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from flask import Flask
from flask_wtf.csrf import CSRFProtect
from jinja2 import PackageLoader

from strawberrywatch.dashboard import profiles, security

csrf = CSRFProtect()


@dataclass
class Context:
    """What every request shares."""

    store: object
    models: list
    inventory: object
    watcher: object
    login: object
    # Swappable so tests can pin "now"
    clock: Callable[[], float] = field(default=time.time)
    profile: profiles.Profile = profiles.LOCAL
    # Pulls model updates from GitHub. Website only.
    updater: object = None
    # Reads the addresses on Night Heron's alert rules, for the one-time copy. Website only.
    copy_list: Callable[[], list] | None = None


def create_app(
    store=None,
    models=None,
    inventory=None,
    watcher=None,
    login=None,
    profile=profiles.LOCAL,
    clock=None,
    signing_key=None,
    updater=None,
    copy_list=None,
):
    """
    Builds the app. Arguments left out are made the way `strawberrywatch
    dashboard` makes them. The caller starts the watcher, and tests can leave it
    stopped.
    """
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import models as watch_models
    from strawberrywatch.watch.runner import Watcher
    from strawberrywatch.watch.store import Store

    store = store or Store()
    inventory = inventory or inv.load()
    models = models if models is not None else watch_models.registered(inventory)
    if watcher is None:
        from strawberrywatch.dashboard import mail

        watcher = Watcher(
            store,
            models,
            api_token=lambda: os.getenv("SCMG_API_TOKEN"),
            after_check=lambda check_id: mail.after_check(store, inventory, check_id),
        )

    app = Flask(__name__)
    app.config.update(
        # New every launch. A session ends when the app does. The website passes
        # its own, since it may run more than one process.
        SECRET_KEY=signing_key or secrets.token_hex(32),
        SESSION_COOKIE_NAME="strawberrywatch",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict" if profile.token_login else "Lax",
        # http on 127.0.0.1, where a Secure cookie wouldn't be sent back
        SESSION_COOKIE_SECURE=not profile.token_login,
        MAX_CONTENT_LENGTH=256 * 1024,
        WTF_CSRF_TIME_LIMIT=None,
        DEBUG=False,
        PROPAGATE_EXCEPTIONS=False,
        TRAP_HTTP_EXCEPTIONS=False,
    )
    app.debug = False
    app.jinja_env.lstrip_blocks = True
    app.jinja_env.trim_blocks = True
    app.jinja_loader = PackageLoader("strawberrywatch.dashboard")

    csrf.init_app(app)
    if login is None:
        login = security.Login() if profile.token_login else security.SiteLogin()
    app.extensions["strawberrywatch"] = Context(
        store, models, inventory, watcher, login, clock or time.time, profile, updater, copy_list
    )
    security.init(app, profile)

    from strawberrywatch.dashboard.routes import bp

    app.register_blueprint(bp)
    return app
