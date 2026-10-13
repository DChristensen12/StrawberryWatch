"""
The local dashboard: a Flask app that shows what the models notice on this
computer and lets someone change how they watch.

It's dressed as a page of strawberrycreek.org, so it looks like the rest of the
website. There's no Docker, database server or rate limiter, since this runs on
one computer for one person.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from flask import Flask
from flask_wtf.csrf import CSRFProtect
from jinja2 import PackageLoader

from strawberrywatch.dashboard import security

csrf = CSRFProtect()


@dataclass
class Context:
    """What every request shares."""

    store: object
    models: list
    inventory: object
    watcher: object
    login: security.Login


def create_app(store=None, models=None, inventory=None, watcher=None, login=None):
    """
    The app, wired to a store, the registered models and a watcher. Anything left
    out is made the way `strawberrywatch dashboard` makes it, except the watcher,
    which the caller starts so a test can leave it stopped.
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
            store, models, after_check=lambda check_id: mail.after_check(store, inventory, check_id)
        )

    app = Flask(__name__)
    app.config.update(
        # New every launch, so a session ends when the app does
        SECRET_KEY=secrets.token_hex(32),
        SESSION_COOKIE_NAME="strawberrywatch",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        # http on 127.0.0.1, so a Secure cookie would never be sent back
        SESSION_COOKIE_SECURE=False,
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
    app.extensions["strawberrywatch"] = Context(
        store, models, inventory, watcher, login or security.Login()
    )
    security.init(app)

    from strawberrywatch.dashboard.routes import bp

    app.register_blueprint(bp)
    return app
