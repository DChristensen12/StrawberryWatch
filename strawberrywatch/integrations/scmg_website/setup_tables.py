"""
Creates StrawberryWatch's sw_ tables on the website's database, using Night
Heron's MySQL login. Safe to run twice, existing tables are left alone.

    python -m strawberrywatch.integrations.scmg_website.setup_tables
"""

from __future__ import annotations


def main(store=None):
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import models as watch_models
    from strawberrywatch.watch import settings as ws
    from strawberrywatch.watch import sources
    from strawberrywatch.watch.store_mysql import MysqlStore

    store = store or MysqlStore()
    store.create_tables()
    if store.source() is None:
        store.set_source(sources.SITE)
    if store.applied()[1] is None:
        models = watch_models.registered()
        clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
        store.apply(clean, by="setup_tables")
    store.session_key()
    print("The sw_ tables are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
