"""
Runs one website check in its own process:

    python -m strawberrywatch.watch.check_once

Reads the dashboard's settings from the website's database, checks the sensor
tables, saves the results to the sw_ tables, emails the list about new
alerts, and prints one line of JSON with them. A crash or out-of-memory
kill only ends this check.
"""

from __future__ import annotations

import json
import logging
import resource
import sys
import time

logger = logging.getLogger(__name__)

# Readings sent with each alert, for the email's chart
ALERT_READINGS = 200
# When each model may alert again, kept per model the way Night Heron keeps
# reactivate_at per rule
QUIET_UNTIL = "alerts_quiet_until:{}"


def peak_mb():
    """This process's highest memory use so far. Linux counts ru_maxrss in kilobytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sendable(store, check_id, clock=time.time):
    """
    This check's new alerts from models that may send one now. A model that
    sends starts its minutes between alerts, and until they're up its new
    alerts are held: they stay on the dashboard but nobody is emailed or texted.
    """
    from strawberrywatch.watch import settings as ws

    now = clock()
    applied = store.applied()[1] or {"models": {}}
    items = [n for n in store.noticed(flagged=True) if n["check_id"] == check_id]
    out = []
    for model in sorted({n["model"] for n in items}):
        mine = [n for n in items if n["model"] == model]
        try:
            until = float(store.pref(QUIET_UNTIL.format(model)) or 0)
        except ValueError:
            until = 0.0
        if now < until:
            logger.info(
                "%s: %d new alert(s) held, quiet for %.0f more minutes",
                model,
                len(mine),
                (until - now) / 60,
            )
            continue
        spec = applied["models"].get(model, {})
        gap = float(spec.get("alert_gap_minutes", ws.ALERT_GAP_DEFAULT))
        store.set_pref(QUIET_UNTIL.format(model), str(now + gap * 60))
        out += mine
    return out


def alerts(store, items):
    """The alerts to send, each with its recent readings, newest last."""
    out = []
    for item in items:
        rows = store.readings(item["site"], since=item["last_over"] - 2 * 86400, live=True)
        values = [r[item["variable"]] for r in rows if r.get(item["variable"]) is not None]
        out.append(
            {
                "model": item["model"],
                "site": item["site"],
                "variable": item["variable"],
                "rule": item["rule"],
                "first_over": item["first_over"],
                "values": values[-ALERT_READINGS:],
            }
        )
    return out


def main(store=None, models=None):
    from strawberrywatch.watch import models as watch_models
    from strawberrywatch.watch import runner, sources
    from strawberrywatch.watch import settings as ws
    from strawberrywatch.watch.store_mysql import MysqlStore

    began = time.time()
    try:
        import torch

        # Shared server. One thread, same as gnn_alerts
        torch.set_num_threads(1)
    except ImportError:
        pass
    store = store or MysqlStore()
    models = models if models is not None else watch_models.registered()
    if (store.source() or {}).get("kind") != sources.SITE:
        store.set_source(sources.SITE)
    limits = ws.website_limits()
    check_id = runner.run_check(
        store, models, memory=peak_mb, max_models=int(limits["max_models_on"])
    )
    check = store.last_check() if check_id is not None else None
    on = store.pref("email_on") == "yes"
    items = sendable(store, check_id) if check_id is not None else []
    if items:
        from strawberrywatch import inventory as inv
        from strawberrywatch.dashboard import mail

        mail.website_alert(store, inv.load(), check_id, ws.website_address(), items)
    result = {
        "check_id": check_id,
        "ok": bool(check and check.get("ok")),
        "message": (check or {}).get("message") or "",
        "seconds": round(time.time() - began, 1),
        "peak_mb": round(peak_mb(), 1),
        # Only what passed each model's minutes between alerts, so Night Heron's
        # texts keep the same pace as the emails
        "alerts": alerts(store, items),
        "email_on": on,
    }
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    raise SystemExit(main())
