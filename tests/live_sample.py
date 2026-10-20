"""
A store that looks live, for pages that only show live data. One real check of
a past event, moved so its last reading is now. test_live_dashboard/run.py does
the same with more checks.
"""

import time

from strawberrywatch import inventory as inv
from strawberrywatch.watch import runner, sources
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st

# Dusk Crayfish flags south_fork_1's conductivity, and it lasts until 17 hours
# before the event's last reading. Moved to now, it shows on Overview
EVENT = "anomaly_2025_09_10_overnight_sf"
STEP = 900
TIMES = {
    "checks": ("started_at", "finished_at", "window_start", "window_end"),
    "readings": ("time",),
    "scores": ("time",),
    "weather": ("time",),
    "noticed": ("first_over", "last_over", "created_at"),
}


def live_store(folder, models, event=EVENT):
    store = st.Store(folder)
    store.set_source(sources.REPLAY, event=event)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    runner.run_check(store, models)
    now = time.time()
    with store.db() as db:
        check = db.execute("SELECT * FROM checks ORDER BY id DESC LIMIT 1").fetchone()
        delta = (now // STEP) * STEP - check["window_end"]
        for table, columns in TIMES.items():
            # Through negatives, to keep primary keys from colliding halfway
            away = ", ".join(f"{c} = -({c} + ?)" for c in columns)
            db.execute(f"UPDATE {table} SET {away}", [delta] * len(columns))
            db.execute(f"UPDATE {table} SET {', '.join(f'{c} = -{c}' for c in columns)}")
        db.execute(
            "UPDATE checks SET source = ?, source_detail = NULL, started_at = ?, finished_at = ?",
            (sources.SITE, now - 60, now - 30),
        )
        db.execute("UPDATE noticed SET created_at = first_over + ?", (STEP,))
        db.execute("UPDATE readings SET live = 1")
        db.execute("UPDATE source SET kind = ?, event = NULL", (sources.SITE,))
    return store
