"""A store that looks live: one real check of a past event, moved so its last alert is now."""
# test_live_dashboard/run.py does the same with more checks

import time

from strawberrywatch import inventory as inv
from strawberrywatch.watch import runner, sources
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st

# Dusk Crayfish flags south_fork_1's conductivity. How long before the event's
# last reading the alert ends depends on the model, so the sample is moved by its
# last alert, and the readings after it are dropped
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
        last = db.execute("SELECT MAX(last_over) FROM noticed WHERE flagged = 1").fetchone()[0]
        delta = (now // STEP) * STEP - last
        for table, columns in TIMES.items():
            # Through negatives, to keep primary keys from colliding halfway
            away = ", ".join(f"{c} = -({c} + ?)" for c in columns)
            db.execute(f"UPDATE {table} SET {away}", [delta] * len(columns))
            db.execute(f"UPDATE {table} SET {', '.join(f'{c} = -{c}' for c in columns)}")
        for table in ("readings", "scores", "weather"):
            db.execute(f"DELETE FROM {table} WHERE time > ?", (now,))
        db.execute(
            "UPDATE checks SET source = ?, source_detail = NULL, started_at = ?, finished_at = ?,"
            " window_end = ?",
            (sources.SITE, now - 60, now - 30, (now // STEP) * STEP),
        )
        db.execute("UPDATE noticed SET created_at = first_over + ?", (STEP,))
        db.execute("UPDATE readings SET live = 1")
        db.execute("UPDATE source SET kind = ?, event = NULL", (sources.SITE,))
    return store
