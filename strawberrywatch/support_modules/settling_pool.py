"""
Settling Pool: flags readings the model should not treat as measurements.

It reads the inventory for thresholds and sensor state, calls the tests in
qc_tests.py, and returns flags. It corrects nothing and drops nothing. What a
consumer does with a flag is set by that model's FLAG_POLICY, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strawberrywatch import inventory as inv
from strawberrywatch.support_modules import qc_tests as qc
from strawberrywatch.support_modules.base import SupportScreen

# How an inventory state lands on a QARTOD level when no test has run yet.
# NOT_INSTALLED is UNKNOWN rather than MISSING on purpose: QARTOD's 2 means not
# evaluated, and a probe that was never in the creek has nothing to evaluate.
STATE_FLAGS = {
    inv.NOT_INSTALLED: qc.UNKNOWN,
    inv.PRESENT: qc.GOOD,
    inv.STALE: qc.SUSPECT,
    inv.MISSING: qc.MISSING,
}

# Which failure mode each test names, for the report and the audit record.
TEST_CAUSES = {
    "gross_range": "reading outside the range this site has ever produced",
    "flat_line": "sensor has stopped changing",
    "attenuated_signal": "variability collapsed, probe likely fouled or wrapped in debris",
    "spike": "single step excursion",
    "rate_of_change": "sustained change faster than the creek moves",
    "sentinel": "wrong SDI-12 channel address, field fixable",
    "duplicate_feed": "this site is reporting another site's numbers",
    "staleness": "no reading within the allowed gap",
}


def _series_for(raw, site, variable, index):
    """One site's variable reindexed onto the run's timestamps."""
    if raw is None or len(raw) == 0 or "location" not in raw.columns:
        return pd.Series(np.nan, index=index)
    rows = raw[raw["location"] == site]
    if len(rows) == 0 or variable not in rows.columns:
        return pd.Series(np.nan, index=index)
    values = pd.to_numeric(rows[variable], errors="coerce")
    values = values[~values.index.duplicated(keep="first")]
    values.index = pd.DatetimeIndex(pd.to_datetime(values.index, utc=True, errors="coerce"))
    return values.reindex(index)


def flag_one(values, timestamps, site, variable, inventory, other=None):
    """
    Run every test on one (site, variable) series and return flags plus detail.

    Returns (combined, {test name: flags}). Combined takes the worst level, so
    one failed test is not outvoted by six that passed.
    """
    entry = inventory.site(site)
    values = np.asarray(values, dtype=float)
    per_test = {}

    fail_range = entry.threshold(variable, "fail_range")
    suspect_range = entry.threshold(variable, "suspect_range")
    if fail_range or suspect_range:
        per_test["gross_range"] = qc.gross_range(values, fail_range, suspect_range)

    per_test["sentinel"] = qc.sentinel(values, inventory.sentinel_values)

    per_test["flat_line"] = qc.flat_line(
        values, inventory.flat_line_suspect_count, inventory.flat_line_fail_count
    )

    attenuation = entry.threshold(variable, "attenuation_std")
    if attenuation is not None:
        per_test["attenuated_signal"] = qc.attenuated_signal(
            values, inventory.window_steps, min_std=attenuation
        )

    spike = entry.threshold(variable, "spike")
    if spike is not None:
        per_test["spike"] = qc.spike(values, spike)

    roc = entry.threshold(variable, "rate_of_change")
    if roc is not None:
        per_test["rate_of_change"] = qc.rate_of_change(values, roc, inventory.window_steps)

    if other is not None:
        per_test["duplicate_feed"] = qc.duplicate_feed(
            values, other, inventory.duplicate_match_fraction
        )

    epoch = pd.DatetimeIndex(timestamps).as_unit("ns").asi8 / 1e9
    per_test["staleness"] = qc.staleness(epoch, ~np.isnan(values), inventory.staleness_hours * 3600)

    combined = qc.worst(*per_test.values())
    return combined, per_test


class SettlingPool(SupportScreen):
    """Flags readings that are not measurements, before the model reads them"""

    name = "settling_pool"

    def __init__(self, inventory=None):
        self.inventory = inventory or inv.load()

    def describe(self):
        return (
            "settling_pool: flags readings against the inventory's per site "
            "thresholds and sensor state (screen, no budget)"
        )

    def flags(self, window, nodes):
        """
        Flag every (site, variable, timestamp) the window covers.

        Returns {(site, variable): flag array}. Keyed by the pair and nothing
        else, so neither model's tensor layout reaches this module.
        """
        raw = (window or {}).get("raw")
        stamps = (window or {}).get("timestamps")
        if stamps is None:
            raise ValueError("settling_pool needs window['timestamps'] to flag anything")

        index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True, errors="coerce"))
        as_of = index.max() if len(index) else None

        out = {}
        for site in nodes:
            entry = self.inventory.site(site)
            for variable in entry.variables:
                values = _series_for(raw, site, variable, index)
                states = self.inventory.resolve_series(site, variable, values, as_of=as_of)

                combined, _per_test = flag_one(
                    values.to_numpy(), index, site, variable, self.inventory
                )

                # State outranks the tests where it says the reading was never a
                # measurement. A test on a probe that is not there means nothing.
                from_state = np.array([STATE_FLAGS[s] for s in states.to_numpy()], dtype=np.uint8)
                not_measured = np.isin(from_state, [qc.UNKNOWN, qc.MISSING, qc.SUSPECT]) & (
                    states.to_numpy() != inv.PRESENT
                )
                combined = combined.copy()
                combined[not_measured] = from_state[not_measured]

                out[(site, variable)] = combined
        return out

    def admit(self, window, nodes):
        """
        Per timestep per node bool of what the model may read.

        The reshape from (site, variable) flags to a node grid lives in
        models/contracts.py, because the shape is the model's business and the
        flags are not.

        A site switched off in the inventory is withheld outright, whatever its
        flags say. That is what taking a site out of service means, and doing it
        through the mask keeps the model's node count the same.
        """
        from strawberrywatch.models import contracts

        model = (window or {}).get("model")
        flags = self.flags(window, nodes)
        stamps = pd.DatetimeIndex(pd.to_datetime(window["timestamps"], utc=True, errors="coerce"))
        grid = contracts.admit_grid(flags, nodes, len(stamps), model)

        for j, node in enumerate(nodes):
            if node in self.inventory.sites and not self.inventory.site(node).in_service:
                grid[:, j] = False
        return grid


__all__ = ["STATE_FLAGS", "TEST_CAUSES", "SettlingPool", "flag_one"]
