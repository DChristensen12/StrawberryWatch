"""
Trial Bed: names the pollutant a firing looks like, from the way each channel moved.

Reads sensor state from the inventory, so a site that never had a pH probe and a
site whose pH probe died give different answers rather than the same silence.
An explainer, so it spends no false alarm budget and cannot change what fired.
"""

from __future__ import annotations

import pandas as pd

from strawberrywatch import inventory as inv
from strawberrywatch.support_modules import spill_signatures as sig
from strawberrywatch.support_modules.base import SupportExplainer

_LABELS = {
    "temperature": "temperature",
    "dissolved_oxygen": "dissolved oxygen",
    "ph": "pH",
    "conductivity": "conductivity",
    "floating_conductivity": "floating conductivity",
    "depth": "depth",
}

# FLAT means measured and did not move, which is evidence. Saying "flat" in the
# sentence reads as missing data, so it is spelled out.
_MOVE_WORDS = {sig.UP: "up", sig.DOWN: "down", sig.FLAT: "normal"}

DIAGNOSED = "diagnosed"
POSSIBLE_NEW_TYPE = "possible_new_type"
CANNOT_EVALUATE = "cannot_evaluate"
NOTHING_TO_EXPLAIN = "nothing_to_explain"

# Why a parameter has no direction. The first two look identical in the data.
NEVER_INSTALLED = "never_installed"
INSTALLED_NO_DATA = "installed_but_silent"
NOT_IN_WINDOW = "no_readings_in_window"


def observed_direction(baseline_vals, event_vals):
    """
    Report whether a parameter went UP, DOWN or FLAT from baseline to event.

    The baseline's own spread is the yardstick, so temperature and conductivity
    are judged on their own noise. None where there is not enough data.
    """
    b = pd.to_numeric(pd.Series(baseline_vals), errors="coerce").dropna()
    e = pd.to_numeric(pd.Series(event_vals), errors="coerce").dropna()
    if len(b) < 2 or len(e) < 1:
        return None

    baseline_mean = b.mean()
    baseline_std = b.std()
    event_mean = e.mean()

    if pd.isna(baseline_std) or baseline_std == 0:
        if baseline_mean == 0:
            return sig.FLAT if event_mean == 0 else (sig.UP if event_mean > 0 else sig.DOWN)
        rel = (event_mean - baseline_mean) / abs(baseline_mean)
        if abs(rel) < 0.05:
            return sig.FLAT
        return sig.UP if rel > 0 else sig.DOWN

    shift = (event_mean - baseline_mean) / baseline_std
    if abs(shift) < sig.CHANGE_THRESHOLD_STD:
        return sig.FLAT
    return sig.UP if shift > 0 else sig.DOWN


def observed_directions(baseline_df, event_df):
    """Which way every signature parameter both frames carry actually moved."""
    observed = {}
    for col in event_df.columns:
        param = sig.resolve_parameter(col)
        if param is None or param not in sig.PARAMETERS:
            continue
        if col not in baseline_df.columns:
            continue
        moved = observed_direction(baseline_df[col].values, event_df[col].values)
        if moved is not None:
            observed[param] = moved
    return observed


def event_means(event_df):
    """Event mean per signature parameter, for the numeric tapwater checks."""
    means = {}
    for col in event_df.columns:
        param = sig.resolve_parameter(col)
        if param is None:
            continue
        values = pd.to_numeric(event_df[col], errors="coerce").dropna()
        if len(values):
            means[param] = float(values.mean())
    return means


def absence_reasons(site, observed, inventory=None):
    """
    Why each unobserved parameter is unobserved, from the inventory.

    Never fitted and fitted but silent are the same silence in the data and
    different problems for whoever gets the alert.
    """
    inventory = inventory or inv.load()
    try:
        entry = inventory.site(site)
    except inv.InventoryError:
        return {p: NOT_IN_WINDOW for p in sig.PARAMETERS if p not in observed}

    reasons = {}
    for param in sig.PARAMETERS:
        if param in observed:
            continue
        sensor = entry.sensor_for(param)
        if sensor is None or not sensor.ever_installed:
            reasons[param] = NEVER_INSTALLED
        else:
            reasons[param] = INSTALLED_NO_DATA
    return reasons


def _absence_phrase(param, reason):
    label = _LABELS.get(param, param)
    if reason == NEVER_INSTALLED:
        return f"{label} never installed here"
    if reason == INSTALLED_NO_DATA:
        return f"{label} installed but reporting nothing"
    return f"{label} not measured in this window"


def _evidence_sentence(observed, reasons):
    """Every parameter the table reads, with how it moved or why it is absent."""
    parts = []
    for param in sig.PARAMETERS:
        if param in observed:
            parts.append(f"{_LABELS.get(param, param)} {_MOVE_WORDS[observed[param]]}")
        else:
            parts.append(_absence_phrase(param, reasons.get(param, NOT_IN_WINDOW)))
    return ", ".join(parts)


def _confidence(discriminating_available):
    """How well the populated channels separate one pollutant from another."""
    if not discriminating_available:
        return "none (no discriminating channels populated)"
    if len(discriminating_available) == 1:
        return f"moderate (one discriminating channel: {discriminating_available[0]})"
    return f"good ({len(discriminating_available)} discriminating channels)"


def _score_row(pollutant, observed, table, means=None):
    """Score one signature row against the observed directions, per parameter."""
    comparable = 0
    agreements = 0
    per_param = {}
    for param in sig.PARAMETERS:
        expected = sig.direction(pollutant, param, table)
        if expected == sig.INDET:
            per_param[param] = "skipped (not diagnostic)"
            continue
        comparable += 1
        if observed[param] == expected:
            agreements += 1
            per_param[param] = (
                f"match ({observed[param]}): {sig.mechanism(pollutant, param, table)}"
            )
        else:
            per_param[param] = (
                f"differ (saw {observed[param]}, expected {expected}): "
                f"{sig.mechanism(pollutant, param, table)}"
            )

    row = {
        "pollutant": pollutant,
        "score": (agreements / comparable) if comparable else 0.0,
        "agreements": agreements,
        "comparable": comparable,
        "per_param": per_param,
    }
    if pollutant == "tapwater" and means:
        row["numeric"] = _tapwater_numbers(means)
    return row


def _tapwater_numbers(means):
    """Check the event means against Berkeley tap water rather than a direction."""
    out = {}
    temp = means.get("temperature")
    if temp is not None:
        out["temperature_c"] = temp
        out["matches_tap_temperature"] = sig.looks_like_tapwater_temperature(temp)
    ph = means.get("ph")
    if ph is not None:
        out["ph"] = ph
        out["matches_tap_ph"] = sig.looks_like_tapwater_ph(ph)
    return out


def evaluable_signatures(site, inventory=None, table=None):
    """
    Which signature rows this site could evaluate if every fitted probe reported.

    A row needing pH is not evaluable where no pH probe exists.
    """
    inventory = inventory or inv.load()
    rows = sig.SIGNATURES if table is None else table
    entry = inventory.site(site)

    available = []
    for param in sig.PARAMETERS:
        sensor = entry.sensor_for(param)
        if sensor is not None and sensor.ever_installed:
            available.append(param)
    return sig.evaluable_on(available, rows)


def classify(observed, table=None, reasons=None, means=None):
    """
    Match observed directions against every signature row that can be judged.

    A row needing pH is reported unevaluable rather than scored on the rest and
    called a partial match.
    """
    table = sig.SIGNATURES if table is None else table
    reasons = reasons or {}
    available = sorted(observed)
    discriminating = [p for p in available if p in sig.DISCRIMINATING]

    evaluable = sig.evaluable_on(available, table)
    unevaluable = {
        pollutant: sig.missing_parameters(pollutant, observed, table)
        for pollutant in table
        if pollutant not in evaluable
    }

    ranked = [_score_row(p, observed, table, means) for p in evaluable]
    ranked.sort(key=lambda r: (r["score"], r["comparable"]), reverse=True)

    top_score = ranked[0]["score"] if ranked else 0.0
    tied = [r["pollutant"] for r in ranked if r["score"] == top_score and top_score > 0]

    if not ranked or len(discriminating) < sig.MIN_DISCRIMINATING_FOR_DIAGNOSIS:
        verdict = CANNOT_EVALUATE
        cause = None
        absent = sorted({p for missing in unevaluable.values() for p in missing})
        detail = ", ".join(_absence_phrase(p, reasons.get(p, NOT_IN_WINDOW)) for p in absent)
        note = (
            f"No signature can be evaluated on the channels that reported. "
            f"{len(evaluable)} of {len(table)} rows are judgeable; the rest need "
            f"{detail or 'channels that are absent'}."
        )
    elif top_score < sig.NEW_TYPE_SCORE_FLOOR:
        verdict = POSSIBLE_NEW_TYPE
        cause = None
        note = (
            f"Matches no known signature well (best score {top_score:.2f}, below "
            f"{sig.NEW_TYPE_SCORE_FLOOR:.2f}). May be a new type and is worth a look."
        )
    elif len(tied) > 1:
        verdict = POSSIBLE_NEW_TYPE
        cause = None
        note = (
            f"Unresolved tie between {', '.join(tied)}, all at score {top_score:.2f}. "
            f"The reporting channels did not separate them, so none is named."
        )
    else:
        verdict = DIAGNOSED
        cause = ranked[0]["pollutant"]
        note = (
            f"Best match is {cause} (score {top_score:.2f}, "
            f"{ranked[0]['agreements']}/{ranked[0]['comparable']} parameters agreed)."
        )

    evidence = _evidence_sentence(observed, reasons)
    if verdict == DIAGNOSED:
        explanation = f"consistent with {cause}: {evidence}"
    elif verdict == POSSIBLE_NEW_TYPE:
        explanation = f"consistent with no known type: {evidence}"
    else:
        explanation = f"cannot evaluate: {evidence}"

    return {
        "verdict": verdict,
        "cause": cause,
        "confidence": _confidence(discriminating),
        "evidence": dict(observed),
        "absence": dict(reasons),
        "explanation": explanation,
        "note": note,
        "ranked": ranked,
        "unevaluable": unevaluable,
        "available_parameters": available,
        "discriminating_available": discriminating,
        "top_candidates": tied,
    }


def cannot_evaluate(reason, observed=None, reasons=None):
    """A result that names what was missing instead of guessing past it."""
    observed = observed or {}
    reasons = reasons or {}
    discriminating = [p for p in sorted(observed) if p in sig.DISCRIMINATING]
    return {
        "verdict": CANNOT_EVALUATE,
        "cause": None,
        "confidence": _confidence(discriminating),
        "evidence": dict(observed),
        "absence": dict(reasons),
        "explanation": f"cannot evaluate: {reason}",
        "note": reason,
        "ranked": [],
        "unevaluable": {p: sig.required_parameters(p) for p in sig.SIGNATURES},
        "available_parameters": sorted(observed),
        "discriminating_available": discriminating,
        "top_candidates": [],
    }


class TrialBed(SupportExplainer):
    """Names the pollutant a firing looks like, from the way each channel moved"""

    name = "trial_bed"

    def __init__(self, inventory=None):
        self.inventory = inventory or inv.load()

    def describe(self):
        return (
            "trial_bed: matches the direction each channel moved against the "
            "spill signature table (explainer, no budget)"
        )

    def evaluable_per_site(self, sites=None):
        """How many of the five signatures each site could evaluate today."""
        sites = sites or self.inventory.tables
        return {
            site: evaluable_signatures(site, self.inventory)
            for site in sites
            if site in self.inventory.sites
        }

    def explain(self, window, nodes, fired):
        """One account per node, built from the readings behind its firings."""
        raw = (window or {}).get("raw")
        timestamps = (window or {}).get("timestamps")

        if raw is None or timestamps is None or len(raw) == 0:
            return [cannot_evaluate("no readings in the window") for _ in nodes]

        times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce"))
        return [self._explain_node(raw, times, node, fired[:, j]) for j, node in enumerate(nodes)]

    def _explain_node(self, raw, times, node, fired_column):
        """Split one node's readings on its firings, then read the directions."""
        rows = raw[raw["location"] == node] if "location" in raw.columns else raw
        if len(rows) == 0:
            return cannot_evaluate(
                f"{node} reported no readings in this window",
                reasons=absence_reasons(node, {}, self.inventory),
            )

        if not fired_column.any():
            result = cannot_evaluate("nothing fired at this node")
            result["verdict"] = NOTHING_TO_EXPLAIN
            result["explanation"] = "nothing fired at this node, so there is nothing to explain"
            return result

        # The reading times, not the window indices. The record is per sequence
        # and the raw frame is per sample, so timestamps are what join them.
        event_times = times[fired_column]
        index = pd.DatetimeIndex(pd.to_datetime(rows.index, utc=True, errors="coerce"))
        is_event = index.isin(event_times)

        event = rows[is_event]
        baseline = rows[~is_event]
        if len(event) == 0:
            return cannot_evaluate(f"{node} has no readings at the timesteps that fired")
        if len(baseline) < 2:
            return cannot_evaluate(f"{node} has too few unflagged readings for a baseline")

        observed = observed_directions(baseline, event)
        return classify(
            observed,
            reasons=absence_reasons(node, observed, self.inventory),
            means=event_means(event),
        )


__all__ = [
    "CANNOT_EVALUATE",
    "DIAGNOSED",
    "INSTALLED_NO_DATA",
    "NEVER_INSTALLED",
    "NOTHING_TO_EXPLAIN",
    "NOT_IN_WINDOW",
    "POSSIBLE_NEW_TYPE",
    "TrialBed",
    "absence_reasons",
    "cannot_evaluate",
    "classify",
    "evaluable_signatures",
    "event_means",
    "observed_direction",
    "observed_directions",
]
