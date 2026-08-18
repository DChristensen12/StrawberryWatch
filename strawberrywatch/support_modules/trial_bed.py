"""Trial Bed: names the pollutant a firing looks like, from the way each channel moved."""

from __future__ import annotations

import pandas as pd

from strawberrywatch.support_modules import spill_signatures as sig
from strawberrywatch.support_modules.base import SupportExplainer

# How each parameter is written in a sentence an operator reads.
_LABELS = {
    "temperature": "temperature",
    "dissolved_oxygen": "dissolved oxygen",
    "ph": "pH",
    "conductivity": "conductivity",
    "floating_conductivity": "floating conductivity",
    "depth": "depth",
}

# FLAT means the channel was measured and did not move, which is evidence.
# Saying "flat" in the sentence reads as missing data, so it is spelled out.
_MOVE_WORDS = {sig.UP: "up", sig.DOWN: "down", sig.FLAT: "normal"}

DIAGNOSED = "diagnosed"
POSSIBLE_NEW_TYPE = "possible_new_type"
CANNOT_EVALUATE = "cannot_evaluate"
NOTHING_TO_EXPLAIN = "nothing_to_explain"


def observed_direction(baseline_vals, event_vals):
    """
    Report whether a parameter went UP, DOWN, or FLAT from baseline to event.

    Uses the baseline's own variability as the yardstick: the event mean has to move
    more than CHANGE_THRESHOLD_STD standard deviations to count as a real move. This
    adapts to each parameter's natural noise rather than hardcoding one threshold for
    wildly different scales like temperature and conductivity.

    Returns UP, DOWN, FLAT, or None if there's not enough data.
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
        direction = observed_direction(baseline_df[col].values, event_df[col].values)
        if direction is not None:
            observed[param] = direction
    return observed


def _evidence_sentence(observed):
    """Every parameter the table reads, with how it moved or that it is absent."""
    parts = []
    for param in sig.PARAMETERS:
        label = _LABELS.get(param, param)
        if param in observed:
            parts.append(f"{label} {_MOVE_WORDS[observed[param]]}")
        else:
            parts.append(f"{label} not measured")
    return ", ".join(parts)


def _confidence(discriminating_available):
    """How much the populated channels can separate one pollutant from another."""
    if not discriminating_available:
        return "none (no discriminating channels populated)"
    if len(discriminating_available) == 1:
        return f"moderate (one discriminating channel: {discriminating_available[0]})"
    return f"good ({len(discriminating_available)} discriminating channels)"


def _score_row(pollutant, observed, table):
    """One signature row against the observed directions, per parameter."""
    comparable = 0
    agreements = 0
    per_param = {}
    for param in sig.PARAMETERS:
        sig_dir, _sig_mag = table[pollutant].get(param, (sig.INDET, sig.NORMAL))
        if sig_dir == sig.INDET:
            per_param[param] = "skipped (not diagnostic)"
            continue
        comparable += 1
        if observed[param] == sig_dir:
            agreements += 1
            per_param[param] = f"match ({observed[param]})"
        else:
            per_param[param] = f"differ (saw {observed[param]}, expected {sig_dir})"
    return {
        "pollutant": pollutant,
        "score": (agreements / comparable) if comparable else 0.0,
        "agreements": agreements,
        "comparable": comparable,
        "per_param": per_param,
    }


def classify(observed, table=None):
    """
    Match observed directions against every signature row that can be judged.

    A row is judged only when every direction it asserts has a measurement
    behind it. A row needing pH is not scored on the rest and called a partial
    match; it is reported as unevaluable, naming the channels that were absent.
    """
    table = sig.SIGNATURES if table is None else table
    available = sorted(observed)
    discriminating = [p for p in available if p in sig.DISCRIMINATING]

    evaluable = sig.evaluable_on(available, table)
    unevaluable = {
        pollutant: sig.missing_parameters(pollutant, observed, table)
        for pollutant in table
        if pollutant not in evaluable
    }

    ranked = [_score_row(p, observed, table) for p in evaluable]
    ranked.sort(key=lambda r: (r["score"], r["comparable"]), reverse=True)

    top_score = ranked[0]["score"] if ranked else 0.0
    tied = [r["pollutant"] for r in ranked if r["score"] == top_score and top_score > 0]

    if not ranked or len(discriminating) < sig.MIN_DISCRIMINATING_FOR_DIAGNOSIS:
        verdict = CANNOT_EVALUATE
        cause = None
        absent = sorted({p for missing in unevaluable.values() for p in missing})
        note = (
            f"No signature can be evaluated on the channels that reported. "
            f"{len(evaluable)} of {len(table)} rows are judgeable; the rest need "
            f"{', '.join(_LABELS.get(p, p) for p in absent) or 'channels that are absent'}."
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

    evidence = _evidence_sentence(observed)
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
        "explanation": explanation,
        "note": note,
        "ranked": ranked,
        "unevaluable": unevaluable,
        "available_parameters": available,
        "discriminating_available": discriminating,
        "top_candidates": tied,
    }


def cannot_evaluate(reason, observed=None):
    """A result that names what was missing instead of guessing past it."""
    observed = observed or {}
    discriminating = [p for p in sorted(observed) if p in sig.DISCRIMINATING]
    return {
        "verdict": CANNOT_EVALUATE,
        "cause": None,
        "confidence": _confidence(discriminating),
        "evidence": dict(observed),
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

    def describe(self):
        return (
            "trial_bed: matches the direction each channel moved against the "
            "spill signature table (explainer, no budget)"
        )

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
            return cannot_evaluate(f"{node} reported no readings in this window")

        if not fired_column.any():
            result = cannot_evaluate("nothing fired at this node")
            result["verdict"] = NOTHING_TO_EXPLAIN
            result["explanation"] = "nothing fired at this node, so there is nothing to explain"
            return result

        # The reading times, not the window indices: the record is per sequence
        # and the raw frame is per sample, so the timestamps are what join them.
        event_times = times[fired_column]
        # Both sides through the same conversion, so a naive raw index and an
        # aware timestamp list still compare rather than silently missing.
        index = pd.DatetimeIndex(pd.to_datetime(rows.index, utc=True, errors="coerce"))
        is_event = index.isin(event_times)

        event = rows[is_event]
        baseline = rows[~is_event]
        if len(event) == 0:
            return cannot_evaluate(f"{node} has no readings at the timesteps that fired")
        if len(baseline) < 2:
            return cannot_evaluate(f"{node} has too few unflagged readings for a baseline")

        return classify(observed_directions(baseline, event))


__all__ = [
    "CANNOT_EVALUATE",
    "DIAGNOSED",
    "NOTHING_TO_EXPLAIN",
    "POSSIBLE_NEW_TYPE",
    "TrialBed",
    "cannot_evaluate",
    "classify",
    "observed_direction",
    "observed_directions",
]
