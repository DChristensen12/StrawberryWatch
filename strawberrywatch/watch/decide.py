"""
Deciding what a check noticed, and what the last few days would have noticed at
other settings. One set of rules for both, so the preview on Advanced settings
says what checks would really have done.

A (model, site, variable, rule) is noticed in a check when at least min_steps of
the steps it scored in that check's window are over threshold * scale. That is
serving.notice() over a window, done on arrays so the preview can replay a week
of checks quickly. The same thing stays one item: while an event is in the
window its first step over doesn't move. A new crossing soon after an item is a
repeat and is ignored for repeat_hours.
"""

from __future__ import annotations

import numpy as np

NEW, SAME, REPEAT = "new", "same", "repeat"


def series(rows):
    """
    Scores grouped into arrays, {(site, variable, rule): (time, score, threshold, base)},
    each sorted by time. rows are serving Scores or the store's score dicts.
    """
    grouped = {}
    for row in rows:
        if isinstance(row, dict):
            key = (row["site"], row["variable"], row["rule"])
            values = (row["time"], row["score"], row["threshold"], row["base"])
        else:
            key = (row.site, row.variable, row.rule)
            values = (row.time.timestamp(), row.score, row.threshold, row.base)
        grouped.setdefault(key, []).append(values)
    out = {}
    for key, values in grouped.items():
        values.sort(key=lambda v: v[0])
        arr = np.array(values, dtype=float)
        out[key] = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    return out


def notice_window(grouped, start, end, scales, min_steps):
    """
    What one check notices in [start, end]. scales is {rule: multiple of the line}.

    Returns [{site, variable, rule, first_over, last_over, peak, base, scale, steps_over}].
    """
    out = []
    for (site, variable, rule), (times, scores, lines, bases) in grouped.items():
        inside = (times >= start) & (times <= end)
        if not inside.any():
            continue
        scale = float(scales.get(rule, 1.0))
        values, limit = scores[inside], lines[inside] * scale
        over = np.isfinite(values) & (values > limit)
        count = int(over.sum())
        if count < max(1, min_steps):
            continue
        hits = np.flatnonzero(over)
        stamps = times[inside]
        out.append(
            {
                "site": site,
                "variable": variable,
                "rule": rule,
                "first_over": float(stamps[hits[0]]),
                "last_over": float(stamps[hits[-1]]),
                "peak": float(np.nanmax(values)),
                "base": float(bases[inside][hits[0]]),
                "scale": scale,
                "steps_over": count,
            }
        )
    return out


def over_steps(grouped, start, end, scales):
    """Every step over its scaled line in [start, end], as {(site, variable, rule): times}."""
    out = {}
    for key, (times, scores, lines, _) in grouped.items():
        inside = (times >= start) & (times <= end)
        values, limit = scores[inside], lines[inside] * float(scales.get(key[2], 1.0))
        over = np.isfinite(values) & (values > limit)
        out[key] = times[inside][over]
    return out


def classify(item, previous, checked_at, repeat_hours):
    """
    new, same or repeat, against the last item recorded for the same key.

    previous is a dict with first_over and checked_at, or None.
    """
    if previous is None:
        return NEW
    if abs(previous["first_over"] - item["first_over"]) < 1:
        return SAME
    if checked_at - previous["checked_at"] < repeat_hours * 3600:
        return REPEAT
    return NEW


def replay(grouped_by_model, checks, settings, min_steps, seed=None):
    """
    Walk a run of checks with one set of settings and say what each would have
    noticed and kept.

    grouped_by_model is {model: series(...)} from stored scores. checks are the
    store's rows, in order, each with window_start, window_end and started_at.
    seed is {(model, site, variable, rule): previous} from before the first
    check. Returns [(check, model, item, flagged)] for the new items only.
    """
    last = dict(seed or {})
    kept = []
    dont_flag = set(settings.get("dont_flag", []))
    for check in checks:
        for model, grouped in grouped_by_model.items():
            spec = settings["models"].get(model)
            if not spec or not spec.get("on"):
                continue
            items = notice_window(
                grouped,
                check["window_start"],
                check["window_end"],
                spec["scales"],
                min_steps[model],
            )
            for item in items:
                key = (model, item["site"], item["variable"], item["rule"])
                outcome = classify(
                    item, last.get(key), check["started_at"], settings["repeat_hours"]
                )
                if outcome == NEW:
                    last[key] = {
                        "first_over": item["first_over"],
                        "checked_at": check["started_at"],
                    }
                    kept.append((check, model, item, item["site"] not in dont_flag))
    return kept
