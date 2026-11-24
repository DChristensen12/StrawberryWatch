"""Scoring against labelled events, each counted once, without point adjustment (Kim et al. 2022)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def events(flags, gap=0):
    """Numbers each run of flagged steps 1, 2, 3, runs closer than gap steps counted as one."""
    flags = np.asarray(flags, dtype=bool)
    out = np.zeros(len(flags), dtype=int)
    hits = np.flatnonzero(flags)
    if hits.size == 0:
        return out
    breaks = np.flatnonzero(np.diff(hits) > gap + 1) + 1
    for number, run in enumerate(np.split(hits, breaks), start=1):
        out[run[0] : run[-1] + 1] = number
    return out


def compare_events(labelled, detected):
    """tp, fp, fn or tn per step. Overlapping labelled and detected events make one tp."""
    labelled = np.asarray(labelled)
    detected = np.asarray(detected)
    group = np.zeros(len(labelled), dtype=int)
    index = 0
    for i in range(1, len(labelled)):
        was = (labelled[i - 1], detected[i - 1])
        now = (labelled[i], detected[i])
        if was != now:
            # leaving quiet, entering quiet, or both changing at once
            if was == (0, 0) or now == (0, 0) or (was[0] != now[0] and was[1] != now[1]):
                index += 1
        group[i] = index

    out = np.full(len(labelled), "tn", dtype=object)
    for g in np.unique(group):
        rows = group == g
        has_label, has_detection = labelled[rows].any(), detected[rows].any()
        if has_label and has_detection:
            out[rows] = "tp"
        elif has_detection:
            out[rows] = "fp"
        elif has_label:
            out[rows] = "fn"
    return out


def event_metrics(classes):
    """Counts of tp, fp and fn events, with precision and recall. Every event weighs the same."""
    classes = np.asarray(classes, dtype=object)
    starts = np.concatenate([[True], classes[1:] != classes[:-1]])
    counts = {kind: int((classes[starts] == kind).sum()) for kind in ("tp", "fp", "fn")}
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
    }


def caught(times, over, start, end):
    """(caught, minutes from start to the first step over the line between start and end)."""
    times = pd.DatetimeIndex(times)
    inside = np.asarray((times >= start) & (times <= end)) & np.asarray(over, dtype=bool)
    if not inside.any():
        return False, None
    first = times[np.flatnonzero(inside)[0]]
    return True, (first - start) / pd.Timedelta(minutes=1)


def false_alarms_per_week(over, clean, step_minutes=15, gap=0):
    """False alarm events per week of clean steps, runs closer than gap counted once."""
    over = np.asarray(over, dtype=bool) & np.asarray(clean, dtype=bool)
    weeks = np.asarray(clean, dtype=bool).sum() * step_minutes / (7 * 24 * 60)
    if weeks == 0:
        return np.nan
    return int(events(over, gap).max()) / weeks


def range_scores(labelled, detected):
    """Range-based (precision, recall) of Tatbul et al. 2018, from prts. NaN where undefined."""
    from prts import ts_precision, ts_recall

    labelled = np.asarray(labelled, dtype=bool).astype(int)
    detected = np.asarray(detected, dtype=bool).astype(int)
    if not labelled.any() or not detected.any():
        return np.nan, np.nan
    return (
        float(ts_precision(labelled, detected, alpha=0.0, cardinality="one", bias="flat")),
        float(ts_recall(labelled, detected, alpha=0.0, cardinality="one", bias="flat")),
    )


def score(times, over, spans, step_minutes=15, gap=0):
    """Per-event rows, event counts, false alarms per week and range scores for one detector."""
    times = pd.DatetimeIndex(times)
    over = np.asarray(over, dtype=bool)
    labelled = np.zeros(len(times), dtype=bool)
    rows = []
    for start, end in spans:
        labelled |= np.asarray((times >= start) & (times <= end))
        hit, minutes = caught(times, over, start, end)
        rows.append({"start": start, "end": end, "caught": hit, "minutes": minutes})
    classes = compare_events(events(labelled), events(over, gap))
    precision, recall = range_scores(labelled, over)
    return {
        "events": rows,
        **event_metrics(classes),
        "false_alarms_per_week": false_alarms_per_week(over, ~labelled, step_minutes, gap),
        "range_precision": precision,
        "range_recall": recall,
    }


__all__ = [
    "caught",
    "compare_events",
    "event_metrics",
    "events",
    "false_alarms_per_week",
    "range_scores",
    "score",
]
