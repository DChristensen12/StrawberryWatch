"""Re-fit adapters for Dusk Crayfish and Cobble Shoal, the past-events check, and drift references."""

from __future__ import annotations

import copy
import functools
import json
import threading

import numpy as np
import pandas as pd

from strawberrywatch.workbench import checks, data, events, store, text

MODELS = ("dusk_crayfish", "cobble_shoal")
FILENAMES = {
    "dusk_crayfish": "dusk_crayfish_serving.json",
    "cobble_shoal": "cobble_shoal_calibration_real.json",
}
# Alerts a day for each false alarm choice
RATES = {"month": 1 / 30, "week": 1 / 7, "day": 1.0}
CONTEXT = pd.Timedelta(days=2)
# The dashboard's defaults: a check every 15 minutes over the last 2 days, repeats ignored for 6 hours
CHECK_EVERY = 900
LOOKBACK = 2 * 86400
REPEAT_HOURS = 6.0
# Fault-free windows the Cobble Shoal nulls are fitted on, as scripts/refit_cobble_calibration.py does
NULL_ANCHORS = 400
# A site needs this many readings left in before its Dusk Crayfish line is re-fitted
MIN_FIT = 96

# One model scores at a time. torch and the CPU are shared
_lock = threading.Lock()


@functools.cache
def info(model):
    from strawberrywatch.watch import models as wm

    return next(m for m in wm.registered() if m.name == model)


def title(model):
    return info(model).title


def model_sites(model):
    return list(info(model).sites)


def per_step_rate(model, choice):
    """The false alarm choice as a per-step rate, shared out over the model's sites or sensors."""
    keys = len(model_sites(model)) if model == "dusk_crayfish" else len(info(model).nodes)
    return RATES[choice] / (data.PER_DAY * keys)


def shipped_path(model):
    from strawberrywatch import paths

    return paths.model_dir(model) / FILENAMES[model]


def blob(folder, model, calibration_id):
    """The calibration file's contents, shipped or saved."""
    if calibration_id == store.SHIPPED:
        return json.loads(shipped_path(model).read_text())
    path = store.verified(folder, model, calibration_id)
    if path is None:
        return json.loads(shipped_path(model).read_text())
    return json.loads(path.read_text())


def in_use(folder, model):
    return store.in_use(folder).get(model, store.SHIPPED)


@functools.cache
def detector(model):
    from strawberrywatch import paths
    from strawberrywatch.serving import detector_class

    return detector_class(model).load(paths.model_dir(model), device="cpu")


def _pot_line(values, q):
    from strawberrywatch.anomalies import channel_scoring

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fit = channel_scoring.pot_diagnostics(values, q=q)
    line = fit.get("threshold")
    if line is None or not np.isfinite(line):
        line = float(values.max()) if values.size else float("nan")
    return float(line), fit


# Dusk Crayfish


def dusk_pass(tables, weather):
    """One scoring run, kept as raw errors and levels so any calibration can judge it. {site: arrays}."""
    import torch

    det = detector("dusk_crayfish")
    with _lock, torch.no_grad():
        judged = det._judge(tables, weather)
    if judged is None:
        return {}
    frame, result = judged
    stamps = pd.DatetimeIndex(result["timestamps"])
    cal = det.checkpoint.calibration
    mean, scale = det._normalization["conductivity"]
    out = {}
    for site, verdict in result["verdicts"].items():
        if not verdict.get("judged"):
            continue
        times = stamps[verdict["real_timesteps"]]
        rule1 = verdict["rule1"]
        error = np.asarray(rule1["score"]) * cal["error_iqr"][site] + cal["error_median"][site]
        rain = np.asarray(rule1["threshold"]) / cal["node_thresholds"][site]
        cond = frame.loc[frame["location"] == site, "conductivity"]
        cond = cond[~cond.index.duplicated()].reindex(times)
        out[site] = {
            "times": times,
            "error": error,
            "rain": rain,
            "level": (cond.to_numpy(dtype=float) - mean) / scale,
            "k": det.bar("level_shift", site),
        }
    return out


def dusk_rows(scored, calibration):
    """{(site, variable, rule): (times, scores, lines, bases)} the way decide.series groups them."""
    grouped = {}
    for site, d in scored.items():
        times = np.array([t.timestamp() for t in d["times"]])
        line = calibration["node_thresholds"][site]
        score = (d["error"] - calibration["error_median"][site]) / calibration["error_iqr"][site]
        grouped[(site, "conductivity", "forecast_residual")] = (
            times,
            score,
            line * d["rain"],
            np.full(len(times), line),
        )
        level = np.abs(
            (d["level"] - calibration["cond_median"][site]) / calibration["cond_iqr"][site]
        )
        k = np.full(len(times), d["k"])
        grouped[(site, "conductivity", "level_shift")] = (times, level, k, k)
    return grouped


def _robust(values):
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    # A flat distribution would divide by zero, the same guard training uses
    iqr = float(q75 - q25)
    return median, iqr if iqr > 1e-6 else 1.0


def dusk_refit(scored, keep, q, base):
    """The calibration with each site's median, spread and alert line re-fitted on its kept steps."""
    out = copy.deepcopy(base)
    for site, d in scored.items():
        kept = keep[site] & np.isfinite(d["error"])
        if kept.sum() < MIN_FIT:
            continue
        median, iqr = _robust(d["error"][kept])
        line, _ = _pot_line((d["error"][kept] - median) / iqr, q)
        out["error_median"][site] = median
        out["error_iqr"][site] = iqr
        out["node_thresholds"][site] = line
        levels = d["level"][kept & np.isfinite(d["level"])]
        if levels.size >= MIN_FIT:
            out["cond_median"][site], out["cond_iqr"][site] = _robust(levels)
    return out


# Cobble Shoal


def cobble_pass(tables, weather):
    """Every anchor's channel scores, computed once. Any calibration's nulls then combine them."""
    import torch

    from strawberrywatch.ingest.raw_data_loader import tidy_channels
    from strawberrywatch.preprocessing import node_windows as nw
    from strawberrywatch.serving.cobble import _rain_on
    from strawberrywatch.serving.contract import utc_indexed

    det = detector("cobble_shoal")
    frames = {}
    for name in det.table_names:
        raw = tables.get(name)
        if raw is not None and not raw.empty:
            frames[name] = det.clean(tidy_channels(utc_indexed(name, raw)))
    stamps = [f.index for f in frames.values() if len(f)]
    if not stamps:
        return None
    start = min(s.min() for s in stamps).floor(nw.GRID_FREQ)
    end = max(s.max() for s in stamps).floor(nw.GRID_FREQ)
    with _lock, torch.no_grad():
        win = det.prepare(
            nw.build_window(
                frames, start, end, inventory=det.inventory, scaler=det.scaler, as_of=end
            )
        )
        grid, nodes = win["grid"], win["nodes"]
        covered = win["target_mask"].sum(axis=1) >= det.min_coverage * len(nodes)
        anchors = [a for a in range(det.window, len(grid)) if covered[a]]
        per_anchor = []
        for anchor in anchors:
            chans = det.model.channels(nw.to_batch(win, anchor, det.window))
            per_anchor.append(
                {k: np.asarray(torch.as_tensor(v).cpu()).ravel() for k, v in chans.items()}
            )
    rain, _ = _rain_on(grid, weather)
    bars = det.gate.thresholds(
        rain, grid.tz_localize(None).to_numpy(), nodes=[node.key for node in nodes]
    )
    rain_factor = np.asarray(bars)[anchors] / float(det.gate.base_threshold)
    inventory_name = {canonical: name for name, canonical in nw.VARIABLE_MAP.items()}
    keys = [(nw.SITE_TO_TABLE.get(n.site, n.site), inventory_name[n.var]) for n in nodes]
    return {
        "grid": grid,
        "anchors": anchors,
        "channels": per_anchor,
        "rain": rain_factor,
        "keys": keys,
        "window": det.window,
    }


def _nulls(calibration):
    from strawberrywatch.anomalies import channel_scoring

    return channel_scoring.ChannelNulls.from_dict(calibration["channel_nulls"])


def _combined(chans, nulls):
    from strawberrywatch.anomalies import channel_scoring

    return np.asarray(
        channel_scoring.combine(pvalues=nulls.pvalues(chans), rule="fisher"), dtype=float
    ).ravel()


def cobble_rows(scored, calibration):
    if scored is None or not scored["anchors"]:
        return {}
    nulls = _nulls(calibration)
    line = float(calibration["z_q"])
    scores = np.vstack([_combined(c, nulls) for c in scored["channels"]])
    times = np.array([scored["grid"][a].timestamp() for a in scored["anchors"]])
    grouped = {}
    for j, (site, variable) in enumerate(scored["keys"]):
        finite = np.isfinite(scores[:, j])
        grouped[(site, variable, "combined_fisher")] = (
            times[finite],
            scores[finite, j],
            line * scored["rain"][finite, j],
            np.full(int(finite.sum()), line),
        )
    return grouped


def cobble_refit(scored, keep, q, base):
    """Nulls and the POT line re-fitted on fault-free anchors of the picked data."""
    from strawberrywatch.anomalies import channel_scoring

    out = copy.deepcopy(base)
    clean = [i for i, ok in enumerate(keep) if ok]
    if not clean:
        return out
    picks = sorted(
        {clean[int(i)] for i in np.linspace(0, len(clean) - 1, min(NULL_ANCHORS, len(clean)))}
    )
    names = scored["channels"][0].keys()
    per_channel = {n: np.concatenate([scored["channels"][i][n] for i in picks]) for n in names}
    nulls = channel_scoring.ChannelNulls().fit(per_channel)
    combined = np.concatenate([_combined(scored["channels"][i], nulls) for i in picks])
    line, fit = _pot_line(combined, q)
    grid = scored["grid"]
    out.update(
        {
            "channel_nulls": nulls.to_dict(),
            "pot": {f"combined_fisher@{q:g}": fit},
            "operating_q": q,
            "z_q": line,
            "n_null_windows": len(picks),
            "corpus": {
                "start": str(grid[scored["anchors"][0]]),
                "end": str(grid[scored["anchors"][-1]]),
                "min_coverage": detector("cobble_shoal").min_coverage,
            },
        }
    )
    return out


# Shared by both models


def score(model, tables, weather):
    return dusk_pass(tables, weather) if model == "dusk_crayfish" else cobble_pass(tables, weather)


def rows(model, scored, calibration):
    if model == "dusk_crayfish":
        return dusk_rows(scored, calibration)
    return cobble_rows(scored, calibration)


def alerts(model, grouped, start, end, sites=None):
    """Alerts that start between start and end, counted the way the dashboard's checks count them."""
    from strawberrywatch.watch import decide

    if not grouped:
        return []
    lo, hi = start.timestamp(), end.timestamp()
    checks_run = [
        {"window_start": t - LOOKBACK, "window_end": t, "started_at": t}
        for t in np.arange(lo, hi + 1, CHECK_EVERY)
    ]
    settings = {
        "models": {model: {"on": True, "scales": {}}},
        "repeat_hours": REPEAT_HOURS,
        "dont_flag": [],
    }
    kept = decide.replay({model: grouped}, checks_run, settings, {model: info(model).min_steps})
    return [
        item
        for _, _, item, _ in kept
        if lo <= item["first_over"] <= hi and (sites is None or item["site"] in sites)
    ]


def _keep_steps(model, scored, every, rain, leave_events, leave_rain):
    """What the re-fit may use: per site for Dusk Crayfish, per anchor window for Cobble Shoal."""
    if model == "dusk_crayfish":
        return {
            site: ~data.left_out(d["times"], site, every, rain, leave_events, leave_rain)
            for site, d in scored.items()
        }
    grid = scored["grid"]
    out_any = np.zeros(len(grid), dtype=bool)
    for site in {k[0] for k in scored["keys"]}:
        out_any |= data.left_out(grid, site, every, rain, leave_events, leave_rain)
    # An anchor is clean when nothing in its window was left out
    cumulative = np.concatenate([[0], np.cumsum(out_any)])
    w = scored["window"]
    return [cumulative[a + 1] - cumulative[a - w] == 0 for a in scored["anchors"]]


def has_scores(grouped, start, end, sites):
    lo, hi = start.timestamp(), end.timestamp()
    for (site, _, _), (times, *_rest) in grouped.items():
        if site in sites and ((times >= lo) & (times <= hi)).any():
            return True
    return False


def verdict(found, should_alert):
    if found is None:
        return text.NOT_ENOUGH
    if should_alert:
        return text.CAUGHT if found else text.MISSED
    return text.ALERTED if found else text.STAYED_QUIET


# Scores of past events don't depend on the calibration, only on the readings
_event_scores = {}


def event_scores(folder, model, event, kind):
    key = (model, event["id"], str(event["start"]), str(event["end"]), kind)
    if key in _event_scores:
        return _event_scores[key]
    names = model_sites(model)
    if event["shipped"] and event["shipped_id"]:
        tables, weather = events.shipped_raw(event, names)
    else:
        start, end = event["start"] - CONTEXT, event["end"]
        try:
            data.fetch(folder, kind, names, start, end)
        except Exception:
            return None
        tables = data.raw_tables(data.readings(folder, kind, names, start, end))
        weather = data.weather(folder, kind, start, end)
    scored = score(model, tables, weather) if tables else None
    _event_scores[key] = scored
    return scored


def past_events(folder, model, calibrations, kind):
    """One row per event at the model's sites, judged under each calibration given."""
    sites = set(model_sites(model))
    out = []
    for event in events.every(folder):
        if not sites & set(event["sites"]):
            continue
        scored = event_scores(folder, model, event, kind)
        mine = sorted(sites & set(event["sites"]))
        row = {"id": event["id"], "name": event["name"], "should_alert": event["should_alert"]}
        row["dates"] = text.short_dates(
            text.local(event["start"]).date(), text.local(event["end"]).date()
        )
        results = []
        for calibration in calibrations:
            grouped = rows(model, scored, calibration) if scored else {}
            if not has_scores(grouped, event["start"], event["end"], mine):
                results.append(text.NOT_ENOUGH)
                continue
            found = alerts(model, grouped, event["start"], event["end"], mine)
            results.append(verdict(bool(found), event["should_alert"]))
        row["results"] = results
        out.append(row)
    return out


def run_check(folder, session, model, choice):
    """Re-fits on the picked data, then judges past events and the picked data under both lines."""
    kind = session["source"]
    start, end = data.utc(session["start_date"]), data.utc(session["end_date"], end=True)
    names = model_sites(model)
    tables = data.raw_tables(tables_tidy(folder, kind, names, start, end))
    weather = data.weather(folder, kind, start, end)
    rain = weather["rain_mm"] if weather is not None and "rain_mm" in weather else None
    every = events.every(folder)
    current_id = in_use(folder, model)
    current = blob(folder, model, current_id)
    scored = score(model, tables, weather)
    if not scored:
        return None
    keep = _keep_steps(
        model,
        scored,
        every,
        rain,
        bool(session["leave_out_events"]),
        bool(session["leave_out_rain"]),
    )
    q = per_step_rate(model, choice)
    base = blob(folder, model, store.SHIPPED)
    refit = dusk_refit if model == "dusk_crayfish" else cobble_refit
    new = refit(scored, keep, q, base)
    before = len(alerts(model, rows(model, scored, current), start, end))
    after = len(alerts(model, rows(model, scored, new), start, end))
    return {
        "model": model,
        "false_alarms": choice,
        "calibration": current_id,
        "events": past_events(folder, model, [current, new], kind),
        "before": before,
        "after": after,
        "blob": new,
        "reference": reference_from(
            tables_tidy(folder, kind, names, start, end), every, rain, session
        ),
    }


def tables_tidy(folder, kind, names, start, end):
    """The picked readings at the reviewed level. The re-fit and its drift reference read these."""
    return data.reviewed(data.readings(folder, kind, names, start, end))


def reference_from(tables, every, rain, session):
    """The drift reference for a calibration made from these readings: site, reading, value."""
    rows_out = []
    for site, frame in tables.items():
        mask = data.left_out(
            frame.index,
            site,
            every,
            rain,
            bool(session["leave_out_events"]),
            bool(session["leave_out_rain"]),
        )
        for reading in text.READINGS:
            if reading in frame.columns:
                values = checks.reference_sample(frame[reading][~mask].to_numpy(dtype=float))
                rows_out += [(site, reading, float(v)) for v in values]
    return rows_out


# Drift references for the shipped calibrations, rebuilt from the repository's data


def _sample_frame(rows_in):
    return pd.DataFrame(rows_in, columns=["site", "reading", "value"])


def shipped_reference(folder, model):
    """The shipped calibration's reference sample, built once from the repository's data when it's here."""
    path = store.calibration_dir(folder, model, store.SHIPPED) / "reference.csv.gz"
    if path.is_file():
        return pd.read_csv(path)
    try:
        built = _dusk_reference() if model == "dusk_crayfish" else _cobble_reference()
    except Exception:
        return None
    if built is None or built.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    built.to_csv(path, index=False)
    return built


def _shipped_left_out(index, site, rain):
    return data.left_out(index, site, events.shipped(), rain, True, rain is not None)


def _archive_rain(start, end):
    from strawberrywatch.ingest import weather_archive

    try:
        weather = weather_archive.load_weather(start, end, allow_fetch=False)
    except Exception:
        return None
    return pd.to_numeric(weather["rain_mm"], errors="coerce").resample("15min").sum()


def _dusk_reference():
    from strawberrywatch import paths
    from strawberrywatch.config import Config

    corpus = paths.training_corpus()
    if not corpus.is_file():
        return None
    df = pd.read_csv(corpus)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")
    # The last fifth of its sequences was the validation data the lines came from
    n_sequences = len(df) - Config.SEQUENCE_LENGTH
    start = df["datetime"].iloc[int(n_sequences * Config.TRAIN_SPLIT) + Config.SEQUENCE_LENGTH]
    df = df[df["datetime"] >= start].set_index("datetime")
    rain = df["rain_mm"] if "rain_mm" in df else None
    rows_out = []
    for site in model_sites("dusk_crayfish"):
        mask = _shipped_left_out(df.index, site, rain)
        for reading in text.READINGS:
            column = f"{site}_{reading}"
            if column in df.columns:
                values = checks.reference_sample(df[column][~mask].to_numpy(dtype=float))
                rows_out += [(site, reading, float(v)) for v in values]
    return _sample_frame(rows_out)


def _cobble_reference():
    from strawberrywatch import inventory as inv
    from strawberrywatch.ingest.raw_data_loader import load_archive_by_table

    cal = json.loads(shipped_path("cobble_shoal").read_text())
    span = cal.get("corpus") or {}
    if not span:
        return None
    start, end = pd.Timestamp(span["start"]), pd.Timestamp(span["end"])
    tables = load_archive_by_table(earliest=inv.load().earliest_plausible_reading)
    rain = _archive_rain(start, end)
    excluded = span.get("excluded_spans") or []
    rows_out = []
    for site in model_sites("cobble_shoal"):
        frame = tables.get(site)
        if frame is None or frame.empty:
            continue
        frame = frame[(frame.index >= start) & (frame.index <= end)]
        frame = frame.groupby(frame.index.floor("15min")).median()
        mask = _shipped_left_out(frame.index, site, rain)
        for lo, hi in excluded:
            mask |= (frame.index >= pd.Timestamp(lo, tz="UTC")) & (
                frame.index < pd.Timestamp(hi, tz="UTC")
            )
        for reading in text.READINGS:
            if reading in frame.columns:
                values = checks.reference_sample(frame[reading][~mask].to_numpy(dtype=float))
                rows_out += [(site, reading, float(v)) for v in values]
    return _sample_frame(rows_out)


def reference(folder, model, calibration_id):
    if calibration_id == store.SHIPPED:
        return shipped_reference(folder, model)
    path = store.calibration_dir(folder, model, calibration_id) / "reference.csv.gz"
    return pd.read_csv(path) if path.is_file() else None


def drift(folder, model, kind, now=None):
    """The last 30 days at each of the model's sites against the in-use calibration's reference."""
    calibration_id = in_use(folder, model)
    ref = reference(folder, model, calibration_id)
    if ref is None:
        return calibration_id, {"no_reference": True}
    now = now or pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=checks.DRIFT_DAYS)
    names = model_sites(model)
    data.fetch(folder, kind, names, start, now)
    tables = data.readings(folder, kind, names, start, now)
    weather = data.weather(folder, kind, start, now)
    rain = weather["rain_mm"] if weather is not None and "rain_mm" in weather else None
    every = events.every(folder)
    pairs, found = [], []
    for site in names:
        frame = tables.get(site)
        if frame is None:
            continue
        mask = data.left_out(frame.index, site, every, rain, True, True)
        for reading in text.READINGS:
            if reading not in frame.columns:
                continue
            reference_values = ref[(ref["site"] == site) & (ref["reading"] == reading)]["value"]
            result = checks.drifted(frame[reading][~mask].to_numpy(dtype=float), reference_values)
            if result is None:
                continue
            moved, ratio = result
            pairs.append([site, reading, ratio])
            if moved:
                found.append([site, reading])
    return calibration_id, {"checked": len(pairs), "pairs": pairs, "drifted": found}


def save(folder, session, check):
    """Saves the checked calibration, puts it in use, and returns its id."""
    model = check["model"]
    record = {
        "model": model,
        "source": session["source"],
        "sites": session["sites"],
        "readings": session["readings"],
        "start_date": session["start_date"],
        "end_date": session["end_date"],
        "leave_out_events": bool(session["leave_out_events"]),
        "leave_out_rain": bool(session["leave_out_rain"]),
        "false_alarms": check["false_alarms"],
        "notes": session.get("notes") or "",
        "events": check["events"],
        "before": check["before"],
        "after": check["after"],
        "replaced": check["calibration"],
    }
    reference_rows = _sample_frame(check.get("reference") or [])
    calibration_id = store.save_calibration(
        folder, model, FILENAMES[model], check["blob"], record, reference_rows
    )
    store.use(folder, model, calibration_id)
    return calibration_id


def compare(folder, model, calibration_id, kind):
    """The past-events check rerun for the one in use and this one."""
    return past_events(
        folder,
        model,
        [blob(folder, model, in_use(folder, model)), blob(folder, model, calibration_id)],
        kind,
    )
