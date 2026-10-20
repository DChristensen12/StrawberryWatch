"""
Turns what's in the store into what the pages show. Templates only lay it out.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from strawberrywatch.watch import decide, sources
from strawberrywatch.watch import settings as ws

H = 3600
STEP = 900
# The server runs in UTC, and "2:15 pm" has to mean 2:15 pm at the creek
PACIFIC = ZoneInfo("America/Los_Angeles")
# A Mayfly uploads a few minutes after it samples. A reading isn't late until
# this long past when it was due
LATE_AFTER = STEP + 5 * 60
RECENT_HOURS = 24
# Overview warns after this long without a finished check. A stopped watcher
# shouldn't look like a calm creek
STALE_MINUTES = 45
STRIP_HOURS = 24
SPARK_HOURS = 72
SPARK_GAP = 6 * H  # Monitor My Watershed breaks a line across a gap this long

LABELS = {
    "conductivity": "conductivity",
    "temperature": "temperature",
    "depth": "depth",
    "dissolved_oxygen": "dissolved oxygen",
    "floating_conductivity": "floating conductivity",
    "battery": "battery",
}
UNITS = {
    "conductivity": "µS/cm",
    "temperature": "°C",
    "depth": "mm",
    "dissolved_oxygen": "mg/L",
    "floating_conductivity": "µS/cm",
    "battery": "V",
}
# The site's own map labels, from strawberry_creek_map.html
SHORT = {
    "botanical_garden": "BG",
    "south_fork_0": "SF0",
    "south_fork_1": "SF1",
    "south_fork_2": "SF2",
    "south_fork_3": "SF3",
    "kingman_hall": "KM",
    "university_house": "UH",
    "north_fork_0": "NF0",
    "scnf010": "NF1",
    "oxford": "OX",
    "codornices": "CC",
}


# What the website calls a site, and the landmark after the comma
site_name = ws.site_name
site_place = ws.site_place


# Times


def _local(ts):
    return datetime.fromtimestamp(ts, PACIFIC)


def midnight(d):
    """The moment a Pacific day starts."""
    return datetime(d.year, d.month, d.day, tzinfo=PACIFIC).timestamp()


def hour_start(d, hour_of_day):
    return datetime(d.year, d.month, d.day, hour_of_day, tzinfo=PACIFIC).timestamp()


def clock(ts):
    """2:15 pm"""
    return _local(ts).strftime("%-I:%M %p").replace("AM", "am").replace("PM", "pm")


def day(ts):
    """12 Oct"""
    return _local(ts).strftime("%-d %b")


def hour(ts):
    """2 pm"""
    return _local(ts).strftime("%-I %p").replace("AM", "am").replace("PM", "pm")


def stamp(ts):
    """2026-10-12 09:40, which a spreadsheet sorts and reads as a date."""
    return _local(ts).strftime("%Y-%m-%d %H:%M")


def long_day(d):
    """Monday 12 October 2026"""
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


def day_title(d, today):
    """Today, Yesterday, or Saturday 10 Oct, with the year when it isn't this one."""
    if d == today:
        return "Today"
    if d.toordinal() == today.toordinal() - 1:
        return "Yesterday"
    label = f"{d.strftime('%A')} {d.day} {d.strftime('%b')}"
    return label if d.year == today.year else f"{label} {d.year}"


def moment(ts, ref):
    """
    2:15 pm today, 12 Oct, 2:15 pm otherwise, with the year when it isn't this
    one. With no ref, for a file that's read later, always the full date.
    """
    if ref is None:
        then = _local(ts)
        return f"{day(ts)} {then.year}, {clock(ts)}"
    then, now = _local(ts), _local(ref)
    if then.date() == now.date():
        return clock(ts)
    stamp = f"{day(ts)}, {clock(ts)}"
    return stamp if then.year == now.year else f"{day(ts)} {then.year}, {clock(ts)}"


def when_words(ts, ref):
    """
    Today at 2:15 pm, Yesterday at 11:50 pm, Saturday 10 Oct at 2:15 pm, or
    12 Mar 2025 at 2:15 pm when it isn't this year.
    """
    then, now = _local(ts), _local(ref)
    days = now.date().toordinal() - then.date().toordinal()
    if days == 0:
        return f"Today at {clock(ts)}"
    if days == 1:
        return f"Yesterday at {clock(ts)}"
    if then.year != now.year:
        return f"{day(ts)} {then.year} at {clock(ts)}"
    return f"{then.strftime('%A')} {day(ts)} at {clock(ts)}"


def ago(ts, ref):
    """12 minutes ago"""
    seconds = max(0, ref - ts)
    minutes = int(seconds // 60)
    if minutes < 1:
        return "just now"
    if minutes < 90:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = round(seconds / H)
    if hours < 36:
        return f"{hours} hours ago"
    days = round(seconds / 86400)
    return f"{days} days ago"


def reference_now(clock=time.time):
    """ "Now" for the pages: the real clock, or the one a test fixed."""
    return clock()


def first_check_pending(store):
    """True until a live check has finished. Until then Overview has no news."""
    return store.last_finished_check(kinds=sources.LIVE) is None


def stale_since(store, ref):
    """When the last check finished, if that was over 45 minutes ago. None when checks are keeping up."""
    last = store.last_finished_check(kinds=sources.LIVE)
    if last is None:
        first = store.last_check(kinds=sources.LIVE)
        if first and ref - first["started_at"] > STALE_MINUTES * 60:
            return {"since": None}
        return None
    if ref - last["finished_at"] > STALE_MINUTES * 60:
        return {"since": last["finished_at"]}
    return None


# The creek map


def creek_layout(inventory):
    """
    Columns and rows, flow left to right. Each branch into an outlet gets its
    own row, the longest on top. Sites with no edges sit on a row below.
    """
    isolated = set(inventory.isolated())
    outlets = sorted(
        t for t, s in inventory.sites.items() if s.downstream is None and t not in isolated
    )

    def chain_up(site):
        ups = inventory.upstream_of(site)
        if not ups:
            return [site]
        return max((chain_up(u) for u in ups), key=len) + [site]

    positions, rows = {}, 0
    for outlet in outlets:
        branches = sorted(
            (chain_up(u) for u in inventory.upstream_of(outlet)), key=lambda c: (-len(c), c[0])
        )
        width = max((len(b) for b in branches), default=0)
        first = rows
        for branch in branches:
            for i, site in enumerate(branch):
                positions[site] = (width - len(branch) + i, rows)
            rows += 1
        positions[outlet] = (width, (first + rows - 1) / 2)
    for site in sorted(isolated):
        positions[site] = (0, rows)
    for site in sorted(set(inventory.sites) - set(positions)):
        rows += 1
        positions[site] = (0, rows)
    return positions, [list(e) for e in inventory.edges()], sorted(isolated)


def creek_map(inventory, statuses, arrows=(), ticks=None):
    """
    Everything the map template draws, in pixels. Edges stop short of each
    square so the arrowhead shows. `arrows` are edges a problem travelled
    along, and with `ticks` each square shows whether alerts are on instead.
    """
    positions, edges, isolated = creek_layout(inventory)
    col, row, left, top = 118, 64, 96, 34
    px = {s: (left + x * col, top + y * row) for s, (x, y) in positions.items()}
    width = max(p[0] for p in px.values()) + 60
    height = max(p[1] for p in px.values()) + 32
    travelled = {tuple(a) for a in arrows}
    lines = []
    for a, b in edges:
        (x1, y1), (x2, y2) = px[a], px[b]
        length = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        lines.append(
            {
                "x1": round(x1 + ux * 12, 1),
                "y1": round(y1 + uy * 12, 1),
                "x2": round(x2 - ux * 13, 1),
                "y2": round(y2 - uy * 13, 1),
                "travel": (a, b) in travelled,
            }
        )
    nodes = []
    for site, (x, y) in px.items():
        status = statuses.get(site, {"key": "reporting", "text": "Normal"})
        watched = None if ticks is None else site in ticks
        text = status["text"]
        if watched is not None:
            text = "Alerts on" if watched else "Alerts off"
        nodes.append(
            {
                "site": site,
                "short": SHORT.get(site, site),
                "x": x,
                "y": y,
                "key": status["key"],
                "text": text,
                "watched": watched,
            }
        )
    labels = [{"x": px[s][0] - 22, "y": px[s][1] + 4, "text": "Control"} for s in isolated]
    return {"width": width, "height": height, "edges": lines, "nodes": nodes, "labels": labels}


def flow_path(inventory, order):
    """The edges along the creek between sites in the order they fired, for arrows."""
    path = []
    for a, b in zip(order, order[1:], strict=False):
        chain = [a, *inventory.downstream_chain(a)]
        if b in chain:
            path += [[chain[i], chain[i + 1]] for i in range(chain.index(b))]
    return path


# Sites


def recent_noticed(store, ref):
    """
    Live alerts with an unusual reading in the last 24 hours. Past events from
    load_test.py or the test dashboard don't count.
    """
    since = ref - RECENT_HOURS * H
    return [n for n in store.noticed(flagged=True, kinds=sources.LIVE) if n["last_over"] >= since]


def site_statuses(store, inventory, applied, ref):
    """Each site's status: flagged, late, muted (alerts off) or reporting."""
    unusual = {n["site"] for n in recent_noticed(store, ref)}
    last = store.last_readings(live=True)
    dont_flag = set(applied["dont_flag"]) if applied else set()
    out = {}
    for site in inventory.sites:
        newest = last.get(site, {}).get("time")
        if site in unusual:
            out[site] = {"key": "flagged", "text": "Alert"}
        elif newest is None:
            out[site] = {"key": "late", "text": "No data yet"}
        elif ref - newest > LATE_AFTER:
            out[site] = {"key": "late", "text": f"No data for {_span(ref - newest)}"}
        elif site in dont_flag:
            out[site] = {"key": "muted", "text": "Alerts off"}
        else:
            out[site] = {"key": "reporting", "text": "Normal"}
    return out


def _span(seconds):
    minutes = int(seconds // 60)
    if minutes < 90:
        return f"{minutes} minutes"
    hours = round(seconds / H)
    return f"{hours} hours" if hours < 48 else f"{round(seconds / 86400)} days"


def site_order(inventory):
    positions, _, _ = creek_layout(inventory)
    return sorted(inventory.sites, key=lambda s: (positions[s][1], positions[s][0]))


def hour_span(ts):
    """2 to 3 pm, or 11 am to 12 pm when it crosses noon or midnight."""
    start, end = hour(ts), hour(ts + H)
    if start[-2:] == end[-2:]:
        return f"{start[:-3]} to {end}"
    return f"{start} to {end}"


def reporting_marks(store, site, ref):
    """
    One mark per hour for the last 24, full, partial or empty by how many of its
    four readings came. The current hour only judges readings already due.
    Pacific hours start on the UTC hour, and the arithmetic stays in seconds.
    """
    this_hour = ref - ref % H
    start = this_hour - (STRIP_HOURS - 1) * H
    rows = store.readings(site, since=start, until=ref, live=True)
    came = {r["time"] - r["time"] % STEP for r in rows}
    marks = []
    for k in range(STRIP_HOURS):
        begins = start + k * H
        quarters = [begins + q * STEP for q in range(4)]
        count = sum(t in came for t in quarters)
        span = hour_span(begins)
        if begins < this_hour:
            expected = 4
            label = f"{span}: {count} of 4 readings"
        else:
            expected = sum(1 for t in quarters if t + LATE_AFTER <= ref)
            label = f"{span}: {count} of 4 readings so far"
        if count >= max(expected, 1):
            state = "full"
        elif expected == 0:
            state = "pending"
        else:
            state = "partial" if count else "empty"
        marks.append({"state": state, "label": label, "count": count})
    return {"marks": marks, "count": sum(m["count"] for m in marks)}


def sparklines(store, inventory, site, ref):
    """One small line per sensor over 72 hours, broken across gaps over 6 hours."""
    since = ref - SPARK_HOURS * H
    rows = store.readings(site, since=since, until=ref, live=True)
    entry = inventory.site(site)
    columns = [
        v
        for v in (
            "conductivity",
            "temperature",
            "depth",
            "dissolved_oxygen",
            "floating_conductivity",
        )
        if v in entry.scored_variables
    ]
    if any(r.get("battery") is not None for r in rows):
        columns.append("battery")
    out = []
    for column in columns:
        xs, ys, prev = [], [], None
        for r in rows:
            value = r.get(column)
            if value is None:
                continue
            if prev is not None and r["time"] - prev > SPARK_GAP:
                xs.append(prev + 1)
                ys.append(None)
            xs.append(r["time"])
            ys.append(value)
            prev = r["time"]
        newest = next((r for r in reversed(rows) if r.get(column) is not None), None)
        out.append(
            {
                "column": column,
                "label": LABELS[column],
                "unit": UNITS[column],
                "x": xs,
                "y": ys,
                "last": None if newest is None else newest[column],
                "last_time": None if newest is None else newest["time"],
                "start": since,
                "end": ref,
            }
        )
    return out


# Alerts, in plain words


def _direction(store, item):
    """up, down or None, from the readings before and during the item."""
    live = sources.is_live(item["check_source"]) if item.get("check_source") else None
    rows = store.readings(
        item["site"], since=item["first_over"] - 24 * H, until=item["last_over"], live=live
    )
    variable = item["variable"]
    before = [
        r[variable] for r in rows if r["time"] < item["first_over"] and r.get(variable) is not None
    ]
    during = [
        r[variable] for r in rows if r["time"] >= item["first_over"] and r.get(variable) is not None
    ]
    if len(before) < 2 or not during:
        return None
    return "up" if np.mean(during) > np.mean(before) else "down"


# Rules whose sentence says which way the reading went
SHIFT_RULES = ("level_shift", "cusum_shift")
# Sentences written but not yet approved, marked on Alert examples
PROPOSED_RULES = ("combined_fisher",)


def what_happened(store, item, ref):
    """One sentence on what the reading did and when."""
    direction = _direction(store, item) if item["rule"] in SHIFT_RULES else None
    return happened(item["variable"], item["rule"], moment(item["first_over"], ref), direction)


def happened(variable, rule, when, direction=None):
    """The sentence for one rule. direction is "up", "down" or None when it's unclear."""
    variable = LABELS.get(variable, variable).capitalize()
    if rule == "level_shift":
        if direction == "up":
            return f"{variable} jumped at {when} and stayed high."
        if direction == "down":
            return f"{variable} dropped at {when} and stayed low."
        return f"{variable} changed suddenly at {when} and stayed at the new level."
    if rule == "forecast_residual":
        return f"{variable} didn't match the model's forecast from {when}."
    if rule == "combined_fisher":
        return f"{variable} stopped matching the model's forecast and the creek's other sensors from {when}."
    if rule == "hampel_spike":
        return f"{variable} spiked at {when}."
    if rule == "cusum_shift":
        if direction == "up":
            return f"{variable} rose at {when} and stayed high."
        if direction == "down":
            return f"{variable} fell at {when} and stayed low."
        return f"{variable} shifted at {when} and stayed at the new level."
    if rule == "ewma_drift":
        return f"{variable} started drifting at {when}."
    return f"{variable} didn't match the model's expected pattern from {when}."


def extras_said(item):
    """
    A sentence per support module that had something to say. Their raw output
    is on the alert's page.
    """
    out = []
    accounts = item.get("explanations") or {}
    run = accounts.get("newtwork_run")
    if run:
        order = run.get("order") or []
        verdict = run.get("verdict")
        names = [site_name(s) for s in order]
        here = site_name(item["site"])
        if verdict == "consistent_with_transport" and len(order) > 1:
            if order[0] == item["site"]:
                out.append(f"Started here, then reached {', then '.join(names[1:])} downstream.")
            else:
                before = names[: order.index(item["site"])] if item["site"] in order else names[:-1]
                out.append(f"Reached {here} after {', then '.join(before)}, moving downstream.")
        elif verdict == "catchment_wide":
            out.append(
                "Other sites changed at the same time, which points to rain or a creek-wide cause."
            )
        elif verdict == "not_transport":
            out.append(
                "A site on another creek changed at the same time, so this didn't come down Strawberry Creek."
            )
        elif verdict == "local_to_this_site":
            out.append("Only this site changed.")
    bed = accounts.get("trial_bed")
    if bed:
        if bed.get("verdict") == "diagnosed" and bed.get("cause"):
            out.append(f"Closest match: {bed['cause']}.")
        elif bed.get("verdict") == "possible_new_type":
            out.append("Doesn't match any known type of spill.")
        else:
            out.append("Not enough sensors at this site to suggest a cause.")
    pool = accounts.get("settling_pool")
    if pool and pool.get("verdict") == "held_back":
        out.append(f"Settling Pool {pool['explanation']}.")
    if item.get("check_weather_ok") == 0:
        out.append("No weather data for this check, so rain wasn't allowed for.")
    return out


def item_view(store, item, ref, direction=None):
    """Everything the pages say about one alert. Without a store, direction is taken as given."""
    if store is None:
        sentence = happened(
            item["variable"], item["rule"], moment(item["first_over"], ref), direction
        )
    else:
        sentence = what_happened(store, item, ref)
    return {
        **item,
        "title": item["model"].replace("_", " ").title(),
        "label": LABELS.get(item["variable"], item["variable"]),
        "site_name": site_name(item["site"]),
        "happened": sentence,
        "extras": extras_said(item),
        "when": moment(item["first_over"], ref),
        "when_words": when_words(item["first_over"], ref if ref is not None else time.time()),
        "day": day(item["first_over"]),
    }


def score_chart(store, item, before=36 * H, after=12 * H):
    """Score, alert line (with and without rain) and rain around one alert."""
    start, end = item["first_over"] - before, item["last_over"] + after
    rows = store.scores(
        model=item["model"],
        site=item["site"],
        variable=item["variable"],
        rule=item["rule"],
        since=start,
        until=end,
    )
    if not rows:
        return None
    scale = item.get("scale") or 1.0
    rain = dict(store.weather(start, end))
    return {
        "x": [r["time"] for r in rows],
        "score": [r["score"] for r in rows],
        "line": [r["threshold"] * scale for r in rows],
        "dry": [r["base"] * scale for r in rows],
        "rain": [rain.get(r["time"] - r["time"] % STEP) for r in rows],
        "flag": item["first_over"],
        "label": LABELS.get(item["variable"], item["variable"]),
        # What this rule's score measures, for the chart's key
        "about": ws.RULE_WORDS.get(item["rule"], ("", ""))[1],
    }


def reading_chart(store, item, ref, before=24 * H, after=6 * H):
    """
    The reading around one alert, with the steps over the alert line picked
    out the way decide counts them. Broken across gaps like the sparklines.
    """
    column = item["variable"]
    if column not in UNITS:
        return None
    start = item["first_over"] - before
    end = min(item["last_over"] + after, ref)
    rows = store.readings(item["site"], since=start, until=end, live=True)
    rows = [r for r in rows if r.get(column) is not None]
    if not rows:
        return None
    scale = item.get("scale") or 1.0
    over = {item["first_over"]} | {
        s["time"]
        for s in store.scores(
            model=item["model"],
            site=item["site"],
            variable=item["variable"],
            rule=item["rule"],
            since=item["first_over"],
            until=item["last_over"],
        )
        if s["score"] is not None
        and s["threshold"] is not None
        and s["score"] > s["threshold"] * scale
    }
    xs, ys, unusual, prev = [], [], [], None
    for r in rows:
        if prev is not None and r["time"] - prev > SPARK_GAP:
            xs.append(prev + 1)
            ys.append(None)
            unusual.append(None)
        xs.append(r["time"])
        ys.append(r[column])
        unusual.append(r[column] if r["time"] - r["time"] % STEP in over else None)
        prev = r["time"]
    return {
        "x": xs,
        "y": ys,
        "unusual": unusual,
        "label": f"{LABELS[column].capitalize()}, {UNITS[column]}",
    }


# The 7 day preview on Advanced settings


def preview_counts(store, models, draft, days=7):
    """
    Per model, how many alerts the last few days of checks would have raised
    with the draft's values, next to how many they did. Same decide() rules.
    """
    since = time.time() - days * 86400
    checks = [
        c for c in store.checks() if c["started_at"] >= since and c["window_start"] is not None
    ]
    out = {}
    if not checks:
        return {m.name: None for m in models}
    first = min(c["window_start"] for c in checks)
    grouped = {}
    for m in models:
        rows = store.scores(model=m.name, since=first)
        if rows and m.sensitivity == "rate":
            rows = at_rates(rows, m, draft["models"].get(m.name, {}))
        if rows:
            grouped[m.name] = decide.series(rows)
    seed = {}
    for item in store.noticed():
        if item["checked_at"] < checks[0]["started_at"]:
            key = (item["model"], item["site"], item["variable"], item["rule"])
            if key not in seed or item["checked_at"] > seed[key]["checked_at"]:
                seed[key] = {"first_over": item["first_over"], "checked_at": item["checked_at"]}
    kept = decide.replay(grouped, checks, draft, {m.name: m.min_steps for m in models}, seed)
    recorded = [n for n in store.noticed(flagged=True) if n["checked_at"] >= since]
    for m in models:
        if m.name not in grouped:
            out[m.name] = None
            continue
        out[m.name] = {
            "would": sum(1 for _, model, _, flagged in kept if model == m.name and flagged),
            "did": sum(1 for n in recorded if n["model"] == m.name),
        }
    return out


def threshold_at(model, spec, key, rule):
    """A calibrated model's threshold for one sensor and test, at a spec's stop or typed value."""
    from strawberrywatch.models import Water_Strider as wsm

    typed = (spec.get("thresholds") or {}).get(f"{key}.{rule}")
    if typed is not None:
        return typed
    tail = model.tails[key][rule]
    rate = (spec.get("rates") or {}).get(rule, wsm.Q_TOTAL)
    return wsm.threshold_for(tail, wsm.per_test(rate))


def at_rates(rows, model, spec):
    """
    Stored scores with each alert line moved to where the spec puts it. Rain's
    share of the line is kept. It's the detector's own arithmetic without
    re-running it.
    """
    out, lines = [], {}
    for row in rows:
        key = f"{row['site']}.{row['variable']}"
        if key not in model.tails or not row.get("base"):
            continue
        cache = (key, row["rule"])
        if cache not in lines:
            lines[cache] = threshold_at(model, spec, key, row["rule"])
        factor = lines[cache] / row["base"]
        out.append({**row, "threshold": row["threshold"] * factor, "base": lines[cache]})
    return out


def rate_previews(store, model, draft, days=7):
    """
    Per test, the alerts the last few days would have raised at the draft's
    setting and at the calibrated one, as {rule: (draft, calibrated)}. None
    when there are no scores to count yet.
    """
    since = time.time() - days * 86400
    checks = [
        c for c in store.checks() if c["started_at"] >= since and c["window_start"] is not None
    ]
    rows = store.scores(
        model=model.name, since=min((c["window_start"] for c in checks), default=since)
    )
    if not checks or not rows:
        return None
    spec = draft["models"].get(model.name, {})
    calibrated = {**spec, "rates": {}, "thresholds": {}}
    out = {}
    for rule in model.rules:
        mine = [r for r in rows if r["rule"] == rule]
        counts = []
        for setting in (spec, calibrated):
            settings = {
                **draft,
                "models": {model.name: {**spec, "on": True, "scales": {rule: 1.0}}},
            }
            grouped = {model.name: decide.series(at_rates(mine, model, setting))} if mine else {}
            kept = decide.replay(grouped, checks, settings, {model.name: model.min_steps})
            counts.append(sum(1 for *_, flagged in kept if flagged))
        out[rule] = tuple(counts)
    return out


def rain_note(draft, applied, model):
    """Shown under the preview when the draft changes something it can't replay."""
    a = applied["models"].get(model, {})
    d = draft["models"].get(model, {})
    if (
        d.get("leave_out") != a.get("leave_out")
        or d.get("rain_multiplier") != a.get("rain_multiplier")
        or draft["rain"] != applied["rain"]
    ):
        return "Rain and left-out site changes aren't in this count. They take effect from the next check."
    return ""


def model_titles(models, applied):
    """Models on: Dusk Crayfish. Only the running ones, since the full list will get long."""
    on, custom = [], []
    for m in models:
        spec = applied["models"].get(m.name, {})
        if not spec.get("on"):
            continue
        on.append(m.title)
        if m.sensitivity == "rate" and ws.custom_sensitivity(spec):
            custom.append(f"{m.title} uses custom sensitivity.")
    if not on:
        return "No models are on."
    return " ".join([f"Models on: {', '.join(on)}.", *custom])


def check_took(check):
    if not check or not check.get("finished_at"):
        return None
    return check["finished_at"] - check["started_at"]


def alerts_label(settings):
    choice = ws.alerts_choice(settings)
    return {key: label for key, label, _, _ in ws.PRESETS}.get(
        choice, "Custom (set in Advanced settings)"
    )
