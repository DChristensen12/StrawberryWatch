"""
What the pages show, worked out from the store: times in plain words, the creek
map, each site's status, the sentences for something noticed, and chart data.
Templates only lay these out.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from strawberrywatch.watch import decide, sources
from strawberrywatch.watch import settings as ws

H = 3600
STEP = 900
# A Mayfly uploads a few minutes after it samples, so a reading isn't late
# until this long past when it was due
LATE_AFTER = STEP + 5 * 60
RECENT_HOURS = 24
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


# Times


def _local(ts):
    return datetime.fromtimestamp(ts).astimezone()


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


def reference_now(store):
    """
    "Now" for the pages. For the API that's the clock. A CSV folder or a replay
    runs on its own readings' time, so it's the end of the newest check, and
    a replay of 2025 doesn't read "a year ago" everywhere.
    """
    config = store.source()
    last = store.last_check()
    if config and config["kind"] != sources.API and last and last.get("window_end"):
        return last["window_end"]
    return time.time()


# The creek map, from the inventory's own downstream links


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
    Everything the map template draws, in pixels. Each edge stops short of the
    squares at both ends so its arrowhead shows; an edge in `arrows` is one a
    problem travelled along. With `ticks`, a square is watched or not instead
    of showing a status.
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
        status = statuses.get(site, {"key": "reporting", "text": "Reporting"})
        watched = None if ticks is None else site in ticks
        text = status["text"]
        if watched is not None:
            text = "Watched" if watched else "Not watched"
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
    What Overview and Unusual readings show: everything noticed in the last
    day, or for a replay, everything the replay noticed, since the moment it's
    known for is rarely in its last day.
    """
    config = store.source() or {}
    last = store.last_check()
    since = ref - RECENT_HOURS * H
    if config.get("kind") == sources.REPLAY and last and last.get("window_start"):
        since = last["window_start"]
    return [n for n in store.noticed(flagged=True) if n["last_over"] >= since]


STATUS_WORDS = {
    "reporting": "Reporting",
    "late": "Late",
    "flagged": "Flagged",
    "muted": "Muted",
}


def site_statuses(store, inventory, applied, ref):
    """
    One of four states per site, the words the legend uses: flagged when
    something there looks unusual, late when a reading hasn't come, muted when
    it's checked but never flagged, and otherwise reporting.
    """
    unusual = {n["site"] for n in recent_noticed(store, ref)}
    last = store.last_readings()
    dont_flag = set(applied["dont_flag"]) if applied else set()
    out = {}
    for site in inventory.sites:
        newest = last.get(site, {}).get("time")
        if site in unusual:
            out[site] = {"key": "flagged", "text": "Flagged: something looks unusual"}
        elif newest is None:
            out[site] = {"key": "late", "text": "Late: no readings yet"}
        elif ref - newest > LATE_AFTER:
            out[site] = {"key": "late", "text": f"Late: no reading for {_span(ref - newest)}"}
        elif site in dont_flag:
            out[site] = {"key": "muted", "text": "Muted: checked, never flagged"}
        else:
            out[site] = {"key": "reporting", "text": "Reporting"}
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


def reporting_strip(store, site, ref):
    """A mark per expected reading over the last 24 hours, None where it never came."""
    end = ref - ref % STEP
    start = end - STRIP_HOURS * H + STEP
    times = {r["time"] - r["time"] % STEP for r in store.readings(site, since=start, until=end)}
    slots = [start + i * STEP for i in range(STRIP_HOURS * 4)]
    return {
        "x": slots,
        "y": [1 if t in times else None for t in slots],
        "count": sum(t in times for t in slots),
    }


def sparklines(store, inventory, site, ref):
    """One small line per sensor over 72 hours, broken across gaps over 6 hours."""
    since = ref - SPARK_HOURS * H
    rows = store.readings(site, since=since, until=ref)
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


# Something noticed, in plain words


def _direction(store, item):
    """up, down or None, from the readings before and during the item."""
    rows = store.readings(item["site"], since=item["first_over"] - 24 * H, until=item["last_over"])
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


def what_happened(store, item, ref):
    """The plain sentence for one noticed item: what the reading did and when."""
    variable = LABELS.get(item["variable"], item["variable"]).capitalize()
    when = moment(item["first_over"], ref)
    if item["rule"] == "level_shift":
        direction = _direction(store, item)
        if direction == "up":
            return f"{variable} jumped at {when} and stayed high."
        if direction == "down":
            return f"{variable} dropped at {when} and stayed low."
        return f"{variable} moved suddenly at {when} and stayed there."
    if item["rule"] == "forecast_residual":
        return f"{variable} was far from what the model expected at {when}."
    return f"{variable} looked unusual from {when}."


def extras_said(item):
    """
    One plain sentence per extra that had something to say. The modules' own
    wording is on the detail page; these say the same thing for anyone.
    """
    out = []
    accounts = item.get("explanations") or {}
    run = accounts.get("newtwork_run")
    if run:
        order = run.get("order") or []
        verdict = run.get("verdict")
        if verdict == "consistent_with_transport" and len(order) > 1:
            if order[0] == item["site"]:
                out.append(
                    f"It showed up here first, then at {', then '.join(order[1:])}, the way a spill travels."
                )
            else:
                before = order[: order.index(item["site"])] if item["site"] in order else order[:-1]
                out.append(
                    f"It reached {item['site']} after {', then '.join(before)}, the way a spill travels."
                )
        elif verdict == "catchment_wide":
            out.append(
                "Other sites changed at the same time, which is what rain or something creek-wide looks like."
            )
        elif verdict == "not_transport":
            out.append(
                "A site on a separate creek changed at the same moment, so this didn't travel down Strawberry Creek."
            )
        elif verdict == "local_to_this_site":
            out.append("No other site showed it.")
    bed = accounts.get("trial_bed")
    if bed:
        if bed.get("verdict") == "diagnosed" and bed.get("cause"):
            out.append(f"The way the readings moved looks most like {bed['cause']}.")
        elif bed.get("verdict") == "possible_new_type":
            out.append("The way the readings moved doesn't match any known kind of spill.")
        else:
            out.append("Trial Bed couldn't tell what it might be from the sensors at this site.")
    pool = accounts.get("settling_pool")
    if pool and pool.get("verdict") == "held_back":
        out.append(f"Settling Pool {pool['explanation']}.")
    return out


def item_view(store, item, ref):
    return {
        **item,
        "title": item["model"].replace("_", " ").title(),
        "label": LABELS.get(item["variable"], item["variable"]),
        "happened": what_happened(store, item, ref),
        "extras": extras_said(item),
        "when": moment(item["first_over"], ref),
        "day": day(item["first_over"]),
    }


def score_chart(store, item, before=36 * H, after=12 * H):
    """The score, the line it had to cross with and without rain, and rain, around one item."""
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
    }


# The 7 day preview on Advanced settings


def preview_counts(store, models, draft, days=7):
    """
    For each model, what the last days of checks would have noticed at the
    draft's values, and what they did notice. Same decide() rules the checks use.
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


def rain_note(draft, applied, model):
    """Said under the preview when the draft changes something it can't replay."""
    a = applied["models"].get(model, {})
    d = draft["models"].get(model, {})
    if (
        d.get("leave_out") != a.get("leave_out")
        or d.get("rain_multiplier") != a.get("rain_multiplier")
        or draft["rain"] != applied["rain"]
    ):
        return "Changes to rain or to sites left out change what the model reads, so they show from the next check rather than in this count."
    return ""


def model_titles(models, applied):
    """Dusk Crayfish is on. Cobble Shoal is off."""
    return " ".join(
        f"{m.title} is {'on' if applied['models'].get(m.name, {}).get('on') else 'off'}."
        for m in models
    )


def check_took(check):
    if not check or not check.get("finished_at"):
        return None
    return check["finished_at"] - check["started_at"]


def alerts_label(settings):
    choice = ws.alerts_choice(settings)
    return {key: label for key, label, _, _ in ws.PRESETS}.get(
        choice, "Custom, set in Advanced settings"
    )
