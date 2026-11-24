"""The Workbench's Flask app and every route."""

from __future__ import annotations

import base64
import json
import re
import secrets
import threading
import types
from datetime import timedelta
from pathlib import Path

import pandas as pd
from flask import (
    Blueprint,
    Flask,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask import session as cookie
from flask_wtf.csrf import CSRFProtect, generate_csrf

from strawberrywatch.dashboard import profiles, security
from strawberrywatch.watch import settings as ws
from strawberrywatch.workbench import calibrate, checks, data, events, export, store, text

HERE = Path(__file__).resolve().parent
DASHBOARD_STATIC = HERE.parent / "dashboard" / "static"
# What the Workbench borrows from the dashboard's static folder
SHARED = ("scmg-logo.png", "fonts/sen-latin.woff2", "fonts/sen-latin-ext.woff2", "fonts/OFL.txt")
SHARED_DIRS = ("uplot/",)
QUIET_MINUTES = 60
STATUS_EVERY = timedelta(minutes=5)
DRIFT_EVERY = timedelta(days=1)

bp = Blueprint("main", __name__)
csrf = CSRFProtect()


def create_app(folder=None, browser=False, login=None):
    folder = store.ready(folder or store.default_folder())
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        SECRET_KEY=secrets.token_hex(32),
        SESSION_COOKIE_NAME="workbench",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        # http on 127.0.0.1, where a Secure cookie wouldn't come back
        SESSION_COOKIE_SECURE=False,
        # A PNG of 3 charts, base64
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        WTF_CSRF_TIME_LIMIT=None,
        DEBUG=False,
        PROPAGATE_EXCEPTIONS=False,
    )
    app.debug = False
    app.jinja_env.lstrip_blocks = True
    app.jinja_env.trim_blocks = True
    csrf.init_app(app)
    app.extensions["strawberrywatch"] = types.SimpleNamespace(
        login=login or security.Login(),
        folder=folder,
        browser=browser,
        jobs={},
        lock=threading.Lock(),
        status_at=None,
    )
    security.init(app, profiles.LOCAL)
    app.register_blueprint(bp)
    return app


def ctx():
    return current_app.extensions["strawberrywatch"]


def folder():
    return ctx().folder


# Background work


def start_job(work, *args):
    """Runs work(*args) on a thread. GET /jobs/<id> says how it's going."""
    job_id = secrets.token_hex(8)
    jobs = ctx().jobs
    jobs[job_id] = {"state": "running", "message": ""}
    app = current_app._get_current_object()

    def run():
        with app.app_context():
            try:
                jobs[job_id]["result"] = work(*args)
                jobs[job_id]["state"] = "done"
            except Exception as exc:
                jobs[job_id]["state"] = "error"
                jobs[job_id]["message"] = reason(exc)

    threading.Thread(target=run, daemon=True).start()
    return job_id


def reason(exc):
    """The exception's first line, with any secret taken out."""
    lines = str(exc).strip().splitlines()
    first = lines[0] if lines else type(exc).__name__
    for key in store.SECRETS:
        value = store.secret(key)
        if value:
            first = first.replace(value, "")
    return first


@bp.route("/jobs/<job_id>")
def job(job_id):
    found = ctx().jobs.get(job_id)
    if found is None:
        abort(404)
    return jsonify({"state": found["state"], "message": found["message"]})


# Shared page parts


def in_use_kind():
    return store.setting(folder(), "source", "api")


def sites_all():
    from strawberrywatch import inventory as inv

    return list(inv.load().sites)


def site_name(site):
    return ws.site_name(site)


def current_session():
    found = None
    if cookie.get("tuning"):
        found = store.session(folder(), cookie["tuning"])
    if found is None or found["finished"]:
        found = store.unfinished_session(folder()) or found
    return found


def step_states(found, page=None):
    """Each step's (status, filled circle), from the session and the page open now."""
    loaded = bool(found and found["loaded"])
    explored = bool(found and found["explored"])
    reached = bool(found and (found["step"] or 1) >= 3)
    finished = bool(found and found["finished"])
    one = text.DONE if loaded else text.IN_PROGRESS
    if not loaded:
        two = text.LOCKED
    elif reached:
        two = text.DONE
    elif explored or page == 2:
        two = text.IN_PROGRESS
    else:
        two = text.NOT_STARTED
    if not explored:
        three = text.LOCKED
    elif finished:
        three = text.DONE
    elif reached or page == 3:
        three = text.IN_PROGRESS
    else:
        three = text.NOT_STARTED
    states = [one, two, three]
    return [
        {
            "status": status,
            "filled": status == text.DONE or page == i + 1,
            "current": page == i + 1,
        }
        for i, status in enumerate(states)
    ]


@bp.app_context_processor
def frame():
    words = {
        "plurals": text.PLURALS,
        "months": text.SHORT_MONTHS,
        "hours": [text.time_of_day(pd.Timestamp(2026, 1, 1, h)) for h in range(24)],
    }
    return {
        "theme": store.setting(folder(), "theme", "system"),
        "size": store.setting(folder(), "size", "normal"),
        "csrf": generate_csrf,
        "words": json.dumps(words, ensure_ascii=False),
        "browser": ctx().browser,
    }


def page(template, section, **values):
    steps = None
    if section == "tuning":
        steps = step_states(current_session(), values.get("step_page"))
    titles = {
        "home": "Home",
        "tuning": "Model tuning",
        "sensors": "Sensors",
        "events": "Events",
        "calibrations": "Past calibrations",
        "manual": "User manual",
        "settings": "Settings",
    }
    return render_template(
        template, section=section, page_title=titles[section], steps=steps, **values
    )


@bp.route("/login")
def login():
    return security.sign_in(ctx().login)


@bp.route("/shared/<path:filename>")
def shared(filename):
    if filename not in SHARED and not filename.startswith(SHARED_DIRS):
        abort(404)
    return send_from_directory(DASHBOARD_STATIC, filename)


# Sensor status


def refresh_status(kind):
    """Fetches the last day for every site in the background, at most every 5 minutes."""
    now = pd.Timestamp.now(tz="UTC")
    if not data.is_set_up(folder(), kind):
        return
    if ctx().status_at is not None and now - ctx().status_at < STATUS_EVERY:
        return
    ctx().status_at = now
    start_job(data.refresh_status, folder(), kind, sites_all())


def site_status(kind, now=None):
    """[{site, name, cells, last, quiet}] for every inventory site."""
    from strawberrywatch import inventory as inv

    now = now or pd.Timestamp.now(tz="UTC")
    inventory = inv.load()
    seen = data.last_seen(folder(), kind, sites_all())
    out = []
    for site in sites_all():
        have = set(inventory.site(site).scored_variables)
        cells, live_any, times = [], False, []
        for reading in text.READINGS:
            last = seen[site].get(reading)
            if last is not None:
                times.append(last)
            if reading not in have:
                cells.append({"reading": reading, "status": "none", "word": text.NONE})
                continue
            live = last is not None and now - last <= pd.Timedelta(minutes=QUIET_MINUTES)
            live_any |= live
            cells.append(
                {
                    "reading": reading,
                    "status": "live" if live else "quiet",
                    "word": text.LIVE if live else text.QUIET,
                    "last": last,
                }
            )
        last = max(times) if times else None
        out.append(
            {
                "site": site,
                "name": site_name(site),
                "cells": cells,
                "last": last,
                "ago": text.ago((now - last).total_seconds()) if last is not None else "",
                "quiet": not live_any,
            }
        )
    return out


# Creek map positions, from the mockup
MAP = {
    "botanical_garden": (70, 50, 70, 30),
    "south_fork_0": (180, 50, 180, 30),
    "south_fork_1": (290, 50, 290, 30),
    "south_fork_2": (400, 50, 400, 30),
    "south_fork_3": (510, 50, 510, 30),
    "kingman_hall": (70, 150, 70, 178),
    "university_house": (180, 150, 180, 178),
    "north_fork_0": (290, 150, 290, 178),
    "scnf010": (400, 150, 400, 178),
    "oxford": (640, 105, 684, 96),
    "codornices": (630, 168, 630, 190),
}


def creek_map(statuses=None, ticked=()):
    """Dots and labels for _creek_map.html, plus the sites the map has no place for."""
    dots, extra = [], []
    for site in sites_all():
        name = site_name(site)
        quiet = bool(statuses and statuses.get(site))
        if site not in MAP:
            extra.append({"site": site, "name": name, "ticked": site in ticked})
            continue
        x, y, lx, ly = MAP[site]
        dots.append(
            {
                "site": site,
                "name": name,
                "x": x,
                "y": y,
                "lx": lx,
                "ly": ly,
                "r": 10 if site == "oxford" else 9,
                "quiet": quiet,
                "ticked": site in ticked,
                "bold": quiet or site in ticked,
                "left": round(x / 720 * 100, 3),
                "top": round(y / 196 * 100, 3),
            }
        )
    return dots, extra


# Home


@bp.route("/")
def overview():
    kind = in_use_kind()
    refresh_status(kind)
    status = site_status(kind)
    quiet = [s["name"] for s in status if s["quiet"]]
    reporting = len(status) - len(quiet)
    dots, extra = creek_map({s["site"]: s["quiet"] for s in status})
    start_drift(kind)
    found = store.unfinished_session(folder())
    unfinished = None
    if found:
        model = found["model"] or guess_model(found["sites"])
        start, end = text.parse_date(found["start_date"]), text.parse_date(found["end_date"])
        step = found["step"] or 1
        unfinished = {
            "model": calibrate.title(model),
            "sites": text.site_group([site_name(s) for s in found["sites"]]),
            "dates_long": text.long_dates(start, end) if start and end else "",
            "step": step,
            "step_name": ("Pick data", "Explore", "Recalibrate")[step - 1],
            "href": url_for(
                ("main.tuning_pick", "main.tuning_explore", "main.tuning_recalibrate")[step - 1]
            ),
        }
    return page(
        "home.html",
        "home",
        reporting=reporting,
        total=len(status),
        quiet_sites=text.join_list(quiet),
        n_quiet_sites=len(quiet),
        dots=dots,
        extra=extra,
        unfinished=unfinished,
        models=model_rows(),
    )


def guess_model(sites):
    for model in calibrate.MODELS:
        if set(calibrate.model_sites(model)) <= set(sites):
            return model
    return calibrate.MODELS[0]


def model_rows():
    rows = []
    chosen = store.in_use(folder())
    for model in calibrate.MODELS:
        calibration_id = chosen.get(model, store.SHIPPED)
        row = {"model": model, "title": calibrate.title(model), "shipped": True, "drift": None}
        if calibration_id != store.SHIPPED:
            record = store.calibration_record(folder(), model, calibration_id)
            row["shipped"] = False
            row["saved_date"] = saved_date(record)
        result = store.drift_result(folder(), model, calibration_id)
        if result is not None:
            if result.get("no_reference"):
                row["drift"] = "none_here"
            elif result.get("drifted"):
                row["drift"] = "found"
                row["drifted"] = text.drift_phrase([tuple(p) for p in result["drifted"]], site_name)
            elif result.get("checked"):
                row["drift"] = "clear"
        rows.append(row)
    return rows


def saved_date(record):
    if not record:
        return ""
    moment = pd.Timestamp(record["saved_utc"], tz="UTC")
    return text.short_dates(text.local(moment).date())


def start_drift(kind):
    """Each model's drift check, at most once a day, in the background."""
    if not data.is_set_up(folder(), kind):
        return
    now = pd.Timestamp.now(tz="UTC")
    with ctx().lock:
        for model in calibrate.MODELS:
            calibration_id = calibrate.in_use(folder(), model)
            last = store.drift_result(folder(), model, calibration_id)
            if last and now - pd.Timestamp(last["computed_utc"], tz="UTC") < DRIFT_EVERY:
                continue
            key = f"drift:{model}"
            if ctx().jobs.get(key, {}).get("state") == "running":
                continue
            ctx().jobs[key] = {"state": "running", "message": ""}
            start_job(run_drift, model, kind, key)


def run_drift(model, kind, key):
    try:
        calibration_id, result = calibrate.drift(folder(), model, kind)
        store.save_drift(folder(), model, calibration_id, result)
    finally:
        ctx().jobs[key]["state"] = "done"


# Sensors


@bp.route("/sensors")
def sensors():
    kind = in_use_kind()
    refresh_status(kind)
    status = site_status(kind)
    quiet_only = request.args.get("show") == "quiet"
    shown = [s for s in status if s["quiet"]] if quiet_only else status
    chosen = request.args.get("site")
    by_site = {s["site"]: s for s in status}
    detail = by_site.get(chosen) or next((s for s in status if s["quiet"]), None) or status[0]
    counts = data.hour_counts(folder(), kind, detail["site"])
    marks = ["full" if c >= 4 else "half" if c else "empty" for c in counts]
    n_quiet = sum(1 for s in status if s["quiet"])
    last = detail["last"]
    return page(
        "sensors.html",
        "sensors",
        rows=shown,
        quiet_only=quiet_only,
        reporting=len(status) - n_quiet,
        total=len(status),
        n_quiet=n_quiet,
        site=detail["name"],
        site_id=detail["site"],
        time=text.time_of_day(text.local(last)) if last is not None else "",
        ago=detail["ago"],
        marks=marks,
        n_full=marks.count("full"),
        n_half=marks.count("half"),
        n_empty=marks.count("empty"),
        readings=[text.READING_NAMES[r] for r in text.READINGS],
    )


@bp.route("/export/sensors.csv")
def export_sensors():
    rows = []
    for s in site_status(in_use_kind()):
        for cell in s["cells"]:
            last = cell.get("last")
            local = text.local(last).strftime("%Y-%m-%d %H:%M") if last is not None else ""
            rows.append([s["site"], cell["reading"], cell["status"], local])
    body = export.csv_text(["site", "reading", "status", "last_reading_local"], rows)
    return attachment(body, "sensor_status.csv", "text/csv")


def attachment(body, name, mimetype):
    response = current_app.response_class(body, mimetype=mimetype)
    response.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return response


# Events


def event_rows(selected=None, kind=None):
    out = []
    for e in events.every(folder()):
        if kind and e["kind"] != kind:
            continue
        start, end = text.local(e["start"]).date(), text.local(e["end"]).date()
        out.append(
            {
                "id": e["id"],
                "name": e["name"],
                "kind": text.KINDS.get(e["kind"], text.KINDS["other"]),
                "sites": text.join_list(site_name(s) for s in e["sites"]),
                "dates": text.short_dates(start, end),
                "alert": text.YES if e["should_alert"] else text.NO,
                "source": text.SHIPPED if e["shipped"] else text.ADDED_HERE,
                "selected": e["id"] == selected,
            }
        )
    return out


def local_input(moment):
    return text.local(moment).strftime("%Y-%m-%dT%H:%M") if moment is not None else ""


@bp.route("/events")
@bp.route("/events/new", endpoint="event_new")
@bp.route("/events/<event_id>", endpoint="event_view")
def events_page(event_id=None, problem=False, form=None, is_new=None):
    kind = request.args.get("kind") or None
    if kind not in text.KINDS:
        kind = None
    editing = None
    if is_new is None:
        is_new = request.endpoint == "main.event_new"
    if event_id is not None:
        editing = events.one(folder(), event_id)
        if editing is None:
            abort(404)
    if is_new:
        now = pd.Timestamp.now(tz="UTC").floor("15min")
        editing = {
            "id": None,
            "name": "",
            "kind": "spill",
            "sites": [],
            "start": now - pd.Timedelta(hours=2),
            "end": now,
            "should_alert": True,
            "note": "",
            "shipped": False,
        }
    if editing is not None and form is not None:
        editing = {**editing, **form}
    editor = None
    if editing is not None:
        first = editing["sites"][0] if editing["sites"] else None
        editor = {
            "id": editing["id"],
            "new": editing["id"] is None,
            "name": editing["name"],
            "shipped": editing["shipped"],
            "kind": editing["kind"],
            "should_alert": editing["should_alert"],
            "note": editing["note"],
            "starts": local_input(editing["start"]),
            "ends": local_input(editing["end"]),
            "ticked": set(editing["sites"]),
            "first": first,
        }
    reading = request.args.get("reading", "conductivity")
    if reading not in text.READINGS:
        reading = "conductivity"
    return page(
        "events.html",
        "events",
        rows=event_rows(editing["id"] if editing else None, kind),
        kind=kind,
        kinds=text.KINDS,
        editor=editor,
        problem=problem,
        all_sites=[{"site": s, "name": site_name(s)} for s in sites_all()],
        site=site_name(editor["first"]) if editor and editor["first"] else "",
        reading_label=text.READING_LABELS[reading],
        reading=text.READING_NAMES[reading].lower() if reading != "ph" else "pH",
        start=text.time_of_day(text.local(editing["start"]))
        if editing and editing["start"] is not None
        else "",
        end=text.time_of_day(text.local(editing["end"]))
        if editing and editing["end"] is not None
        else "",
        reading_key=reading,
        reading_buttons=[(r, text.READING_NAMES[r]) for r in text.DEFAULT_READINGS],
        alert_kinds=list(text.ALERT_KINDS),
        yes=text.YES,
        no=text.NO,
    )


def parse_local(value):
    try:
        moment = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(moment):
        return None
    return moment.tz_localize(text.LOCAL).tz_convert("UTC")


@bp.route("/events/new", methods=["POST"], endpoint="event_create")
@bp.route("/events/<event_id>", methods=["POST"])
def event_save(event_id=None):
    sites = [s for s in request.form.getlist("sites") if s in sites_all()]
    start = parse_local(request.form.get("starts"))
    end = parse_local(request.form.get("ends"))
    kind = request.form.get("kind")
    kind = kind if kind in text.KINDS else "other"
    current = events.one(folder(), event_id) if event_id else None
    if event_id and current is None:
        abort(404)
    name = request.form.get("name", "").strip() if current is None else current["name"]
    fields = {
        "name": name,
        "kind": kind,
        "sites": sites,
        "start": start,
        "end": end,
        "should_alert": request.form.get("should_alert") == "yes",
        "note": request.form.get("note", "").strip(),
    }
    if not sites or start is None or end is None or end < start:
        return events_page(event_id, problem=True, form=fields, is_new=current is None), 400
    saved = events.save(folder(), event_id, fields)
    return redirect(url_for("main.event_view", event_id=saved))


@bp.route("/events/<event_id>/delete", methods=["POST"])
def event_delete(event_id):
    if not events.delete(folder(), event_id):
        abort(404)
    return redirect(url_for("main.events_page"))


@bp.route("/export/events.csv")
def export_events():
    rows = []
    for e in events.every(folder()):
        rows.append(
            [
                e["name"],
                text.KINDS.get(e["kind"], text.KINDS["other"]),
                "; ".join(site_name(s) for s in e["sites"]),
                text.local(e["start"]).strftime("%Y-%m-%d %H:%M"),
                text.local(e["end"]).strftime("%Y-%m-%d %H:%M"),
                text.YES if e["should_alert"] else text.NO,
                e["note"],
                text.SHIPPED if e["shipped"] else text.ADDED_HERE,
            ]
        )
    header = ["name", "kind", "sites", "start_local", "end_local", "should_alert", "note", "source"]
    return attachment(export.csv_text(header, rows), "events.csv", "text/csv")


@bp.route("/api/event-series")
def api_event_series():
    event = events.one(folder(), request.args.get("event", ""))
    site = request.args.get("site", "")
    reading = request.args.get("reading", "conductivity")
    if event is None or site not in sites_all() or reading not in data.STORED:
        abort(404)
    start, end = event["start"] - pd.Timedelta(hours=6), event["end"] + pd.Timedelta(hours=6)
    frame = None
    if event["shipped"]:
        tables, _ = events.shipped_tables(event, [site])
        frame = tables.get(site)
    if frame is None or frame.empty:
        frame = data.readings(folder(), in_use_kind(), [site], start, end).get(site)
    return series_json(frame, reading, start, end)


def series_json(frame, reading, start, end):
    if frame is None or reading not in frame.columns:
        return jsonify({"times": [], "values": []})
    values = frame[reading]
    values = values[(values.index >= start) & (values.index <= end)].dropna()
    return jsonify(
        {
            "times": [t.timestamp() for t in values.index],
            "values": values.tolist(),
        }
    )


# Model tuning


@bp.route("/tuning")
def tuning():
    return redirect(url_for("main.tuning_pick"))


@bp.route("/tuning/pick")
def tuning_pick():
    found = current_session()
    picked = found or {}
    model = request.args.get("model")
    start, end = text.last_days(7)
    defaults = {
        "source": in_use_kind(),
        "sites": [],
        "readings": list(text.DEFAULT_READINGS),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "leave_out_events": 1,
        "leave_out_rain": 1,
    }
    values = {k: picked.get(k, v) if picked else v for k, v in defaults.items()}
    if model in calibrate.MODELS:
        start, end = text.last_days(30)
        values.update(
            sites=calibrate.model_sites(model),
            readings=list(text.DEFAULT_READINGS),
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
    days = (text.parse_date(values["end_date"]) - text.parse_date(values["start_date"])).days + 1
    pairs = data.expected_pairs(values["sites"], values["readings"])
    dots, extra = creek_map(ticked=set(values["sites"]))
    last7, last30 = text.last_days(7), text.last_days(30)
    return page(
        "tuning_pick.html",
        "tuning",
        step_page=1,
        values=values,
        dots=dots,
        extra=extra,
        ticked_sites=text.join_list(site_name(s) for s in values["sites"]),
        readings_all=[(r, text.READING_NAMES[r]) for r in text.READINGS],
        models=[
            {
                "model": m,
                "title": calibrate.title(m),
                "sites": ",".join(calibrate.model_sites(m)),
                "readings": ",".join(text.DEFAULT_READINGS),
            }
            for m in calibrate.MODELS
        ],
        set_up={k: data.is_set_up(folder(), k) for k in data.KINDS},
        past=[
            {
                "id": e["id"],
                "name": e["name"],
                "from": (text.local(e["start"]).date() - timedelta(days=3)).isoformat(),
                "to": (text.local(e["end"]).date() + timedelta(days=3)).isoformat(),
            }
            for e in events.every(folder())
        ],
        last7=[d.isoformat() for d in last7],
        last30=[d.isoformat() for d in last30],
        days=text.count(days, "day"),
        sites=text.count(len(values["sites"]), "site"),
        readings=text.count(len(values["readings"]), "reading"),
        total=text.count(data.PER_DAY * days * len(pairs), "reading"),
        pairs=json.dumps(pairs_by_site()),
        job=request.args.get("job"),
        source_name=text.SOURCE_NAMES.get(values["source"], ""),
    )


def pairs_by_site():
    from strawberrywatch import inventory as inv

    inventory = inv.load()
    return {s: list(inventory.site(s).scored_variables) for s in sites_all()}


@bp.route("/tuning/pick", methods=["POST"])
def tuning_pick_post():
    form = request.form
    kind = form.get("source") if form.get("source") in data.KINDS else "api"
    sites = [s for s in form.getlist("sites") if s in sites_all()]
    readings = [r for r in form.getlist("readings") if r in text.READINGS]
    start, end = text.parse_date(form.get("start_date")), text.parse_date(form.get("end_date"))
    if not sites or not readings or not start or not end or end < start:
        return redirect(url_for("main.tuning_pick"))
    fields = {
        "source": kind,
        "sites": sites,
        "readings": readings,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "leave_out_events": int(form.get("leave_out_events") == "yes"),
        "leave_out_rain": int(form.get("leave_out_rain") == "yes"),
        "loaded": 0,
        "explored": 0,
        "step": 1,
    }
    found = current_session()
    if found and not found["finished"]:
        store.update_session(folder(), found["id"], check=None, **fields)
        session_id = found["id"]
    else:
        session_id = store.new_session(folder(), **fields)
    cookie["tuning"] = session_id
    store.set_setting(folder(), "source", kind)
    if not data.is_set_up(folder(), kind):
        return redirect(url_for("main.tuning_pick"))
    job_id = start_job(load_readings, session_id)
    return redirect(url_for("main.tuning_pick", job=job_id))


def load_readings(session_id):
    found = store.session(folder(), session_id)
    start, end = data.utc(found["start_date"]), data.utc(found["end_date"], end=True)
    # Model sites too, so step 3 can re-fit whichever model is picked
    names = sorted(set(found["sites"]))
    data.fetch(folder(), found["source"], names, start, end)
    tables = data.readings(folder(), found["source"], names, start, end)
    if not any(not f.empty for f in tables.values()):
        raise LookupError("empty")
    store.update_session(folder(), session_id, loaded=1)
    return True


@bp.route("/tuning/drift/<model>")
def tuning_drift(model):
    if model not in calibrate.MODELS:
        abort(404)
    result = store.drift_result(folder(), model, calibrate.in_use(folder(), model)) or {}
    drifted = result.get("drifted") or []
    start, end = text.last_days(checks.DRIFT_DAYS)
    session_id = store.new_session(
        folder(),
        source=in_use_kind(),
        sites=sorted({p[0] for p in drifted}) or calibrate.model_sites(model),
        readings=[r for r in text.READINGS if r in {p[1] for p in drifted}]
        or list(text.DEFAULT_READINGS),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        leave_out_events=1,
        leave_out_rain=1,
        model=model,
        loaded=1,
    )
    cookie["tuning"] = session_id
    return redirect(url_for("main.tuning_explore"))


def need(found, step):
    """The page to go to instead when this step isn't unlocked yet, or None."""
    if found is None or not found["loaded"]:
        return redirect(url_for("main.tuning_pick"))
    if step == 3 and not found["explored"]:
        return redirect(url_for("main.tuning_explore"))
    return None


@bp.route("/tuning/explore")
def tuning_explore():
    found = current_session()
    back = need(found, 2)
    if back:
        return back
    store.update_session(folder(), found["id"], explored=1, step=max(found["step"] or 1, 2))
    found = store.session(folder(), found["id"])
    site = request.args.get("site")
    if site not in found["sites"]:
        site = found["sites"][0]
    reading = request.args.get("reading")
    if reading not in found["readings"]:
        reading = found["readings"][0]
    start, end = data.utc(found["start_date"]), data.utc(found["end_date"], end=True)
    frame = data.readings(folder(), found["source"], [site], start, end).get(site)
    values = (
        frame[reading] if frame is not None and reading in frame.columns else pd.Series(dtype=float)
    )
    n_missing, n_expected = checks.missing(values, start, end)
    n_far = int(checks.far(values).sum())
    dates_long = text.long_dates(
        text.parse_date(found["start_date"]), text.parse_date(found["end_date"])
    )
    return page(
        "tuning_explore.html",
        "tuning",
        step_page=2,
        found=found,
        site=site_name(site),
        site_id=site,
        reading=text.READING_NAMES[reading].lower() if reading != "ph" else "pH",
        reading_key=reading,
        reading_title=text.READING_TITLES[reading],
        reading_buttons=[(r, text.READING_NAMES[r]) for r in found["readings"]],
        site_options=[(s, site_name(s)) for s in found["sites"]],
        dates_long=dates_long,
        bin=text.bin_label(reading),
        missing=text.number(n_missing),
        expected=text.number(n_expected),
        stretches=text.count(checks.stuck(values), "stretch"),
        far_count=text.count(n_far, "reading"),
        notes=found["notes"] or "",
    )


def picked_values(found, site, reading):
    start, end = data.utc(found["start_date"]), data.utc(found["end_date"], end=True)
    frame = data.readings(folder(), found["source"], [site], start, end).get(site)
    if frame is None or reading not in frame.columns:
        return pd.Series(dtype=float), start, end
    return frame[reading].dropna(), start, end


def chart_args():
    found = current_session()
    site, reading = request.args.get("site", ""), request.args.get("reading", "")
    if found is None or site not in found["sites"] or reading not in text.READINGS:
        abort(404)
    return found, site, reading


@bp.route("/api/series")
def api_series():
    found, site, reading = chart_args()
    values, start, end = picked_values(found, site, reading)
    flagged = checks.far(values)
    grid = data.grid(start, end)
    shown = values.reindex(grid)
    flags = values.where(flagged.reindex(values.index, fill_value=False)).reindex(grid)
    return jsonify(
        {
            "times": [t.timestamp() for t in grid],
            "values": checks.clean_nan(shown.tolist()),
            "flags": checks.clean_nan(flags.tolist()),
        }
    )


@bp.route("/api/typical-day")
def api_typical_day():
    found, site, reading = chart_args()
    values, _, _ = picked_values(found, site, reading)
    hours, median, low, high = checks.typical_day(values)
    return jsonify(
        {
            "hours": hours,
            "median": checks.clean_nan(median),
            "low": checks.clean_nan(low),
            "high": checks.clean_nan(high),
        }
    )


@bp.route("/api/histogram")
def api_histogram():
    found, site, reading = chart_args()
    values, _, _ = picked_values(found, site, reading)
    edges, counts = checks.histogram(values, text.BINS[reading])
    return jsonify({"edges": edges, "counts": counts})


@bp.route("/tuning/notes", methods=["POST"])
def tuning_notes():
    found = current_session()
    if found is None:
        abort(404)
    store.update_session(folder(), found["id"], notes=request.form.get("notes", "")[:4000])
    return jsonify({"ok": True})


@bp.route("/export/readings.csv")
def export_readings():
    found = current_session()
    if found is None or not found["loaded"]:
        abort(404)
    start, end = data.utc(found["start_date"]), data.utc(found["end_date"], end=True)
    tables = data.readings(folder(), found["source"], found["sites"], start, end)
    rows = []
    for site in found["sites"]:
        frame = tables.get(site)
        if frame is None:
            continue
        for reading in found["readings"]:
            if reading not in frame.columns:
                continue
            for moment, value in frame[reading].dropna().items():
                local = text.local(moment).strftime("%Y-%m-%d %H:%M")
                rows.append([site, reading, local, value, text.UNITS[reading]])
    body = export.csv_text(["site", "reading", "time_local", "value", "unit"], rows)
    return attachment(body, "readings.csv", "text/csv")


@bp.route("/export/png", methods=["POST"])
def export_png():
    encoded = request.form.get("data", "")
    match = re.match(r"^data:image/png;base64,([A-Za-z0-9+/=]+)$", encoded)
    if not match:
        abort(400)
    return attachment(base64.b64decode(match.group(1)), "charts.png", "image/png")


def pending_path(session_id):
    path = Path(folder()) / "pending"
    path.mkdir(exist_ok=True)
    return path / f"{session_id}.json"


@bp.route("/tuning/recalibrate")
def tuning_recalibrate():
    found = current_session()
    saved = request.args.get("saved") == "1"
    if saved and cookie.get("tuning"):
        found = store.session(folder(), cookie["tuning"])
    back = None if saved else need(found, 3)
    if back:
        return back
    store.update_session(folder(), found["id"], step=3)
    found = store.session(folder(), found["id"])
    check = found["check"]
    picked = set(found["sites"])
    model = request.args.get("model") or (check or {}).get("model") or found["model"]
    choice = (
        request.args.get("false_alarms")
        or (check or {}).get("false_alarms")
        or text.DEFAULT_FALSE_ALARMS
    )
    options = []
    for m in calibrate.MODELS:
        sites = calibrate.model_sites(m)
        lacking = [s for s in sites if s not in picked]
        options.append(
            {
                "model": m,
                "title": calibrate.title(m),
                "text": text.MODEL_TEXT[m],
                "model_sites": text.join_list(site_name(s) for s in sites),
                "n_model_sites": len(sites),
                "missing_sites": text.join_list(site_name(s) for s in lacking),
                "ready": not lacking,
            }
        )
    ready = [o["model"] for o in options if o["ready"]]
    if model not in ready:
        model = ready[0] if ready else None
    rows = []
    checked = bool(check and check["model"] == model and check["false_alarms"] == choice)
    if checked:
        rows = [
            {
                **r,
                "alert": text.YES if r["should_alert"] else text.NO,
                "now": r["results"][0],
                "after": r["results"][1],
            }
            for r in check["events"]
        ]
    elif model:
        sites = set(calibrate.model_sites(model))
        for e in events.every(folder()):
            if sites & set(e["sites"]):
                rows.append(
                    {
                        "name": e["name"],
                        "dates": text.short_dates(
                            text.local(e["start"]).date(), text.local(e["end"]).date()
                        ),
                        "alert": text.YES if e["should_alert"] else text.NO,
                        "now": text.NOT_RUN,
                        "after": text.NOT_RUN,
                    }
                )
    return page(
        "tuning_recalibrate.html",
        "tuning",
        step_page=3,
        options=options,
        model=model,
        choice=choice,
        choices=text.FALSE_ALARMS,
        rows=rows,
        checked=checked,
        before=check["before"] if checked else "",
        after=check["after"] if checked else "",
        check_key=f"{check['model']}:{check['false_alarms']}" if check else "",
        job=request.args.get("job"),
        saved=saved,
    )


@bp.route("/tuning/check", methods=["POST"])
def tuning_check():
    found = current_session()
    back = need(found, 3)
    if back:
        return back
    model = request.form.get("model")
    choice = request.form.get("false_alarms")
    if model not in calibrate.MODELS or choice not in text.FALSE_ALARMS:
        abort(400)
    if not set(calibrate.model_sites(model)) <= set(found["sites"]):
        abort(400)
    store.update_session(folder(), found["id"], model=model, false_alarms=choice, check=None)
    job_id = start_job(check_job, found["id"], model, choice)
    return redirect(
        url_for("main.tuning_recalibrate", job=job_id, model=model, false_alarms=choice)
    )


def check_job(session_id, model, choice):
    found = store.session(folder(), session_id)
    result = calibrate.run_check(folder(), found, model, choice)
    if result is None:
        raise LookupError("empty")
    pending_path(session_id).write_text(json.dumps(result))
    summary = {
        k: result[k] for k in ("model", "false_alarms", "calibration", "events", "before", "after")
    }
    store.update_session(folder(), session_id, check=summary)
    return True


@bp.route("/tuning/save", methods=["POST"])
def tuning_save():
    found = current_session()
    if found is None or not found["check"]:
        abort(400)
    path = pending_path(found["id"])
    if not path.is_file():
        abort(400)
    result = json.loads(path.read_text())
    if result["model"] != request.form.get("model") or result["false_alarms"] != request.form.get(
        "false_alarms"
    ):
        abort(400)
    calibrate.save(folder(), found, result)
    path.unlink()
    store.update_session(folder(), found["id"], finished=1, step=3)
    cookie["tuning"] = found["id"]
    return redirect(url_for("main.tuning_recalibrate", saved=1))


# Past calibrations


def card(calibration_id, record, chosen):
    found = {
        "id": calibration_id,
        "in_use": chosen == calibration_id,
        "saved_date": saved_date(record) if record else "",
    }
    if record:
        start, end = text.parse_date(record["start_date"]), text.parse_date(record["end_date"])
        found.update(
            sites=text.site_group([site_name(s) for s in record["sites"]]),
            dates_short=text.short_dates(start, end),
            source_name=text.SOURCE_NAMES.get(record["source"], ""),
            left_out=text.left_out(record["leave_out_events"], record["leave_out_rain"]),
            false_alarms=text.FALSE_ALARMS.get(record["false_alarms"], ""),
            note=record.get("notes") or "",
        )
    return found


@bp.route("/calibrations")
def calibrations():
    chosen = store.in_use(folder())
    sections = []
    for model in calibrate.MODELS:
        current = chosen.get(model, store.SHIPPED)
        saved = [card(cid, rec, current) for cid, rec in store.saved_calibrations(folder(), model)]
        sections.append(
            {
                "model": model,
                "title": calibrate.title(model),
                "shipped_in_use": current == store.SHIPPED,
                "saved": saved,
            }
        )
    return page(
        "calibrations.html", "calibrations", sections=sections, used=request.args.get("used")
    )


@bp.route("/calibrations/<model>/<calibration_id>/use", methods=["POST"])
def calibration_use(model, calibration_id):
    if model not in calibrate.MODELS:
        abort(404)
    if calibration_id != store.SHIPPED and store.verified(folder(), model, calibration_id) is None:
        abort(404)
    store.use(folder(), model, calibration_id)
    return redirect(url_for("main.calibrations", used=f"{model}:{calibration_id}"))


def saved_or_404(model, calibration_id):
    if model not in calibrate.MODELS:
        abort(404)
    record = store.calibration_record(folder(), model, calibration_id)
    if record is None or store.verified(folder(), model, calibration_id) is None:
        abort(404)
    return record


@bp.route("/calibrations/<model>/<calibration_id>/compare")
def calibration_compare(model, calibration_id):
    record = saved_or_404(model, calibration_id)
    rows = calibrate.compare(folder(), model, calibration_id, record["source"])
    return page(
        "compare.html",
        "calibrations",
        mode="compare",
        model=calibrate.title(model),
        saved_date=saved_date(record),
        rows=[
            {
                **r,
                "alert": text.YES if r["should_alert"] else text.NO,
                "now": r["results"][0],
                "after": r["results"][1],
            }
            for r in rows
        ],
    )


@bp.route("/calibrations/<model>/<calibration_id>/results")
def calibration_results(model, calibration_id):
    record = saved_or_404(model, calibration_id)
    rows = [
        {
            **r,
            "alert": text.YES if r["should_alert"] else text.NO,
            "now": r["results"][0],
            "after": r["results"][1],
        }
        for r in record.get("events", [])
    ]
    return page(
        "compare.html",
        "calibrations",
        mode="results",
        model=calibrate.title(model),
        saved_date=saved_date(record),
        rows=rows,
    )


@bp.route("/calibrations/<model>/<calibration_id>/summary.pdf")
def calibration_summary(model, calibration_id):
    record = saved_or_404(model, calibration_id)
    found = card(calibration_id, record, None)
    fields = [
        calibrate.title(model),
        found["saved_date"],
        f"{found['sites']}, {found['dates_short']}, from {found['source_name']}",
        found["left_out"],
        found["false_alarms"],
        found["note"],
    ]
    rows = [
        (r["name"], text.YES if r["should_alert"] else text.NO, r["results"][0], r["results"][1])
        for r in record.get("events", [])
    ]
    return attachment(
        export.summary_pdf(fields, rows), f"{model}_{calibration_id}.pdf", "application/pdf"
    )


# Settings


@bp.route("/settings")
def settings():
    tested = cookie.pop("tested", None)
    return page(
        "settings.html",
        "settings",
        has_token=bool(store.secret("api_token")),
        host=store.setting(folder(), "mysql_host", "") or "",
        database=store.setting(folder(), "mysql_database", "") or "",
        username=store.setting(folder(), "mysql_user", "") or "",
        csv_folder=store.setting(folder(), "csv_folder", "") or "",
        tested=tested,
        reason=(tested or {}).get("reason", ""),
        saved=request.args.get("saved"),
        cleared=request.args.get("cleared") == "1",
    )


@bp.route("/settings", methods=["POST"])
def settings_post():
    form = request.form
    what = form.get("what")
    if what == "token" and form.get("api_token", "").strip():
        store.set_secret("api_token", form["api_token"].strip())
    elif what == "mysql":
        for key in ("mysql_host", "mysql_database", "mysql_user"):
            store.set_setting(folder(), key, form.get(key, "").strip())
        if form.get("mysql_password"):
            store.set_secret("mysql_password", form["mysql_password"])
    elif what == "csv":
        store.set_setting(folder(), "csv_folder", form.get("csv_folder", "").strip())
    elif what == "size" and form.get("size") in ("normal", "larger"):
        store.set_setting(folder(), "size", form["size"])
    elif what == "theme" and form.get("theme") in ("system", "light", "dark"):
        store.set_setting(folder(), "theme", form["theme"])
    else:
        abort(400)
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify({"ok": True})
    return redirect(url_for("main.settings", saved=what))


@bp.route("/settings/test/<kind>", methods=["POST"])
def settings_test(kind):
    if kind not in ("api", "mysql"):
        abort(404)
    # The MySQL form's Test connection button sends the fields as typed
    if kind == "mysql" and "mysql_host" in request.form:
        for key in ("mysql_host", "mysql_database", "mysql_user"):
            store.set_setting(folder(), key, request.form.get(key, "").strip())
        if request.form.get("mysql_password"):
            store.set_secret("mysql_password", request.form["mysql_password"])
    try:
        if kind == "api":
            test_api()
        else:
            from strawberrywatch.ingest import sql_client

            sql_client.ping(data.credentials(folder()))
        cookie["tested"] = {"kind": kind, "ok": True}
    except Exception as exc:
        cookie["tested"] = {"kind": kind, "ok": False, "reason": reason(exc)}
    return redirect(url_for("main.settings"))


def test_api():
    import requests

    from strawberrywatch.config import Config

    token = store.secret("api_token")
    if not token:
        raise LookupError("API token")
    now = pd.Timestamp.now(tz="UTC")
    response = requests.get(
        Config.API_BASE_URL,
        headers={"Authorization": f"Token {token}"},
        params=[
            ("site", "south_fork_1"),
            ("start", (now - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")),
            ("end", now.strftime("%Y-%m-%dT%H:%M:%S")),
            ("vars", "Meter_Hydros21_Cond"),
        ],
        timeout=20,
    )
    response.raise_for_status()


@bp.route("/settings/clear-readings", methods=["POST"])
def settings_clear():
    store.clear_readings(folder())
    ctx().status_at = None
    return redirect(url_for("main.settings", cleared=1))


# User manual


def manual_pages():
    """Each page of the repository manual's Workbench pages appendix, as its own document."""
    path = HERE.parents[1] / "USER_MANUAL.md"
    if not path.is_file():
        return {}
    pages, current, inside = {}, None, False
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            inside, current = line.endswith("Workbench pages"), None
        elif line == "---":
            current = None
        elif inside and line.startswith("### "):
            current = pages.setdefault(text.slug(line[4:]), [f"# {line[4:]}"])
        elif current is not None:
            # a page's own subheadings sit two levels down in the manual
            current.append("## " + line[5:] if line.startswith("#### ") else line)
    return {slug: "\n".join(lines).strip() + "\n" for slug, lines in pages.items()}


def manual_text(slug):
    return manual_pages().get(slug)


@bp.route("/manual")
def manual_index():
    query = request.args.get("q", "").strip()
    if not query:
        return redirect(url_for("main.manual_page", slug=text.slug(text.MANUAL_PAGES[0])))
    matches = []
    for title in text.MANUAL_PAGES:
        body = manual_text(text.slug(title)) or ""
        if query.lower() in body.lower():
            matches.append({"title": title, "slug": text.slug(title)})
    return page(
        "manual.html",
        "manual",
        contents=contents(None),
        query=query,
        matches=matches,
        article=None,
    )


def contents(current):
    return [
        {"title": t, "slug": text.slug(t), "current": text.slug(t) == current}
        for t in text.MANUAL_PAGES
    ]


@bp.route("/manual/<slug>")
def manual_page(slug):
    import markdown

    titles = {text.slug(t): t for t in text.MANUAL_PAGES}
    body = manual_text(slug) if slug in titles else None
    if body is None:
        abort(404)
    # The page's # title becomes the article heading. Everything under it moves down a level
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    html = markdown.markdown("\n".join(lines))
    html = html.replace("<h2>", "<h3>").replace("</h2>", "</h3>")
    order = list(titles)
    i = order.index(slug)
    return page(
        "manual.html",
        "manual",
        contents=contents(slug),
        query="",
        matches=None,
        article={"title": titles[slug], "html": html},
        prev_title=titles[order[i - 1]] if i > 0 else "",
        prev_slug=order[i - 1] if i > 0 else "",
        next_title=titles[order[i + 1]] if i + 1 < len(order) else "",
        next_slug=order[i + 1] if i + 1 < len(order) else "",
    )
