"""The routes. What to show is worked out in view.py, the layout is in the templates."""

from __future__ import annotations

import copy
import csv
import io
import json
import logging
import time
from datetime import date, timedelta

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from strawberrywatch import catalog
from strawberrywatch.dashboard import examples, forms, mail, security, view
from strawberrywatch.watch import model_updates, sources
from strawberrywatch.watch import models as watch_models
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st

bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

NAV = (
    ("main.overview", "Overview"),
    ("main.unusual", "Alerts"),
    ("main.history", "History"),
    ("main.models", "Models"),
    ("main.settings", "Settings"),
)
SETTINGS_PAGES = {"main.settings", "main.advanced", "main.review"}
# Which tab each page sits under
TAB_OF = {
    "main.site": "main.overview",
    "main.history_day": "main.history",
    "main.notes": "main.history",
    "main.notes_export": "main.history",
    "main.alert_examples": "main.unusual",
    "main.email": "main.settings",
    "main.email_sending": "main.settings",
    "main.remove_recipient": "main.settings",
    "main.model_updates_page": "main.settings",
    "main.update_models": "main.settings",
    "main.undo_model_update": "main.settings",
    "main.email_test": "main.settings",
    "main.relabel": "main.settings",
    "main.person": "main.settings",
    "main.save_label": "main.settings",
    "main.copy_list": "main.settings",
    "main.resume_checks": "main.overview",
    **{page: "main.settings" for page in SETTINGS_PAGES},
}


def ctx():
    return current_app.extensions["strawberrywatch"]


def now():
    """The real clock, or the one a test fixed."""
    return view.reference_now(ctx().clock)


def draft_owner():
    return security.owner() or st.LOCAL


def not_set_up():
    """
    Only reached on a store that skipped setup. The local app sets up when it
    starts, and the website runs setup_tables.
    """
    abort(503, "StrawberryWatch isn't set up yet. Restart it to finish setting up.")


def sites():
    return view.site_order(ctx().inventory)


def site_labels():
    inventory = ctx().inventory
    return {s: inventory.site(s).name for s in inventory.sites}


@bp.app_template_filter("site_name")
def site_name_filter(site):
    """South Fork 1 for south_fork_1, in any template."""
    return view.site_name(site, ctx().inventory)


@bp.app_template_filter("site_place")
def site_place_filter(site):
    """Stephens Hall for south_fork_1, or "" when the name has no landmark."""
    return view.site_place(site, ctx().inventory)


def applied():
    """What's applied, with an entry for any model registered since it was saved."""
    return ws.with_new_models(ctx().store.applied()[1], ctx().models)


def working_copy():
    """The draft if there is one, otherwise a copy of what's applied."""
    draft = ctx().store.draft(draft_owner())
    return ws.with_new_models(draft, ctx().models) if draft else copy.deepcopy(applied())


def save_draft(settings):
    """Keep the draft, or drop it once it's back to what's applied."""
    if not ws.describe(applied(), settings, ctx().models):
        ctx().store.discard_draft(draft_owner())
    else:
        ctx().store.save_draft(settings, draft_owner())


def unapplied():
    draft = ctx().store.draft(draft_owner())
    current = applied()
    if draft is None or current is None:
        return 0
    return len(ws.describe(current, draft, ctx().models))


def banner_text(count):
    if not count:
        return ""
    if count == 1:
        return "1 unsaved change. Save or discard it at the bottom of the page."
    return f"{count} unsaved changes. Save or discard them at the bottom of the page."


def wants_json():
    return request.headers.get("X-Requested-With") == "fetch"


@bp.app_context_processor
def common():
    signed_in = session.get("signed_in") if ctx().profile.token_login else bool(security.owner())
    if not signed_in:
        return {"nav": [], "profile": ctx().profile, "public": True}
    page = request.endpoint
    current = TAB_OF.get(page, page)
    if page == "main.noticed":
        current = "main.unusual" if request.args.get("from") == "unusual" else "main.history"
    unusual = 0
    if not g.get("error_page") and applied() is not None:
        unusual = len(view.recent_noticed(ctx().store, now()))
    nav = [
        {
            "href": url_for(endpoint),
            "text": text,
            "active": endpoint == current,
            "count": unusual if endpoint == "main.unusual" else 0,
        }
        for endpoint, text in NAV
    ]
    count = unapplied() if page in SETTINGS_PAGES and not g.get("error_page") else 0
    return {
        "nav": nav,
        "email_on": not g.get("error_page") and mail.is_on(ctx().store),
        "unsaved": count,
        "banner": banner_text(count),
        "settings_page": page in SETTINGS_PAGES,
        "profile": ctx().profile,
        "can_edit": security.can_edit(),
        "readings": None if g.get("error_page") else _readings_line(),
        "paused": not g.get("error_page") and ctx().store.pref("checks_paused") == "yes",
        "model_updates": ctx().updater is not None,
        # The site's own sign-out page on the website, ours locally
        "sign_out": current_app.config.get("SIGN_OUT_URL")
        or (url_for("main.sign_out") if ctx().profile.token_login else None),
    }


def _readings_line():
    """
    The "where the data comes from" line at the top of every page. Also says
    when the last check fell back to the API.
    """
    store = ctx().store
    config = store.source()
    if config is None:
        return None
    last = store.last_check(kinds=sources.LIVE)
    if config["kind"] == sources.SITE and last and last.get("source") == sources.API:
        return {
            "words": "Strawberry Creek API, as a backup. The sensor database couldn't be reached."
        }
    return {"words": _readings_words(config["kind"])}


def _readings_words(kind):
    return {
        sources.API: "Strawberry Creek API",
        sources.SITE: "strawberrycreek.org sensor database",
    }.get(kind, "")


@bp.route("/login")
def login():
    return security.sign_in(ctx().login)


@bp.route("/signout")
def sign_out():
    """The local app's Sign out. Its link is used up, and signing in again takes a restart."""
    session.clear()
    return render_template("signed_out.html", signed_out=True)


# Overview and Alerts


@bp.route("/")
def overview():
    store, inventory, models = ctx().store, ctx().inventory, ctx().models
    current = applied()
    if store.source() is None or current is None:
        not_set_up()
    ref = now()
    statuses = view.site_statuses(store, inventory, current, ref)
    recent = view.recent_noticed(store, ref)
    stale = view.stale_since(store, ref)
    arrows = []
    for item in recent:
        order = (item["explanations"].get("newtwork_run") or {}).get("order") or []
        if len(order) > 1:
            arrows = view.flow_path(inventory, order)
            break
    last = store.last_readings(live=True)
    rows = []
    for site in sites():
        newest = last.get(site, {}).get("time")
        rows.append(
            {
                "site": site,
                "name": inventory.site(site).name,
                "status": statuses[site],
                "last": None if newest is None else view.ago(newest, ref),
                "marks": view.reporting_marks(store, site, ref),
            }
        )
    flagged = sorted({n["site"] for n in recent}, key=sites().index)
    late = [s for s in sites() if statuses[s]["key"] == "late"]
    reporting = sum(1 for s in sites() if statuses[s]["key"] != "late")
    last_check = store.last_check(kinds=sources.LIVE)
    return render_template(
        "overview.html",
        waiting=view.first_check_pending(store),
        stale=stale,
        stale_text=_stale_text(stale),
        status=_status_line(store, models, current),
        flagged=flagged,
        late=late,
        facts=[
            {"label": "Sites reporting", "value": f"{reporting} of {len(sites())}"},
            {"label": "Alerts, last 24 hours", "value": str(len(recent))},
            {
                "label": "Last check",
                "value": view.clock(last_check["started_at"]) if last_check else "None yet",
            },
        ],
        map=view.creek_map(inventory, statuses, arrows),
        travelled=bool(arrows),
        rows=rows,
    )


@bp.route("/unusual")
def unusual():
    store = ctx().store
    if store.source() is None or applied() is None:
        not_set_up()
    ref = now()
    recent = [
        view.item_view(store, n, ref)
        for n in sorted(
            view.recent_noticed(store, ref), key=lambda n: n["first_over"], reverse=True
        )
    ]
    charts = {item["id"]: view.reading_chart(store, item, ref) for item in recent}
    return render_template("unusual.html", recent=recent, charts=charts)


@bp.route("/alerts/examples")
def alert_examples():
    """Every way an alert can look, filled with made-up data, for checking the wording."""
    ref = now()
    samples = examples.items(ref)
    for s in samples:
        s["history"] = {
            "kind": "unusual",
            "text": _history_text(s["view"]),
            "href": f"#detail-{s['view']['id']}",
            "notes": [],
            "clock": view.clock(s["view"]["first_over"]),
        }
        s["subject"], opener, blocks = mail.alert_parts([s["view"]], ctx().inventory)
        s["email"] = mail.as_text(opener, blocks, _example_footer())
    samples[0]["history"]["notes"] = [
        {
            "author": "ana",
            "when": view.moment(ref - 3600, ref),
            "text": "Checked on site, the water looked cloudy.",
        }
    ]
    several = [s["view"] for s in samples[:3]]
    subject, opener, blocks = mail.alert_parts(several, ctx().inventory)
    return render_template(
        "alert_examples.html",
        samples=samples,
        sentences=examples.sentences(ref, samples),
        boxes=examples.overview_boxes(ref),
        reading=examples.reading_chart(samples[0]["view"]),
        score=examples.score_chart(samples[0]["view"]),
        several={"subject": subject, "email": mail.as_text(opener, blocks, _example_footer())},
        website=not ctx().profile.email_in_env,
    )


def _example_footer():
    link = mail.unsubscribe_link(_address(), "a-made-up-token")
    return mail.footer_text(link)


@bp.route("/alerts/examples/email/<kind>")
def alert_example_email(kind):
    """An example email's HTML version, opened on its own like a mail app would show it."""
    ref = now()
    samples = examples.items(ref)
    images = {name: url_for("static", filename=path) for name, path in mail.LOGOS.items()}
    unsubscribe = mail.unsubscribe_link(_address(), "a-made-up-token")
    if kind == "several":
        chosen = [s["view"] for s in samples[:3]]
    elif kind.isdigit() and 1 <= int(kind) <= len(samples):
        chosen = [samples[int(kind) - 1]["view"]]
    else:
        abort(404)
    subject, opener, blocks = mail.alert_parts(chosen, ctx().inventory)
    _text, html = mail.alert_email(subject, opener, blocks, unsubscribe, images)
    # Emails carry their styles inline. This one page allows inline styles
    g.csp = "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'"
    return html


def _stale_text(stale):
    if stale is None:
        return ""
    if stale["since"] is None:
        return "The first check started over 45 minutes ago and hasn't finished."
    words = view.when_words(stale["since"], now()).removeprefix("Today at ")
    if words.startswith("Yesterday"):
        words = "yesterday" + words[len("Yesterday") :]
    return f"No check has finished since {words}."


def _status_line(store, models, current):
    last = store.last_check(kinds=sources.LIVE)
    watcher = ctx().watcher
    lines = []
    if watcher.busy:
        lines.append("Check running.")
    if last:
        nxt = watcher.next_at or (last["started_at"] + ws.INTERVAL_MINUTES * 60)
        lines.append(
            f"Last check {view.clock(last['started_at'])}. Next check around {view.clock(nxt)}."
        )
    else:
        lines.append("First check starting.")
    if last and last.get("finished_at") and last.get("message"):
        if last["message"].startswith(("Skipped:", "Stopped:")):
            lines.append(f"{last['message']}, at {view.clock(last['started_at'])}.")
        elif last["message"] == sources.UNREACHABLE:
            lines.append(
                f"{sources.UNREACHABLE} at {view.clock(last['started_at'])}. Showing results from "
                "the last check that reached it. The next check will try again."
            )
        else:
            lines.append(f"Last check failed: {last['message']}")
    return {"lines": lines, "models": view.model_titles(models, current)}


@bp.route("/sites/<site>")
def site(site):
    inventory, store = ctx().inventory, ctx().store
    if site not in inventory.sites:
        abort(404)
    ref = now()
    statuses = view.site_statuses(store, inventory, applied(), ref)
    lines = view.sparklines(store, inventory, site, ref)
    for line in lines:
        line["last_text"] = (
            "No data" if line["last"] is None else f"{line['last']:.2f} {line['unit']}"
        )
        line["when"] = "" if line["last_time"] is None else view.ago(line["last_time"], ref)
    entry = inventory.site(site)
    return render_template(
        "site.html",
        site=site,
        entry=entry,
        status=statuses[site],
        lines=lines,
        watched_by=[m.title for m in ctx().models if site in m.sites],
    )


# History


SHOW = {
    "everything": "All",
    "unusual": "Alerts",
    "changes": "Settings changes",
}
BY = {"day": "By day", "hour": "By hour"}


def _history_entries(store, models, show, ref):
    """What History lists: unusual readings, settings changes, or both."""
    entries = []
    if show in ("everything", "unusual"):
        notes = {}
        for note in reversed(store.notes()):
            notes.setdefault(note["noticed_id"], []).append(_note_view(note))
        for item in store.noticed(flagged=True):
            v = view.item_view(store, item, ref)
            entries.append(
                {
                    "time": item["first_over"],
                    "kind": "unusual",
                    "text": _history_text(v),
                    "href": url_for("main.noticed", noticed_id=item["id"]),
                    "notes": notes.get(item["id"], []),
                }
            )
    if show in ("everything", "changes"):
        for when, sentence in _changes(store, models):
            entries.append({"time": when, "kind": "change", "text": sentence, "href": None})
    for entry in entries:
        entry["clock"] = view.clock(entry["time"])
    return sorted(entries, key=lambda e: e["time"], reverse=True)


def _history_text(v):
    return f"{v['label'].capitalize()} alert at {v['site_name']} ({v['title']})"


def _midnight(d):
    return view.midnight(d)


def _day_hours(d, entries, checks, now):
    """The 24 hours of one day: what happened in each, and the checks that ran."""
    hours = []
    for i in range(24):
        start = view.hour_start(d, i)
        end = start + view.H
        ran = [c for c in checks if start <= c["started_at"] < end]
        hours.append(
            {
                "label": view.hour(start),
                "end": end,
                "future": start > now,
                "checks": len(ran),
                "problems": sum(1 for c in ran if c.get("finished_at") and not c.get("ok")),
                "entries": sorted(
                    (e for e in entries if start <= e["time"] < end), key=lambda e: e["time"]
                ),
            }
        )
    return hours


def _fold(hours, show, now):
    """
    Busy hours get their own row, and each run of quiet hours between them
    folds into one. Hours still to come are left off.
    """
    rows, quiet = [], []

    def close():
        if not quiet:
            return
        if len(quiet) == 24:
            label = "All day"
        elif len(quiet) == 1:
            label = quiet[0]["label"]
        else:
            last = quiet[-1]
            until = (
                "now"
                if last["end"] > now
                else ("midnight" if last is hours[-1] else view.hour(last["end"]))
            )
            label = f"{quiet[0]['label']} – {until}"
        rows.append({"label": label, "quiet": True, "checks": sum(h["checks"] for h in quiet)})
        quiet.clear()

    for h in hours:
        if h["future"]:
            break
        if h["entries"] or (show == "everything" and h["problems"]):
            close()
            rows.append(h)
        else:
            quiet.append(h)
    close()
    return rows


@bp.route("/history")
def history():
    store, models = ctx().store, ctx().models
    show = request.args.get("show", "everything")
    show = "unusual" if show == "noticed" else show
    if show not in SHOW:
        show = "everything"
    by = request.args.get("by", "day")
    if by not in BY:
        by = "day"
    ref = now()
    by_day = {}
    for entry in _history_entries(store, models, show, ref):
        by_day.setdefault(view._local(entry["time"]).date(), []).append(entry)
    current = now()
    today = view._local(current).date()
    days = set(by_day)
    if show != "changes":
        days |= {today.fromordinal(today.toordinal() - i) for i in range(14)}
    checks = store.checks() if by == "hour" else []
    listed = []
    for d in sorted(days, reverse=True):
        entries = by_day.get(d, [])
        listed.append(
            {
                "label": view.day_title(d, today),
                "href": url_for("main.history_day", day=d.isoformat()),
                "entries": entries,
                "hours": _fold(_day_hours(d, entries, checks, current), show, current)
                if by == "hour"
                else [],
            }
        )
    return render_template(
        "history.html",
        days=listed,
        show=show,
        by=by,
        shows=SHOW,
        bys=BY,
        note_count=len(store.notes()),
    )


@bp.route("/history/day/<day>")
def history_day(day):
    """One day, every hour of it: the checks that ran and what they found."""
    store, models = ctx().store, ctx().models
    try:
        d = date.fromisoformat(day)
    except ValueError:
        abort(404)
    start, end = _midnight(d), _midnight(d + timedelta(days=1))
    ref = now()
    entries = [
        e for e in _history_entries(store, models, "everything", ref) if start <= e["time"] < end
    ]
    checks = [c for c in store.checks() if start <= c["started_at"] < end]
    current = now()
    today = view._local(current).date()
    return render_template(
        "history_day.html",
        title=view.day_title(d, today),
        hours=_day_hours(d, entries, checks, current),
        found=len([e for e in entries if e["kind"] == "unusual"]),
        ran=len(checks),
        previous=url_for("main.history_day", day=(d - timedelta(days=1)).isoformat()),
        following=(
            url_for("main.history_day", day=(d + timedelta(days=1)).isoformat())
            if d < today
            else None
        ),
    )


def _changes(store, models):
    """Every applied change and every change of source, as (when, sentence)."""
    out = []
    previous = None
    authors = store.version_authors()
    for version_id, when, settings in store.versions():
        # On the website each change says who made it. setup_tables is the
        # install script, not a person.
        author = authors.get(version_id)
        by = f" by {author}" if author and author != "setup_tables" else ""
        if previous is None:
            out.append((when, f"StrawberryWatch set up with default settings{by}."))
        else:
            for row in ws.describe(previous, settings, models):
                out.append((when, row["sentence"][:-1] + by + "." if by else row["sentence"]))
        previous = settings
    # Old local stores can still have choices from the removed source picker
    changes = [c for c in store.source_changes() if sources.is_live(c["kind"])]
    for change in changes[1:]:
        out.append((change["changed_at"], f"Data source changed to {_source_words(change)}."))
    return out + model_updates.history(store)


def _source_words(config):
    return _readings_words(config["kind"]).replace(
        "Strawberry Creek API", "the Strawberry Creek API"
    )


@bp.route("/history/<int:noticed_id>", methods=["GET", "POST"])
def noticed(noticed_id):
    store = ctx().store
    item = store.noticed_item(noticed_id)
    if item is None or not item["flagged"]:
        abort(404)
    form = forms.NoteForm()
    if form.validate_on_submit():
        store.add_note(noticed_id, form.note.data, author=security.owner())
        flash("Note added.")
        return redirect(_noticed_url(noticed_id))
    ref = now()
    v = view.item_view(store, item, ref)
    said = [
        {"module": name.replace("_", " ").title(), "text": entry.get("explanation") or ""}
        for name, entry in sorted(item["explanations"].items())
    ]
    from_unusual = request.args.get("from") == "unusual"
    return render_template(
        "noticed.html",
        item=v,
        chart=view.score_chart(store, item),
        said=said,
        form=form,
        notes=[_note_view(n) for n in reversed(store.notes(noticed_id=noticed_id))],
        back=(
            {"href": url_for("main.unusual"), "text": "Alerts"}
            if from_unusual
            else {"href": url_for("main.history"), "text": "History"}
        ),
    )


def _noticed_url(noticed_id):
    """Back to a reading's page, keeping which tab it was opened from."""
    if request.args.get("from") == "unusual":
        return url_for("main.noticed", noticed_id=noticed_id, **{"from": "unusual"})
    return url_for("main.noticed", noticed_id=noticed_id)


@bp.route("/history/notes/<int:note_id>/delete", methods=["POST"])
def delete_note(note_id):
    note = ctx().store.note(note_id)
    if note is None:
        abort(404)
    ctx().store.delete_note(note_id)
    flash("Note deleted.")
    return redirect(_noticed_url(note["noticed_id"]))


# Notes


def _note_view(note):
    return {**note, "when": view.moment(note["written_at"], now())}


def _note_filters():
    """The Notes page's site and dates. Values that don't parse are left out."""

    def day(name):
        try:
            return date.fromisoformat(request.args.get(name, ""))
        except ValueError:
            return None

    site = request.args.get("site")
    return {
        "site": site if site in ctx().inventory.sites else None,
        "from": day("from"),
        "to": day("to"),
    }


def _chosen_notes(filters, ids=None, ref=None):
    """The notes that match, each with the reading it's about, newest first."""
    store = ctx().store
    notes = store.notes(
        ids=ids,
        since=_midnight(filters["from"]) if filters["from"] else None,
        until=_midnight(filters["to"] + timedelta(days=1)) if filters["to"] else None,
        site=filters["site"],
    )
    items, out = {}, []
    for note in notes:
        key = note["noticed_id"]
        if key not in items:
            item = store.noticed_item(key)
            items[key] = item and {**view.item_view(store, item, ref), "raw": item}
        if items[key] is not None:
            out.append(
                {
                    **_note_view(note),
                    "item": items[key],
                    "clock": view.clock(note["written_at"]),
                    "date": view._local(note["written_at"]).date(),
                }
            )
    return out


def _by_written_day(notes, label):
    days = {}
    for note in notes:
        days.setdefault(note["date"], []).append(note)
    return [{"label": label(d), "notes": items} for d, items in days.items()]


@bp.route("/history/notes")
def notes():
    filters = _note_filters()
    ref = now()
    chosen = _chosen_notes(filters, ref=ref)
    today = view._local(now()).date()
    return render_template(
        "notes.html",
        days=_by_written_day(chosen, lambda d: view.day_title(d, today)),
        count=len(chosen),
        filters=filters,
        filtered=any(filters.values()),
        sites=sites(),
        any_notes=bool(chosen) or bool(ctx().store.notes()),
    )


@bp.route("/history/notes/export")
def notes_export():
    """The ticked notes, or all the filtered ones, oldest first, as a CSV or a printable page."""
    filters = _note_filters()
    try:
        ids = [int(v) for v in request.args.getlist("note")] or None
    except ValueError:
        abort(400)
    chosen = _chosen_notes(filters, ids=ids)[::-1]
    kind = request.args.get("format")
    today = view._local(now()).strftime("%Y-%m-%d")
    if kind == "csv":
        return Response(
            _notes_csv(chosen),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="strawberrywatch-notes-{today}.csv"'
            },
        )
    if kind != "pdf":
        abort(400)
    for note in chosen:
        note["chart"] = view.score_chart(ctx().store, note["item"]["raw"])
    return render_template(
        "notes_print.html",
        days=_by_written_day(chosen, view.long_day),
        summary=_export_summary(chosen, filters),
        back=url_for(
            "main.notes",
            **{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in filters.items() if v},
        ),
    )


def _export_summary(chosen, filters):
    count = f"{len(chosen)} note{'s' if len(chosen) != 1 else ''}"
    parts = [count]
    if filters["site"]:
        parts.append(f"at {view.site_name(filters['site'])}")
    if filters["from"] and filters["to"]:
        parts.append(f"written {view.long_day(filters['from'])} to {view.long_day(filters['to'])}")
    elif filters["from"]:
        parts.append(f"written from {view.long_day(filters['from'])}")
    elif filters["to"]:
        parts.append(f"written up to {view.long_day(filters['to'])}")
    return f"{' '.join(parts)}. Exported {view.moment(now(), None)}."


# A cell starting with one of these is run as a formula by spreadsheet apps
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _cell(value):
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(FORMULA_START) else text


def _notes_csv(chosen):
    """
    One row per note. The byte order mark tells Excel the file is UTF-8, so
    µS/cm and °C come through.
    """
    out = io.StringIO()
    writer = csv.writer(out)
    authors = any(n["author"] for n in chosen)
    writer.writerow(
        [
            "Written",
            *(["Author"] if authors else []),
            "Note",
            "Site",
            "Site name",
            "Sensor",
            "Model",
            "Alert started",
            "Alert",
            "Support modules",
        ]
    )
    inventory = ctx().inventory
    for note in chosen:
        item = note["item"]
        writer.writerow(
            [
                _cell(v)
                for v in (
                    view.stamp(note["written_at"]),
                    *([note["author"]] if authors else []),
                    note["text"],
                    item["site"],
                    inventory.site(item["site"]).name,
                    item["label"],
                    item["title"],
                    view.stamp(item["first_over"]),
                    item["happened"],
                    " ".join(item["extras"]),
                )
            ]
        )
    return "\ufeff" + out.getvalue()


# Models


@bp.route("/models")
def models():
    current = applied() or {"models": {}}
    by_name = {m.name: m for m in ctx().models}
    cards = []
    for card in catalog.cards():
        info = by_name.get(card["name"])
        state = "Not available yet"
        if card["can_run"]:
            state = "On" if current["models"].get(card["name"], {}).get("on") else "Off"
        cards.append(
            {
                **card,
                "state": state,
                "sites": len(info.sites) if info else 0,
                "note": ws.NOT_REVALIDATED.get(card["name"], ""),
            }
        )
    return render_template("models.html", cards=cards)


# Settings, Advanced settings, review and apply


@bp.route("/settings", methods=["GET", "POST"])
def settings():
    if applied() is None:
        not_set_up()
    models = ctx().models
    draft = working_copy()
    choice = ws.alerts_choice(draft)
    Form = forms.settings_form(sites(), site_labels(), models)
    initial = {
        "watch": [s for s in sites() if s not in draft["dont_flag"]],
        "alerts": None if choice == "custom" else choice,
        "rain": draft["rain"],
        **{f"model_{m.name}": "on" if draft["models"][m.name]["on"] else "off" for m in models},
    }
    form = Form(data=initial)
    if request.method == "POST":
        if form.validate():
            new = copy.deepcopy(draft)
            new["dont_flag"] = [s for s in sites() if s not in (form.watch.data or [])]
            if form.alerts.data in {key for key, *_ in ws.PRESETS} and form.alerts.data != choice:
                new = ws.apply_choice(new, form.alerts.data)
            new["rain"] = bool(form.rain.data)
            for m in models:
                value = form[f"model_{m.name}"].data
                if value in ("on", "off"):
                    new["models"][m.name]["on"] = value == "on"
            save_draft(new)
            if wants_json():
                return jsonify({"banner": banner_text(unapplied()), "changes": unapplied()})
            return _to_review("settings")
        if wants_json():
            return jsonify({"errors": form.errors}), 400
    statuses = {s: {"key": "reporting", "text": ""} for s in ctx().inventory.sites}
    ticks = set(initial["watch"]) if request.method == "GET" else set(form.watch.data or [])
    return render_template(
        "settings.html",
        form=form,
        models=models,
        custom=choice == "custom",
        map=view.creek_map(ctx().inventory, statuses, ticks=ticks),
    )


def _support_options():
    from strawberrywatch.support_modules import KINDS, SUPPORT_REGISTRY, SupportError, registry

    out = {}
    for m in ctx().models:
        options = []
        for name in sorted(
            SUPPORT_REGISTRY, key=lambda n: (KINDS.index(SUPPORT_REGISTRY[n].kind), n)
        ):
            module = SUPPORT_REGISTRY[name]()
            try:
                registry.load([name], ws.model_class(m.name))
                reason = ""
            except SupportError as exc:
                reason = str(exc)
            options.append(
                {
                    "name": name,
                    "title": name.replace("_", " ").title(),
                    "kind": module.kind,
                    "describe": catalog.SUPPORT.get(name, module.describe()),
                    "reason": reason,
                }
            )
        out[m.name] = options
    return out


def _strip(model, chosen, options):
    """Screen > Model > Explainers > Result, with what's turned on at each step."""
    from strawberrywatch.support_modules import KINDS

    kinds = [k for k in KINDS if any(o["kind"] == k for o in options)]
    words = {
        "screen": "Screen",
        "modulator": "Modulators",
        "detector": "Detectors",
        "explainer": "Explainers",
    }
    steps = []
    for kind in kinds:
        names = [o["title"] for o in options if o["kind"] == kind and o["name"] in chosen]
        steps.append({"step": words[kind], "what": ", ".join(names) or "None"})
        if kind == "screen":
            steps.append({"step": "Model", "what": model.title})
    if not any(s["step"] == "Model" for s in steps):
        steps.insert(0, {"step": "Model", "what": model.title})
    steps.append({"step": "Result", "what": "Alerts and History"})
    return steps


@bp.route("/settings/advanced", methods=["GET", "POST"])
def advanced():
    if applied() is None:
        not_set_up()
    store, models = ctx().store, ctx().models
    draft = working_copy()
    options = _support_options()
    Form = forms.advanced_form(models, options, ctx().profile.alert_gaps)
    initial = {"repeat_hours": f"{draft['repeat_hours']:g}"}
    for m in models:
        spec = draft["models"][m.name]
        initial[f"model_{m.name}"] = "on" if spec["on"] else "off"
        initial[f"rain_{m.name}"] = f"{spec['rain_multiplier']:g}"
        initial[f"gap_{m.name}"] = f"{spec.get('alert_gap_minutes', ws.ALERT_GAP_DEFAULT):g}"
        initial[f"support_{m.name}"] = list(spec["support"])
        if "leave_out" in m.site_choice:
            initial[f"leave_{m.name}"] = list(spec["leave_out"])
        for rule in m.rules if m.sensitivity == "scale" else ():
            initial[f"scale_{m.name}_{rule}"] = f"{spec['scales'][rule]:.2f}"
        if m.sensitivity == "rate":
            for rule in m.rules:
                initial[f"rate_{m.name}_{rule}"] = str(_stop_index(spec["rates"][rule]))
    form = Form(data=initial)
    if request.method == "POST":
        if form.validate():
            new = copy.deepcopy(draft)
            new["repeat_hours"] = float(form.repeat_hours.data)
            for m in models:
                spec = new["models"][m.name]
                value = form[f"model_{m.name}"].data
                if value in ("on", "off"):
                    spec["on"] = value == "on"
                for rule in m.rules if m.sensitivity == "scale" else ():
                    spec["scales"][rule] = round(float(form[f"scale_{m.name}_{rule}"].data), 4)
                if m.sensitivity == "rate":
                    _read_rates(spec, m, form)
                spec["rain_multiplier"] = float(form[f"rain_{m.name}"].data)
                if ctx().profile.alert_gaps:
                    spec["alert_gap_minutes"] = float(form[f"gap_{m.name}"].data)
                spec["support"] = sorted(form[f"support_{m.name}"].data or [])
                if "leave_out" in m.site_choice:
                    spec["leave_out"] = [
                        s for s in m.sites if s in (form[f"leave_{m.name}"].data or [])
                    ]
            save_draft(new)
            if wants_json():
                current = store.draft(draft_owner()) or applied()
                return jsonify(
                    {
                        "banner": banner_text(unapplied()),
                        "changes": unapplied(),
                        "previews": _previews(current),
                        "rate_previews": _rate_previews(current),
                        "curves": {
                            m.name: watch_models.rain_curve(
                                m, current["rain"], current["models"][m.name]["rain_multiplier"]
                            )
                            for m in models
                        },
                    }
                )
            return _to_review("advanced")
        if wants_json():
            return jsonify({"errors": form.errors}), 400
    blocks = []
    for m in models:
        spec = draft["models"][m.name]
        blocks.append(
            {
                "model": m,
                "on": spec["on"],
                "typed": _typed_rows(m, spec) if m.sensitivity == "rate" else [],
                "strip": _strip(m, spec["support"], options[m.name]),
                "options": options[m.name],
                "curve": watch_models.rain_curve(m, draft["rain"], spec["rain_multiplier"]),
                "warning": ws.NOT_REVALIDATED.get(m.name, ""),
            }
        )
    last = store.last_check()
    took = view.check_took(last)
    config = store.source() or {}
    return render_template(
        "advanced.html",
        form=form,
        blocks=blocks,
        previews=_previews(draft),
        took=None if took is None else round(took),
        source=_source_words(config) if config else "not set up yet",
        interval=ws.INTERVAL_MINUTES,
        lookback=ws.LOOKBACK_DAYS,
        costs=_costs(models),
        rate_previews=_rate_previews(draft),
        stops=ws.stops(),
        calibrated_with=_calibrated_with(),
        limits=ws.website_limits(),
    )


def _load_limits():
    """
    The limits a change has to fit inside. Only the website checks the memory
    floor, since it was measured for that server and would block most things
    on a laptop.
    """
    if ctx().profile.checks_here:
        return {"limits": ws.website_limits()}
    week = ctx().store.checks(since=now() - 7 * 86400)
    return {
        "limits": ws.website_limits(),
        "memory": {"free_mb": ws.free_memory_mb(), "costs": ws.recent_costs(week)},
    }


def _costs(models):
    """What the last check of each model took, for Advanced settings."""
    last = ctx().store.last_finished_check(kinds=sources.LIVE)
    durations = (last or {}).get("durations") or {}
    rows = []
    for m in models:
        if m.name in durations:
            mb = (durations.get("memory_mb") or {}).get(m.name)
            rows.append(
                f"{m.title}: {durations[m.name]:.0f} s"
                + (f", about {mb:,.0f} MB" if mb is not None else "")
            )
    if "peak_mb" in durations:
        rows.append(f"Whole check: {durations['peak_mb']:,.0f} MB at most")
    return rows


@bp.route("/checks/resume", methods=["POST"])
def resume_checks():
    """Try the model checks again after they paused."""
    ctx().store.set_pref("check_failures", "0")
    ctx().store.set_pref("checks_paused", "no")
    ctx().watcher.check_now()
    flash(
        "Checks restarted. A check is running now."
        if ctx().profile.checks_here
        else "Checks will restart within 15 minutes."
    )
    return redirect(url_for("main.overview"))


def _stop_index(rate):
    values = [value for _, _, value in ws.stops()]
    return min(range(len(values)), key=lambda i: abs(values[i] - rate))


def _read_rates(spec, model, form):
    """Reads a calibrated model's sliders, typed threshold and reset button into its spec."""
    if request.form.get("calibrated") == model.name:
        spec["rates"] = {rule: ws.PRESET_STOPS["balanced"] for rule in model.rules}
        spec["thresholds"] = {}
        return
    values = [value for _, _, value in ws.stops()]
    for rule in model.rules:
        spec["rates"][rule] = values[int(form[f"rate_{model.name}_{rule}"].data)]
    typed = dict(spec.get("thresholds") or {})
    for key in form[f"untype_{model.name}"].data or []:
        typed.pop(key, None)
    key, value = form[f"typed_key_{model.name}"].data, form[f"typed_value_{model.name}"].data
    if key and (value or "").strip():
        typed[key] = float(value)
    spec["thresholds"] = typed


def _typed_rows(model, spec):
    """Each threshold typed by hand, with the false alarm rate it means and whether that's trusted."""
    from strawberrywatch.models import Water_Strider as wsm

    rows = []
    for key, value in sorted((spec.get("thresholds") or {}).items()):
        site_var, rule = key.rsplit(".", 1)
        tail = model.tails.get(site_var, {}).get(rule)
        if tail is None:
            continue
        implied = wsm.rate_for(tail, value)
        lo, hi = wsm.RATE_WINDOW
        rows.append(
            {
                "key": key,
                "words": ws.sensor_words(key).capitalize(),
                "value": value,
                "calibrated": tail["threshold"],
                "rate": implied,
                "outside": value < tail["t"] or not lo <= implied <= hi,
            }
        )
    return rows


def _rate_previews(draft):
    """The sentence under each slider: what the last 7 days would have flagged, at this stop and at calibrated."""
    out = {}
    for m in ctx().models:
        if m.sensitivity != "rate" or not draft["models"].get(m.name, {}).get("on"):
            continue
        counts = view.rate_previews(ctx().store, m, draft)
        out[m.name] = {
            rule: (
                f"Nothing to count yet: no checks with {m.title} on in the last 7 days."
                if counts is None
                else f"Last 7 days at this setting: {_alerts(counts[rule][0])}. "
                f"At the calibrated setting: {counts[rule][1]}."
            )
            for rule in m.rules
        }
    return out


def _alerts(n):
    return f"{n} alert{'s' if n != 1 else ''}"


def _calibrated_with():
    """What Water Strider's calibration was fitted with, said once on Advanced settings."""
    import json

    from strawberrywatch import paths

    path = paths.model_dir("water_strider") / "water_strider_calibration.json"
    if not path.exists():
        return None
    calibration = json.loads(path.read_text())
    params = calibration.get("params", {})
    return {
        "window": params.get("hampel_window"),
        "floor": params.get("mad_floor_share"),
        "profile_days": params.get("profile_days"),
        "made": calibration.get("made"),
        "sensors": len(calibration.get("sensors", {})),
    }


def _to_review(page):
    """Save without JavaScript: on to the review page, or back if there are no changes."""
    if ctx().store.draft(draft_owner()) is None:
        flash("No changes to save.")
        return redirect(url_for(f"main.{page}"))
    return redirect(url_for("main.review", **{"from": page}))


def _previews(draft):
    counts = view.preview_counts(ctx().store, ctx().models, draft)
    out = {}
    for m in ctx().models:
        count = counts.get(m.name)
        if not draft["models"][m.name]["on"]:
            out[m.name] = ""
        elif count is None:
            out[m.name] = f"Nothing to count yet: no checks with {m.title} on in the last 7 days."
        else:
            note = view.rain_note(draft, applied(), m.name)
            out[m.name] = (
                f"Last 7 days with these values: {_alerts(count['would'])}. "
                f"With the saved settings: {count['did']}." + (f" {note}" if note else "")
            )
    return out


@bp.route("/settings/review")
def review():
    store, models = ctx().store, ctx().models
    draft = store.draft(draft_owner())
    page = "advanced" if request.args.get("from") == "advanced" else "settings"
    if draft is None:
        flash("No changes to save.")
        return redirect(url_for(f"main.{page}"))
    last = store.last_check()
    _, problems, warnings = ws.check(
        draft, models, sites(), applied(), view.check_took(last), **_load_limits()
    )
    rows = ws.describe(applied(), draft, models)
    return render_template(
        "review.html",
        rows=rows,
        problems=problems,
        warnings=warnings,
        back=url_for(f"main.{page}"),
        page=page,
    )


@bp.route("/settings/apply", methods=["POST"])
def apply():
    store, models = ctx().store, ctx().models
    draft = store.draft(draft_owner())
    if draft is None:
        return redirect(url_for("main.settings"))
    clean, problems, _ = ws.check(
        draft, models, sites(), applied(), view.check_took(store.last_check()), **_load_limits()
    )
    if problems:
        return redirect(url_for("main.review"))
    owner = draft_owner()
    base = store.draft_base(owner)
    try:
        store.apply(clean, by=security.owner(), base_id=base, owner=owner)
    except st.Conflict as exc:
        # Someone applied first. Theirs stands, and ours goes back to being a draft on top
        merged = ws.merge(store.version(base) or applied(), draft, applied())
        store.discard_draft(owner)
        store.save_draft(merged, owner)
        who = exc.applied_by or "Someone else"
        flash(
            f"{who} saved changes at {view.clock(exc.applied_at)} while you were editing. Your "
            "changes are kept below, on top of theirs. Check them before saving.",
            "error",
        )
        return redirect(url_for("main.review"))
    ctx().watcher.check_now()
    flash(
        "Settings saved. A check is running now."
        if ctx().profile.checks_here
        else "Settings saved. They take effect at the next check, within 15 minutes."
    )
    return redirect(url_for("main.overview"))


# Email alerts. Changes here apply straight away, not through the Settings draft.


def _mailing_list():
    return mail.mailing_list(ctx().store, ctx().profile.email_in_env)


def _address():
    """Where the links in the website's emails point. The test dashboard sets its own."""
    return current_app.config.get("PUBLIC_ADDRESS") or ws.website_address()


def _website_only():
    if ctx().profile.email_in_env:
        abort(404)


@bp.route("/settings/email", methods=["GET", "POST"])
def email():
    store = ctx().store
    in_env = ctx().profile.email_in_env
    book = _mailing_list()
    labels = [] if in_env else store.labels()
    form = forms.RecipientForm() if in_env else forms.labelled_recipient_form(labels)()
    if form.validate_on_submit():
        if in_env:
            added = book.add(form.email.data, form.name.data)
        else:
            added = book.add(
                form.email.data, form.name.data, int(form.label.data), security.owner()
            )
        if added:
            flash(f"Added {form.email.data}.")
            if form.note:
                flash(form.note)
            return redirect(url_for("main.email"))
        form.email.errors.append(f"{form.email.data} is already a recipient")
    last = store.emails(limit=1)
    results = store.email_results()
    names = {lb["id"]: lb["name"] for lb in labels}
    when = now()
    people = [
        {
            **p,
            "result": _result_line(results.get(p["email"].lower())),
            "label": names.get(p.get("label_id"), ""),
            "alerts": "" if in_env else _alerts_line(p, ctx().models, when),
        }
        for p in book.people()
    ]
    total = len(people)
    shown = request.args.get("label", "") if labels else ""
    if shown:
        people = [p for p in people if p["label"] == shown]
    order = request.args.get("sort", "")
    if order == "label":
        # No label sorts last, like an empty cell
        people.sort(key=lambda p: (p["label"] == "", p["label"].lower(), p["email"].lower()))
    copied = store.pref("list_copied") if not in_env else None
    return render_template(
        "email.html",
        form=form,
        people=people,
        total=total,
        on=mail.is_on(store),
        gaps=mail.missing(),
        sender=mail.sender(),
        in_env=in_env,
        last=_email_line(last[0]) if last else "",
        labels=labels,
        shown=shown,
        order=order,
        can_copy=ctx().copy_list is not None and not copied and not in_env,
        copied=_copied_line(copied),
    )


def _copied_line(copied):
    if not copied:
        return ""
    done = json.loads(copied)
    people = f"{done['count']} {'address' if done['count'] == 1 else 'addresses'}"
    return f"Copied {people} from Night Heron's alert rules on {view.moment(done['at'], None)}."


def _back_to_list():
    """Back to the list as it was filtered and sorted."""
    args = {k: request.form.get(k) for k in ("label", "sort") if request.form.get(k)}
    return redirect(url_for("main.email", **args))


def _result_line(result):
    if not result:
        return ""
    when = view.moment(result["sent_at"], now())
    what = "Test email" if result["kind"] == "test" else "Alert"
    return f"{what} sent {when}." if result["ok"] else f"{what}, {when}: {result['message']}"


def _email_line(sent):
    when = view.moment(sent["sent_at"], now())
    what = "test email" if sent["kind"] == "test" else "alert"
    if not sent["ok"]:
        return f"The last {what} ({when}) didn't reach everyone. {sent['message']}"
    people = f"{sent['people']} {'person' if sent['people'] == 1 else 'people'}"
    return f"The last {what} went to {people} at {when}."


@bp.route("/settings/email/sending", methods=["POST"])
def email_sending():
    store = ctx().store
    where = "on this computer" if ctx().profile.email_in_env else "on the server"
    if request.form.get("on") == "yes":
        if mail.missing():
            flash(f"Email isn't set up {where}, so sending can't be turned on.", "error")
        else:
            store.set_pref("email_on", "yes")
            flash("Email alerts turned on.")
    else:
        store.set_pref("email_on", "no")
        flash("Email alerts turned off.")
    return redirect(url_for("main.email"))


@bp.route("/settings/email/remove", methods=["POST"])
def remove_recipient():
    address = (request.form.get("email") or "").strip()
    if not address or not _mailing_list().remove(address):
        abort(404)
    flash(f"Removed {address}.")
    return _back_to_list()


@bp.route("/settings/email/test", methods=["POST"])
def email_test():
    """The test email, to the people ticked in the list. No one ticked means everyone shown."""
    people = _mailing_list().people()
    shown = request.form.get("label", "")
    if shown and not ctx().profile.email_in_env:
        names = {lb["id"]: lb["name"] for lb in ctx().store.labels()}
        people = [p for p in people if names.get(p.get("label_id")) == shown]
    ticked = request.form.getlist("to")
    # Only addresses already on the list, whatever the form sent
    to = mail.chosen(people, ticked) if ticked else people
    if not to:
        flash(
            "The people ticked aren't on the list any more."
            if ticked
            else "There's no one on the list to send a test email to.",
            "error",
        )
        return _back_to_list()
    only = {p["email"] for p in to}
    try:
        if ctx().profile.email_in_env:
            sent, count = mail.test(ctx().store, only)
        else:
            sent, count = mail.website_test(ctx().store, _address(), only)
    except mail.MailError as exc:
        flash(f"Test email not sent. {exc}", "error")
    else:
        if sent == count:
            flash(
                f"Test email sent to {count} {'person' if count == 1 else 'people'}. "
                "If it doesn't arrive, ask them to check their spam folder."
            )
        else:
            flash(
                f"Test email sent to {sent} of {count} people. Details are next to each address "
                "below.",
                "error",
            )
    return _back_to_list()


# One person's alerts, website only: which models, and pausing them. Each
# person picks models the way Healthchecks lets each check pick its
# integrations (templates/front/details.html, BSD-3-Clause). "Every model" is
# its own choice, like Uptime Kuma's "Default enabled" notifications
# (src/components/NotificationDialog.vue, MIT), and models added later reach
# those people too.


def _paused_until(person, when):
    if person.get("paused_until") is None:
        return "until turned back on"
    return f"until {view.moment(person['paused_until'], when)}"


def _alerts_line(person, models, when):
    """Paused, All models or 2 of 5 models, for the list."""
    if mail.paused(person, when):
        if person.get("paused_until") is None:
            return "Paused"
        return f"Paused {_paused_until(person, when)}"
    wanted = mail.models_for(person)
    names = [m.name for m in models]
    if wanted is None or set(names) <= set(wanted):
        return "All models"
    return f"{sum(1 for n in names if n in wanted)} of {len(names)} models"


@bp.route("/settings/email/people/<int:recipient_id>", methods=["GET", "POST"])
def person(recipient_id):
    _website_only()
    store = ctx().store
    person = store.recipient(recipient_id)
    if person is None:
        abort(404)
    when = now()
    models = ctx().models
    names = [m.name for m in models]
    wanted = mail.models_for(person)
    pausing = mail.paused(person, when)
    Form = forms.person_form(
        models,
        view._local(when).date(),
        f"Keep paused {_paused_until(person, when)}" if pausing else "",
    )
    form = Form(
        data={
            "name": person.get("name") or "",
            "models": names if wanted is None else [n for n in names if n in wanted],
            "pause": "keep" if pausing else "off",
        }
    )
    who = person.get("name") or person["email"]
    if form.validate_on_submit():
        store.set_name(recipient_id, form.name.data)
        who = form.name.data or person["email"]
        ticked = form.models.data
        store.set_models(
            recipient_id, None if set(names) <= set(ticked) else [n for n in names if n in ticked]
        )
        choice = form.pause.data
        if choice == "off":
            store.pause(recipient_id)
        elif choice == "always":
            store.pause(recipient_id, when)
        elif choice == "date":
            store.pause(recipient_id, when, view.midnight(form.pause_date.day))
        elif choice != "keep":
            store.pause(recipient_id, when, when + int(choice) * 86400)
        flash(f"Saved alert settings for {who}.")
        return redirect(url_for("main.email"))
    current = applied() or {"models": {}}
    off = {
        m.name: {"hint": "Off in Settings"}
        for m in models
        if not current["models"].get(m.name, {}).get("on")
    }
    label = next((lb["name"] for lb in store.labels() if lb["id"] == person.get("label_id")), "")
    return render_template(
        "person.html",
        form=form,
        person=person,
        who=who,
        label=label,
        off=off,
        pause_line=f"Paused {_paused_until(person, when)}" if pausing else "On",
        chosen=[m.title for m in models if wanted is None or m.name in wanted],
    )


# Labels, website only. They tell people apart and don't grant access.


@bp.route("/settings/email/relabel", methods=["POST"])
def relabel():
    _website_only()
    store = ctx().store
    try:
        recipient_id = int(request.form.get("recipient", ""))
        label_id = int(request.form.get("label_id", ""))
    except ValueError:
        abort(400)
    person = store.recipient(recipient_id)
    names = {lb["id"]: lb["name"] for lb in store.labels()}
    if person is None or label_id not in names:
        abort(404)
    store.set_label(recipient_id, label_id)
    flash(f"{person['email']} is now labelled {names[label_id]}.")
    return _back_to_list()


@bp.route("/settings/email/labels", methods=["POST"])
def save_label():
    """Adds a label, or renames one when the form names which."""
    _website_only()
    store = ctx().store
    form = forms.LabelForm()
    if not form.validate_on_submit():
        for error in form.name.errors:
            flash(error, "error")
        return redirect(url_for("main.email"))
    name = form.name.data
    if request.form.get("label_id"):
        try:
            label_id = int(request.form["label_id"])
        except ValueError:
            abort(400)
        old = next((lb["name"] for lb in store.labels() if lb["id"] == label_id), None)
        if old is None:
            abort(404)
        if not store.rename_label(label_id, name):
            flash(f"There's already a label called {name}", "error")
        else:
            flash(f"Renamed {old} to {name}.")
    elif store.add_label(name):
        flash(f"Added the label {name}.")
    else:
        flash(f"There's already a label called {name}", "error")
    return redirect(url_for("main.email"))


@bp.route("/settings/email/copy", methods=["POST"])
def copy_list():
    """Copies the addresses on Night Heron's alert rules onto this list, once."""
    _website_only()
    store = ctx().store
    if ctx().copy_list is None or store.pref("list_copied"):
        abort(404)
    names = {lb["id"]: lb["name"] for lb in store.labels()}
    try:
        label_id = int(request.form.get("label_id", ""))
    except ValueError:
        abort(400)
    if label_id not in names:
        abort(400)
    try:
        found = ctx().copy_list()
    except Exception:
        logger.exception("reading Night Heron's email lists failed")
        flash("Couldn't read Night Heron's alert rules. Try again later.", "error")
        return redirect(url_for("main.email"))
    added = 0
    for address in found:
        if mail.valid(address) and store.add_recipient(
            address, None, label_id, "Copied from Night Heron"
        ):
            added += 1
    store.set_pref("list_copied", json.dumps({"at": time.time(), "count": added}))
    message = f"Copied {added} {'address' if added == 1 else 'addresses'} from Night Heron"
    message += f", labelled {names[label_id]}."
    if len(found) > added:
        message += f" {len(found) - added} were already on the list or not valid."
    flash(message)
    return redirect(url_for("main.email"))


# Unsubscribe, the one public page, website only: every alert email links here,
# and it never shows anyone else's address. A week's pause is offered there
# too, for someone who's away and still wants the alerts after.
SELF_PAUSE_DAYS = 7


@bp.route("/unsubscribe/<token>", methods=["GET", "POST"])
def unsubscribe(token):
    _website_only()
    store = ctx().store
    person = store.recipient_by_token(token)
    if person is None:
        return render_template("unsubscribe.html", state="gone"), 404
    when = now()
    if request.method == "POST" and request.form.get("action") == "pause":
        # A pause someone already has, maybe set longer by an admin, stays as it is
        if not mail.paused(person, when):
            store.pause(person["id"], when, when + SELF_PAUSE_DAYS * 86400)
        person = store.recipient(person["id"])
        return render_template(
            "unsubscribe.html", state="paused", until=_paused_until(person, when)
        )
    if request.method == "POST":
        store.remove_recipient(person["id"])
        return render_template("unsubscribe.html", state="done")
    return render_template(
        "unsubscribe.html",
        state="ask",
        address=person["email"],
        until=_paused_until(person, when) if mail.paused(person, when) else "",
    )


# Model updates, website only


def _updater():
    """404 unless this profile can update models."""
    if ctx().updater is None:
        abort(404)
    return ctx().updater


@bp.route("/settings/models")
def model_updates_page():
    _updater()
    store = ctx().store
    state = model_updates.state(store)
    running = model_updates.is_running(store, ctx().clock)
    ref = now()
    rows = []
    for m in ctx().models:
        update = model_updates.current(store, m.name)
        rows.append(
            {
                "name": m.name,
                "title": m.title,
                "files": (
                    f"From GitHub, {view.moment(update['at'], ref)} ({update['commit'][:7]})"
                    + (f", by {update['by']}" if update.get("by") else "")
                )
                if update
                else "Installed with StrawberryWatch",
                "updated": update is not None,
            }
        )
    return render_template(
        "model_updates.html",
        state=state,
        running=running,
        started=view.moment(state["started_at"], ref) if state.get("started_at") else "",
        finished=view.moment(state["finished_at"], ref) if state.get("finished_at") else "",
        rows=rows,
        repo=model_updates.REPO,
        branch=model_updates.BRANCH,
    )


@bp.route("/settings/models/update", methods=["POST"])
def update_models():
    if not _updater().start(ctx().store, security.owner()):
        flash("An update is already running.")
    return redirect(url_for("main.model_updates_page"))


@bp.route("/settings/models/undo", methods=["POST"])
def undo_model_update():
    _updater()
    by_name = {m.name: m for m in ctx().models}
    name = request.form.get("model", "")
    if name not in by_name or not model_updates.undo(ctx().store, name, security.owner()):
        abort(404)
    flash(f"{by_name[name].title} is back on its previous files, from the next check.")
    return redirect(url_for("main.model_updates_page"))


@bp.route("/settings/discard", methods=["POST"])
def discard():
    ctx().store.discard_draft(draft_owner())
    flash("Changes discarded.")
    back = request.form.get("back")
    return redirect(url_for("main.advanced") if back == "advanced" else url_for("main.settings"))
