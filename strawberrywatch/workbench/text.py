"""Words the Model Tuner shows outside its templates, and the helpers that format them."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

LOCAL = ZoneInfo("America/Los_Angeles")

# Step statuses
IN_PROGRESS = "In progress"
DONE = "Done"
NOT_STARTED = "Not started"
LOCKED = "Cannot start yet"

# Readings in the order every list and table uses
READINGS = ("conductivity", "depth", "temperature", "dissolved_oxygen", "ph")
READING_NAMES = {
    "conductivity": "Conductivity",
    "depth": "Depth",
    "temperature": "Temperature",
    "dissolved_oxygen": "Dissolved oxygen",
    "ph": "pH",
}
# Chart titles
READING_TITLES = {
    "conductivity": "Conductivity (µS/cm)",
    "depth": "Depth (mm)",
    "temperature": "Temperature (°C)",
    "dissolved_oxygen": "Dissolved oxygen (mg/L)",
    "ph": "pH",
}
# Inside a sentence, after a site name
READING_LABELS = {
    "conductivity": "conductivity (µS/cm)",
    "depth": "depth (mm)",
    "temperature": "temperature (°C)",
    "dissolved_oxygen": "dissolved oxygen (mg/L)",
    "ph": "pH",
}
UNITS = {
    "conductivity": "µS/cm",
    "depth": "mm",
    "temperature": "°C",
    "dissolved_oxygen": "mg/L",
    "ph": "",
}
BINS = {"conductivity": 5, "depth": 5, "temperature": 0.5, "dissolved_oxygen": 0.5, "ph": 0.1}
DEFAULT_READINGS = ("conductivity", "depth", "temperature")

# Sensor status words
LIVE = "Live"
QUIET = "Quiet"
NONE = "None"

KINDS = {
    "spill": "Spill or leak",
    "rain": "Rain",
    "sensor": "Sensor problem",
    "visit": "Field visit",
    "other": "Other",
}
# Kinds whose events a model should alert on, for a new event's default
ALERT_KINDS = ("spill", "sensor")
YES = "Yes"
NO = "No"
SHIPPED = "Shipped"
ADDED_HERE = "Added here"

FALSE_ALARMS = {
    "month": "About 1 a month",
    "week": "About 1 a week",
    "day": "About 1 a day",
}
DEFAULT_FALSE_ALARMS = "week"

LEFT_OUT_BOTH = "Known events and rainy hours"
LEFT_OUT_EVENTS = "Known events"
LEFT_OUT_RAIN = "Rainy hours"
LEFT_OUT_NONE = "Nothing"

NOT_RUN = "Not run yet"
CAUGHT = "Caught"
MISSED = "Missed"
STAYED_QUIET = "Stayed quiet"
ALERTED = "Alerted"
NOT_ENOUGH = "Not enough data"

MODEL_TEXT = {
    "dusk_crayfish": "Learns how the creek behaves as a whole and notices when readings stop matching.",
    "cobble_shoal": "Scores each sensor on its own, so it can point to the sensor that changed.",
}

# Source names inside sentences
SOURCE_NAMES = {
    "api": "the Strawberry Creek website",
    "csv": "a folder of CSV files",
    "mysql": "a MySQL database",
}

MANUAL_PAGES = (
    "Getting started",
    "The creek and its sensors",
    "Picking data",
    "Reading the Explore charts",
    "What recalibrating does",
    "Past calibrations",
    "Events",
    "Drift check",
    "Exporting",
    "When something looks wrong",
    "Glossary",
)

PLURALS = {
    "site": "sites",
    "day": "days",
    "reading": "readings",
    "stretch": "stretches",
}
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
SHORT_MONTHS = tuple(m[:3] for m in MONTHS)

ONE_OTHER_SITE = "{first} and 1 other site"
OTHER_SITES = "{first} and {n} other sites"
MINUTES_AGO = "{n} min ago"
ONE_HOUR_AGO = "1 hour ago"
HOURS_AGO = "{n} hours ago"
ONE_DAY_AGO = "1 day ago"
DAYS_AGO = "{n} days ago"


def slug(title):
    return title.lower().replace(" ", "-")


def number(n):
    return f"{n:,}"


def count(n, word):
    return f"{n:,} {word if n == 1 else PLURALS[word]}"


def join_list(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def site_group(names):
    names = list(names)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return ONE_OTHER_SITE.format(first=names[0])
    return OTHER_SITES.format(first=names[0], n=len(names) - 1)


def _day(d, months):
    return f"{d.day} {months[d.month - 1]} {d.year}"


def _dates(start, end, months):
    if start == end:
        return _day(start, months)
    if start.year == end.year and start.month == end.month:
        return f"{start.day} to {end.day} {months[end.month - 1]} {end.year}"
    if start.year == end.year:
        return f"{start.day} {months[start.month - 1]} to {_day(end, months)}"
    return f"{_day(start, months)} to {_day(end, months)}"


def long_dates(start, end=None):
    return _dates(start, end or start, MONTHS)


def short_dates(start, end=None):
    return _dates(start, end or start, SHORT_MONTHS)


def time_of_day(moment):
    hour = moment.hour % 12 or 12
    suffix = "am" if moment.hour < 12 else "pm"
    if moment.minute == 0:
        return f"{hour} {suffix}"
    return f"{hour}:{moment.minute:02d} {suffix}"


def ago(seconds):
    minutes = int(seconds // 60)
    if minutes < 60:
        return MINUTES_AGO.format(n=minutes)
    hours = int(seconds // 3600)
    if hours < 24:
        return ONE_HOUR_AGO if hours == 1 else HOURS_AGO.format(n=hours)
    days = int(seconds // 86400)
    return ONE_DAY_AGO if days == 1 else DAYS_AGO.format(n=days)


def bin_label(reading):
    width = BINS[reading]
    unit = UNITS[reading]
    shown = f"{width:g}"
    return f"{shown} {unit}" if unit else shown


def drift_phrase(pairs, site_name):
    """pairs are (site, reading). Grouped by reading as "conductivity at North Fork 0 and Oxford"."""
    groups = []
    for reading in READINGS:
        sites = [site for site, r in pairs if r == reading]
        if sites:
            label = READING_NAMES[reading]
            label = label if reading == "ph" else label.lower()
            groups.append(f"{label} at {join_list(site_name(s) for s in sites)}")
    return join_list(groups)


def local(moment):
    """A UTC datetime or pandas Timestamp in America/Los_Angeles."""
    return moment.astimezone(LOCAL)


def left_out(events, rain):
    if events and rain:
        return LEFT_OUT_BOTH
    if events:
        return LEFT_OUT_EVENTS
    if rain:
        return LEFT_OUT_RAIN
    return LEFT_OUT_NONE


def last_days(days, today=None):
    """(start, end) dates for "Last 7 days" and "Last 30 days", ending today."""
    today = today or datetime.now(LOCAL).date()
    return today - timedelta(days=days - 1), today


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None
