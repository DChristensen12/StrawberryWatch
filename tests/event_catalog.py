"""The labelled event catalog in tests/events.yaml, for the anomaly tests and audit reports."""
# Checks the catalog's shape and hands back rows. Adding an event is one entry in one file

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from strawberrywatch import paths

CATALOG_PATH = Path(__file__).with_name("events.yaml")

# What a label commits the detector to. relative_only records a known event no test grades yet
LABELS = ("anomaly", "true_negative", "relative_only")

REQUIRED = ("folder", "group", "site", "label")
# field_log lists field log lines as evidence, each by its "date site sensor event" key
# known_failure says why a graded event doesn't pass yet, and its test is expected to fail
OPTIONAL = ("note", "field_log", "known_failure")


class CatalogError(ValueError):
    """The message names the entry and the fix."""


@dataclass(frozen=True)
class Event:
    """One (event, site) pair and what the detector must say about it."""

    folder: str
    group: str
    site: str
    label: str
    note: str = ""
    field_log: tuple = ()
    known_failure: str = ""

    @property
    def path(self) -> Path:
        """The fixture CSV folder, which may not exist."""
        return paths.anomalies_dir() / self.folder

    def as_tuple(self):
        """The (folder, site, label, group) shape the event tests parametrize on."""
        return (self.folder, self.site, self.label, self.group)


def _fail(index, entry, message):
    name = entry.get("group") or entry.get("folder") or "?" if isinstance(entry, dict) else "?"
    raise CatalogError(f"tests/events.yaml entry {index} ({name}): {message}")


def parse(blob, known_sites=None, known_lines=None):
    """Validates a parsed mapping and returns its events in file order."""
    # With known_sites, a misspelled site fails here, not as a skipped test
    if not isinstance(blob, dict) or "events" not in blob:
        raise CatalogError("tests/events.yaml has no top-level 'events:' key")
    rows = blob["events"]
    if not isinstance(rows, list) or not rows:
        raise CatalogError("tests/events.yaml 'events:' must be a non-empty list")

    events, seen = [], {}
    for i, entry in enumerate(rows):
        if not isinstance(entry, dict):
            _fail(i, entry, f"is a {type(entry).__name__}, not a mapping")
        missing = [k for k in REQUIRED if not entry.get(k)]
        if missing:
            _fail(i, entry, f"missing {', '.join(missing)}")
        extra = set(entry) - set(REQUIRED) - set(OPTIONAL)
        if extra:
            _fail(i, entry, f"unknown key(s) {', '.join(sorted(extra))}")
        if entry["label"] not in LABELS:
            _fail(i, entry, f"label {entry['label']!r} is not one of {', '.join(LABELS)}")
        if known_sites is not None and entry["site"] not in known_sites:
            _fail(
                i,
                entry,
                f"site {entry['site']!r} is not a site this repo knows; "
                f"expected one of {', '.join(sorted(known_sites))}",
            )
        cited = entry.get("field_log") or []
        if not isinstance(cited, list):
            _fail(i, entry, "field_log must be a list of field log lines")
        if known_lines is not None:
            for line in cited:
                if line not in known_lines:
                    _fail(
                        i,
                        entry,
                        f"field_log names {line!r}, which isn't a line of field_log.csv; "
                        f"write it as 'date site sensor event', for example "
                        f"'2026-03-19 south_fork_1 ctd stolen'",
                    )
        key = (entry["folder"], entry["site"])
        if key in seen:
            _fail(i, entry, f"duplicates entry {seen[key]}: {key[0]} is already scored at {key[1]}")
        seen[key] = i
        events.append(
            Event(
                folder=entry["folder"],
                group=entry["group"],
                site=entry["site"],
                label=entry["label"],
                note=(entry.get("note") or "").strip(),
                field_log=tuple(cited),
                known_failure=(entry.get("known_failure") or "").strip(),
            )
        )
    return events


def known_sites():
    """Site names in both vocabularies. The model says footbridge, the inventory scnf010."""
    from strawberrywatch.models.Cobble_Shoal import SITE_ORDER

    names = set(SITE_ORDER)
    try:
        from strawberrywatch import inventory as inventory_module

        names |= set(inventory_module.load().sites)
    except Exception:
        # An unloadable inventory is its own test's problem, not this one's
        pass
    return names


def load(path=CATALOG_PATH, check_sites=True):
    """Every event in the catalog, in file order."""
    path = Path(path)
    if not path.exists():
        raise CatalogError(f"no event catalog at {path}")
    blob = yaml.safe_load(path.read_text()) or {}
    return parse(
        blob,
        known_sites=known_sites() if check_sites else None,
        known_lines=known_lines() if check_sites else None,
    )


def known_lines():
    """The keys of the field log's lines, for checking an entry's evidence."""
    from strawberrywatch import field_log

    return set(field_log.load()["key"])


def true_negative_candidates():
    """The field log's visit_ok lines as (site, sensor, UTC time, note), for a person to label."""
    from strawberrywatch import field_log

    visits = field_log.visits(field_log.load())
    return [(row.site, row.sensor, row.start, row.note) for row in visits.itertuples()]


def by_label(label, events=None):
    """The events carrying one label, in file order."""
    return [e for e in (events if events is not None else load()) if e.label == label]


def by_folder(events=None):
    """One row per event folder, (folder, group, sites, label), in file order."""
    # For reports that score a whole event window. A folder's first entry decides its label,
    # since the folder is named for that row
    order, agg = [], {}
    for e in events if events is not None else load():
        if e.folder not in agg:
            order.append(e.folder)
            agg[e.folder] = (e.group, [], e.label)
        agg[e.folder][1].append(e.site)
    return [(f, agg[f][0], list(agg[f][1]), agg[f][2]) for f in order]


def groups(events=None):
    """Distinct event folders, in file order. One folder scored at two sites is one group."""
    seen, out = set(), []
    for e in events if events is not None else load():
        if e.folder not in seen:
            seen.add(e.folder)
            out.append(e)
    return out


__all__ = [
    "CATALOG_PATH",
    "LABELS",
    "CatalogError",
    "Event",
    "by_folder",
    "by_label",
    "groups",
    "known_lines",
    "known_sites",
    "load",
    "parse",
    "true_negative_candidates",
]
