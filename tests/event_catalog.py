"""
The labelled event catalog, read off tests/events.yaml.

The catalog used to be a Python list inside tests/test_anomaly_detection.py and
a second, drifting copy of it inside scripts/run_audit_comparison.py. Two copies
of a list of facts is the same defect class tests/test_audit_sweep.py exists to
catch, and it also meant adding an event was a code edit in two places. Both
readers now come through here, so adding the seventh event is one entry in one
file.

Nothing in here loads data or scores anything. It validates the shape of the
catalog and hands back rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from strawberrywatch import paths

CATALOG_PATH = Path(__file__).with_name("events.yaml")

# What a label commits the detector to. relative_only is a record of an event we
# know about rather than coverage of it: nothing grades those rows yet.
LABELS = ("anomaly", "true_negative", "relative_only")

REQUIRED = ("folder", "group", "site", "label")
OPTIONAL = ("note",)


class CatalogError(ValueError):
    """The catalog is malformed. The message names the entry and the fix."""


@dataclass(frozen=True)
class Event:
    """One (event, site) pair and what the detector must say about it."""

    folder: str
    group: str
    site: str
    label: str
    note: str = ""

    @property
    def path(self) -> Path:
        """Where the fixture CSVs live. Not guaranteed to exist."""
        return paths.anomalies_dir() / self.folder

    def as_tuple(self):
        """The (folder, site, label, group) shape the event tests parametrize on."""
        return (self.folder, self.site, self.label, self.group)


def _fail(index, entry, message):
    name = entry.get("group") or entry.get("folder") or "?" if isinstance(entry, dict) else "?"
    raise CatalogError(f"tests/events.yaml entry {index} ({name}): {message}")


def parse(blob, known_sites=None):
    """
    Validate an already-parsed mapping and return its events, in file order.

    known_sites, when given, is the set of site names the repo recognises. A
    typo in a site name would otherwise surface as a skipped test rather than a
    failing one, which is the quiet failure mode this catalog exists to avoid.
    """
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
            )
        )
    return events


def known_sites():
    """
    Site names the repo recognises, in either vocabulary.

    The model calls the Wickson footbridge 'footbridge' and the inventory calls
    its table 'scnf010'. A catalog entry may reasonably use either, so both are
    accepted and the union is what a typo is checked against.
    """
    from strawberrywatch.models.Cobble_Shoal import SITE_ORDER

    names = set(SITE_ORDER)
    try:
        from strawberrywatch import inventory as inventory_module

        names |= set(inventory_module.load().sites)
    except Exception:
        # An unloadable inventory is its own test's problem, not this one's.
        pass
    return names


def load(path=CATALOG_PATH, check_sites=True):
    """Every event in the catalog, in file order."""
    path = Path(path)
    if not path.exists():
        raise CatalogError(f"no event catalog at {path}")
    blob = yaml.safe_load(path.read_text()) or {}
    return parse(blob, known_sites=known_sites() if check_sites else None)


def by_label(label, events=None):
    """The events carrying one label, in file order."""
    return [e for e in (events if events is not None else load()) if e.label == label]


def by_folder(events=None):
    """
    One row per event folder: (folder, group, sites, label), in file order.

    Reports that score a whole event window rather than a single node want the
    folder, not the (folder, site) pair. Where a folder carries more than one
    label its first entry decides, since that is the row the folder is named
    for; the rest are the same event judged at other nodes.
    """
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
    "known_sites",
    "load",
    "parse",
]
