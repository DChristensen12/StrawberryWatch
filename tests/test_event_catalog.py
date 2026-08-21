"""
The labelled event catalog as a thing somebody edits without writing Python.

Adding an event is meant to be one entry in tests/events.yaml. That only holds
if a bad entry says what is wrong instead of turning into a skipped test, so
every validation rule gets its bad value written and its message checked.

The fixtures themselves are not committed (see the folder check at the bottom),
so anything that needs the CSVs on disk skips rather than fails.
"""

from __future__ import annotations

import pytest
import yaml

from strawberrywatch import paths
from tests import event_catalog as ec


@pytest.fixture(scope="module")
def events():
    return ec.load()


def _blob(**overrides):
    """One valid entry, with whatever the caller wants broken."""
    entry = {
        "folder": "anomaly_2025_06_12_spill_sf",
        "group": "jun25_spill",
        "site": "south_fork_1",
        "label": "anomaly",
    }
    entry.update(overrides)
    return {"events": [entry]}


# The file loads, and says what it should


def test_the_shipped_catalog_loads(events):
    assert events, "tests/events.yaml produced no events"


def test_every_label_is_one_the_tests_know(events):
    assert {e.label for e in events} <= set(ec.LABELS)


def test_the_six_graded_events_are_still_graded(events):
    """
    The graded rows are the evidence base. A refactor that quietly dropped one
    would leave a green suite that tests less than it did, which is the failure
    this file exists to make loud.
    """
    graded = {(e.folder, e.site, e.label) for e in events if e.label != "relative_only"}
    assert graded == {
        ("anomaly_2025_06_12_spill_sf", "south_fork_1", "anomaly"),
        ("anomaly_2025_09_10_overnight_sf", "south_fork_1", "anomaly"),
        ("anomaly_2025_09_10_overnight_sf", "south_fork_2", "anomaly"),
        ("anomaly_2026_03_20_hydrant_nf0", "north_fork_0", "anomaly"),
        ("anomaly_2026_04_01_rainfall", "north_fork_0", "true_negative"),
        ("anomaly_2026_04_01_rainfall", "south_fork_2", "true_negative"),
    }


def test_the_catalog_has_exactly_one_definition():
    """
    It used to live in two places: a list in tests/test_anomaly_detection.py and
    a second copy in scripts/run_audit_comparison.py. Both now read this module,
    and neither may grow its own copy back.
    """
    import inspect

    import tests.test_anomaly_detection as tad

    assert tad.EVENT_CATALOG == [e.as_tuple() for e in ec.load()]

    src = inspect.getsource(tad)
    body = src[src.index("EVENT_CATALOG") :]
    assert "anomaly_2025_" not in body and "anomaly_2026_" not in body, (
        "test_anomaly_detection.py is naming event folders again; the catalog is in events.yaml"
    )

    script = (paths.project_root() / "scripts" / "run_audit_comparison.py").read_text()
    assert "event_catalog" in script, "run_audit_comparison stopped reading the shared catalog"
    assert "anomaly_2025_" not in script and "anomaly_2026_" not in script, (
        "run_audit_comparison.py grew its own copy of the event list again"
    )


# What a bad entry says


def test_a_missing_key_names_the_key():
    with pytest.raises(ec.CatalogError, match="missing site"):
        ec.parse(_blob(site=None))


def test_an_unknown_label_lists_the_ones_that_work():
    with pytest.raises(ec.CatalogError, match="is not one of"):
        ec.parse(_blob(label="probably_bad"))


def test_a_misspelled_site_is_refused_rather_than_skipped():
    """
    An unrecognised site would otherwise reach the model, miss location_to_idx
    and skip. A skipped test reads as a passing suite.
    """
    with pytest.raises(ec.CatalogError, match="not a site this repo knows"):
        ec.parse(_blob(site="south_fork_9"), known_sites=ec.known_sites())


def test_a_site_name_in_either_vocabulary_is_accepted():
    """The model says footbridge, the inventory says scnf010. Both are the same place."""
    for name in ("footbridge", "scnf010"):
        assert ec.parse(_blob(site=name), known_sites=ec.known_sites())[0].site == name


def test_a_typo_in_a_key_is_refused_rather_than_ignored():
    """A silently dropped key is a label nobody notices is not being applied."""
    with pytest.raises(ec.CatalogError, match="unknown key"):
        ec.parse(_blob(lable="anomaly"))


def test_the_same_event_at_the_same_site_twice_is_refused():
    blob = _blob()
    blob["events"] = blob["events"] * 2
    with pytest.raises(ec.CatalogError, match="duplicates entry"):
        ec.parse(blob)


def test_the_same_event_at_two_sites_is_fine():
    blob = _blob()
    blob["events"] = [
        dict(blob["events"][0]),
        dict(blob["events"][0], site="south_fork_2", label="relative_only"),
    ]
    assert len(ec.parse(blob)) == 2


def test_an_empty_or_shapeless_file_says_so():
    for bad, message in (
        ({}, "no top-level"),
        ({"events": []}, "non-empty list"),
        ({"events": ["anomaly_2025_06_12_spill_sf"]}, "not a mapping"),
    ):
        with pytest.raises(ec.CatalogError, match=message):
            ec.parse(bad)


# The catalog against the disk


def test_every_entry_points_at_a_folder_that_exists(events):
    """
    Skips wholesale when data/anomalies is absent, because the fixtures are not
    in git. Present-but-missing-a-folder is a typo and fails.
    """
    if not paths.anomalies_dir().exists():
        pytest.skip("no data/anomalies on this machine")
    missing = sorted({e.folder for e in events if not e.path.is_dir()})
    assert not missing, f"catalog names folders that are not in data/anomalies: {missing}"


def test_every_entry_has_a_csv_for_the_site_it_scores(events):
    """
    A folder present but missing the target site's CSV skips at score time, so
    check it here where it fails instead. footbridge is stored under its table
    name, so both spellings count.
    """
    if not paths.anomalies_dir().exists():
        pytest.skip("no data/anomalies on this machine")
    from strawberrywatch.preprocessing.node_windows import SITE_TO_TABLE

    missing = []
    for e in events:
        if not e.path.is_dir():
            continue
        names = {e.site, SITE_TO_TABLE.get(e.site, e.site)}
        if not any((e.path / f"{n}.csv").exists() for n in names):
            missing.append(f"{e.folder}/{e.site}")
    assert not missing, f"no CSV for the scored site: {missing}"


def test_folders_on_disk_that_the_catalog_ignores_are_named_in_the_file():
    """
    An event fixture nobody listed is evidence sitting unused, which is how
    anomaly_2025_08_sprinklers_nf0 went ungraded for months. Anything left out
    has to be left out on purpose, in writing, in the file itself.
    """
    if not paths.anomalies_dir().exists():
        pytest.skip("no data/anomalies on this machine")
    text = ec.CATALOG_PATH.read_text()
    listed = {e.folder for e in ec.load()}
    on_disk = {p.name for p in paths.anomalies_dir().iterdir() if p.is_dir()}
    unexplained = sorted(f for f in on_disk - listed if f not in text)
    assert not unexplained, (
        "event folders on disk that events.yaml neither grades nor explains: "
        f"{unexplained}. Add an entry, or say in the file why not."
    )


def test_the_file_still_tells_an_editor_what_to_do():
    """The header is the interface for anyone adding an event without reading Python."""
    text = ec.CATALOG_PATH.read_text()
    head = text[: text.index("events:")]
    for needed in ("data/anomalies/", "rebuild_fixture.py", "anomaly", "true_negative"):
        assert needed in head, f"the events.yaml header stopped explaining {needed!r}"
    assert yaml.safe_load(text)["events"], "the header swallowed the events list"
