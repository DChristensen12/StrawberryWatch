"""Shipped events load, edits here override them, and the shipped files never change."""

import hashlib

import pandas as pd
import pytest

from strawberrywatch.watch import sources
from strawberrywatch.workbench import events, store


def shipped_hashes():
    return {
        p.relative_to(sources.EVENTS_DIR): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources.EVENTS_DIR.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def folder(tmp_path):
    return store.ready(tmp_path / "wb")


def test_shipped_events_load(folder):
    every = events.every(folder)
    folders = {e["folder"] for e in sources.replay_events()}
    assert {e["id"] for e in every} == folders
    hydrant = events.one(folder, "anomaly_2026_03_20_hydrant_nf0")
    assert hydrant["kind"] == "spill" and hydrant["should_alert"] and hydrant["shipped"]
    assert hydrant["sites"] == ["north_fork_0"] and hydrant["start"] < hydrant["end"]
    assert events.one(folder, "anomaly_2025_11_13_rain_nf1")["should_alert"] is False


def test_editing_a_shipped_event_overrides_it_and_leaves_its_files_alone(folder):
    before = shipped_hashes()
    event = events.one(folder, "anomaly_2026_03_20_hydrant_nf0")
    start = pd.Timestamp("2026-03-20 18:15", tz="UTC")
    events.save(
        folder,
        event["id"],
        {**event, "start": start, "end": start + pd.Timedelta(hours=7), "note": "Euclid Ave"},
    )
    changed = events.one(folder, event["id"])
    assert changed["note"] == "Euclid Ave" and changed["start"] == start and changed["shipped"]
    assert len(events.every(folder)) == len(sources.replay_events())
    assert shipped_hashes() == before


def test_only_added_events_can_be_deleted(folder):
    start = pd.Timestamp("2026-10-14 17:00", tz="UTC")
    added = events.save(
        folder,
        None,
        {
            "name": "Sensor cleaning",
            "kind": "visit",
            "sites": ["south_fork_1"],
            "start": start,
            "end": start + pd.Timedelta(hours=1),
            "should_alert": False,
            "note": "",
        },
    )
    assert events.one(folder, added)["shipped"] is False
    assert not events.delete(folder, "anomaly_2026_03_20_hydrant_nf0")
    assert events.one(folder, "anomaly_2026_03_20_hydrant_nf0") is not None
    assert events.delete(folder, added)
    assert events.one(folder, added) is None
