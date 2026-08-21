"""
The inventory: what it refuses to load, and how it resolves sensor state.

The load tests matter more than they look. A typo that produces a site with no
sensors gives a site that is silent everywhere, which reads downstream as a
healthy site with nothing to say.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import inventory as inv

MINIMAL = {
    "staleness_hours": 2,
    "balance_service_days": 7,
    "in_service": {
        "upstream": {"ctd": "yes", "do": "-", "fc": "-", "ph": "-"},
        "oxford": {"stage": "yes", "balance_feed": "-"},
        "codornices": {"stage": "yes", "balance_feed": "-"},
    },
    "sensor_variables": {
        "ctd": ["conductivity", "temperature", "depth"],
        "do": ["dissolved_oxygen"],
        "stage": ["depth"],
    },
    "sensor_columns": {
        "mayfly": ["ctd", "do", "fc", "ph"],
        "balance_hydrologics": ["stage", "balance_feed"],
    },
    "sites": {
        "upstream": {
            "name": "Upstream",
            "source": "mayfly",
            "downstream": "oxford",
            "sensors": {"ctd": {"install": "2025-01-01"}, "do": {"install": None}},
        },
        "oxford": {
            "name": "Oxford Street",
            "source": "balance_hydrologics",
            "downstream": None,
            "sensors": {"stage": {"install": "2025-01-01"}},
        },
        "codornices": {
            "name": "Codornices Creek",
            "source": "balance_hydrologics",
            "downstream": None,
            "sensors": {"stage": {"install": "2025-01-01"}},
        },
    },
}


def build(**changes):
    raw = copy.deepcopy(MINIMAL)
    for path, value in changes.items():
        target = raw
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return raw


# Loading


def test_the_shipped_inventory_loads_and_covers_every_site():
    inventory = inv.load()
    assert len(inventory.sites) == 11
    assert inventory.isolated() == ["codornices"]
    for site in inventory.sites.values():
        assert site.sensors, f"{site.table} has no sensors"


def test_a_site_with_no_sensors_is_refused():
    """The failure this whole file exists to prevent."""
    raw = build(**{"sites.upstream.sensors": {}})
    with pytest.raises(inv.InventoryError, match="declares no sensors"):
        inv.from_dict(raw)


def test_an_unknown_sensor_name_is_refused():
    raw = build(
        **{
            "sites.upstream.sensors": {"turbidty": {"install": None}},
            "in_service.upstream": {"ctd": "-", "do": "-", "fc": "-", "ph": "-"},
        }
    )
    with pytest.raises(inv.InventoryError, match="is not a known probe"):
        inv.from_dict(raw)


def test_a_site_with_no_downstream_that_is_not_an_outlet_is_refused():
    raw = build(**{"sites.upstream.downstream": None})
    with pytest.raises(inv.InventoryError, match="no downstream neighbour"):
        inv.from_dict(raw)


def test_a_downstream_that_does_not_exist_is_refused():
    raw = build(**{"sites.upstream.downstream": "sather_gate"})
    with pytest.raises(inv.InventoryError, match="is not a site in this file"):
        inv.from_dict(raw)


def test_a_cycle_in_the_flow_graph_is_refused():
    raw = build(**{"sites.oxford.downstream": "upstream"})
    with pytest.raises(inv.InventoryError, match="loops back on itself"):
        inv.from_dict(raw)


def test_a_malformed_date_is_refused():
    raw = build(**{"sites.upstream.sensors": {"ctd": {"install": "the summer"}}})
    with pytest.raises(inv.InventoryError, match="not a date"):
        inv.from_dict(raw)


def test_a_removal_before_its_install_is_refused():
    raw = build(
        **{"sites.upstream.sensors": {"ctd": {"install": "2025-06-01", "removed": "2025-01-01"}}}
    )
    with pytest.raises(inv.InventoryError, match="before it was"):
        inv.from_dict(raw)


def test_an_unknown_source_is_refused():
    raw = build(**{"sites.upstream.source": "carrier_pigeon"})
    with pytest.raises(inv.InventoryError, match="is not a source this code knows"):
        inv.from_dict(raw)


# State resolution


def index_for(hours):
    return pd.date_range("2025-06-01", periods=hours * 4, freq="15min", tz="UTC")


def test_before_install_is_not_installed_and_after_install_with_no_reading_is_missing():
    """
    The distinction the whole inventory exists to make. These two are the same
    silence in the data and opposite things to whoever gets the alert.
    """
    inventory = inv.from_dict(build())
    index = pd.date_range("2024-12-20", periods=24, freq="D", tz="UTC")
    values = pd.Series(np.nan, index=index)

    states = inventory.resolve_series("upstream", "conductivity", values)
    before = states[states.index < pd.Timestamp("2025-01-01", tz="UTC")]
    after = states[states.index >= pd.Timestamp("2025-01-01", tz="UTC")]

    assert set(before) == {inv.NOT_INSTALLED}
    assert set(after) == {inv.MISSING}
    assert len(before) and len(after)


def test_a_probe_that_was_never_installed_is_never_anything_else():
    inventory = inv.from_dict(build())
    index = index_for(24)
    values = pd.Series(np.arange(len(index), dtype=float), index=index)

    states = inventory.resolve_series("upstream", "dissolved_oxygen", values)
    assert set(states) == {inv.NOT_INSTALLED}


def test_a_gap_inside_the_allowed_two_hours_is_stale_and_beyond_it_is_missing():
    inventory = inv.from_dict(build())
    index = index_for(12)
    values = pd.Series(np.nan, index=index)
    values.iloc[0] = 1.0

    states = inventory.resolve_series("upstream", "conductivity", values)
    assert states.iloc[0] == inv.PRESENT
    # 15 minute steps, so step 8 is two hours after the reading
    assert states.iloc[8] == inv.STALE
    assert states.iloc[9] == inv.MISSING


def test_absence_outside_the_balance_window_is_expected_and_the_same_gap_at_mayfly_is_a_fault():
    """
    Item 7. Read off the source in the inventory, never off the site name.
    """
    inventory = inv.from_dict(build())
    index = pd.date_range("2025-06-01", periods=30, freq="D", tz="UTC")
    values = pd.Series(np.nan, index=index)
    as_of = index[-1]

    balance = inventory.resolve_series("oxford", "depth", values, as_of=as_of)
    mayfly = inventory.resolve_series("upstream", "conductivity", values, as_of=as_of)

    old = index < as_of - pd.Timedelta(days=7)
    assert set(balance[old]) == {inv.NOT_INSTALLED}
    assert set(mayfly[old]) == {inv.MISSING}


def test_the_balance_rule_follows_the_source_when_the_source_changes():
    """Flip only the source and the same absence changes meaning."""
    index = pd.date_range("2025-06-01", periods=30, freq="D", tz="UTC")
    values = pd.Series(np.nan, index=index)
    as_of = index[-1]

    as_balance = inv.from_dict(build()).resolve_series("oxford", "depth", values, as_of=as_of)
    # same site and same variable, moved onto a Mayfly, so only the source differs
    flipped = build(
        **{
            "sites.oxford.source": "mayfly",
            "sites.oxford.sensors": {"ctd": {"install": "2025-01-01"}},
            "in_service.oxford": {"ctd": "yes", "do": "-", "fc": "-", "ph": "-"},
        }
    )
    as_mayfly = inv.from_dict(flipped).resolve_series("oxford", "depth", values, as_of=as_of)

    assert not as_balance.equals(as_mayfly)


def test_a_removed_probe_stops_being_installed():
    raw = build(
        **{"sites.upstream.sensors": {"ctd": {"install": "2025-01-01", "removed": "2025-06-02"}}}
    )
    inventory = inv.from_dict(raw)
    index = index_for(72)
    values = pd.Series(1.0, index=index)

    states = inventory.resolve_series("upstream", "conductivity", values)
    after = states[states.index >= pd.Timestamp("2025-06-02", tz="UTC")]
    assert set(after) == {inv.NOT_INSTALLED}


# Topology


def test_the_flow_graph_reaches_an_outlet_from_every_site():
    inventory = inv.load()
    for table in inventory.tables:
        chain = inventory.downstream_chain(table)
        assert table not in chain
        if table not in inv.TERMINAL_SITES:
            assert chain[-1] in inv.TERMINAL_SITES


def test_upstream_chain_is_transitive_and_downstream_chain_is_its_inverse():
    inventory = inv.load()
    assert "botanical_garden" in inventory.upstream_chain("oxford")
    assert "oxford" in inventory.downstream_chain("botanical_garden")
    assert inventory.upstream_chain("codornices") == []


def test_codornices_is_flow_disconnected_which_is_what_makes_it_a_control():
    inventory = inv.load()
    assert inventory.downstream_chain("codornices") == []
    assert inventory.upstream_chain("codornices") == []


def test_every_inferred_date_carries_the_reading_it_came_from():
    inventory = inv.load()
    inferred = inventory.inferred_dates()
    assert inferred
    for site, sensor, date, evidence in inferred:
        assert evidence, f"{site}.{sensor} was inferred with no evidence recorded"
        assert date in evidence or "first" in evidence


def test_the_dates_that_are_known_are_not_marked_inferred():
    """scnf010 DO and FC have real install dates, so nothing backfilled them."""
    inventory = inv.load()
    for name in ("do", "fc"):
        sensor = inventory.site("scnf010").sensors[name]
        assert sensor.install == pd.Timestamp("2026-03-05", tz="UTC")
        assert not sensor.inferred


def test_the_scalar_and_series_paths_agree():
    """
    resolve_state and resolve_series apply the same rule twice, so they get
    pinned together. Two implementations that quietly disagree is how a state
    table stops matching what the live path reports.
    """
    inventory = inv.from_dict(build())
    index = index_for(6)
    values = pd.Series(np.nan, index=index)
    values.iloc[0] = 1.0
    values.iloc[12] = 2.0

    series = inventory.resolve_series("upstream", "conductivity", values)

    present = values.notna().to_numpy()
    last = None
    for i, stamp in enumerate(index):
        if present[i]:
            last = stamp
        one = inventory.resolve_state("upstream", "conductivity", stamp, last_reading=last)
        assert one == series.iloc[i], f"step {i} disagrees: {one} against {series.iloc[i]}"


def test_resolve_state_honours_the_install_date():
    inventory = inv.from_dict(build())
    before = pd.Timestamp("2024-06-01", tz="UTC")
    after = pd.Timestamp("2025-06-01", tz="UTC")

    assert inventory.resolve_state("upstream", "conductivity", before) == inv.NOT_INSTALLED
    assert inventory.resolve_state("upstream", "conductivity", after) == inv.MISSING
    assert (
        inventory.resolve_state("upstream", "conductivity", after, last_reading=after)
        == inv.PRESENT
    )


def test_resolve_state_honours_a_removal_date():
    raw = build(
        **{"sites.upstream.sensors": {"ctd": {"install": "2025-01-01", "removed": "2025-06-01"}}}
    )
    inventory = inv.from_dict(raw)
    inside = pd.Timestamp("2025-03-01", tz="UTC")
    outside = pd.Timestamp("2025-09-01", tz="UTC")

    assert inventory.resolve_state("upstream", "conductivity", inside) == inv.MISSING
    assert inventory.resolve_state("upstream", "conductivity", outside) == inv.NOT_INSTALLED
