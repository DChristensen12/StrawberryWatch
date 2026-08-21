"""
The adapter that lets Cobble Shoal read the real corpus.

Three properties carry the weight: no lookahead, agreement with the synthetic
generator it mirrors, and never turning an uninstalled sensor into a zero.
The rest is plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from strawberrywatch import inventory as inv
from strawberrywatch.models.Cobble_Shoal import SITE_INVENTORY, SITE_ORDER
from strawberrywatch.preprocessing import node_windows as nw
from tests.synthetic import creek_synthetic as cs

WINDOW = 24


@pytest.fixture(scope="module")
def inventory():
    return inv.load()


@pytest.fixture(scope="module")
def cobble():
    """The shipped weights and a null, or skip. Used only where the model matters."""
    from strawberrywatch.anomalies import cobble_calibration
    from strawberrywatch.models.Cobble_Shoal import CobbleShoal

    try:
        cal = cobble_calibration.load_calibration()
    except cobble_calibration.CalibrationError as exc:
        pytest.skip(str(exc))
    path = cobble_calibration.weights_path()
    if not path.exists():
        pytest.skip(f"no weights at {path}")
    net = CobbleShoal.from_metadata({"seed": cal.seed, "window": WINDOW})
    blob = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
    net.eval()
    return net, cal.nulls


@pytest.fixture(scope="module")
def tables():
    """A small hand-built archive, so these tests do not depend on data/raw_data."""
    idx = pd.date_range("2026-04-01", periods=200, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    out = {}
    for table, base in (
        ("north_fork_0", 500.0),
        ("scnf010", 350.0),
        ("south_fork_1", 580.0),
        ("south_fork_2", 600.0),
        ("oxford", 340.0),
    ):
        out[table] = pd.DataFrame(
            {
                "conductivity": base + rng.normal(0, 5, len(idx)),
                "depth": 100.0 + rng.normal(0, 3, len(idx)),
                "temperature": 15.0 + rng.normal(0, 0.5, len(idx)),
                "dissolved_oxygen": 50.0 + rng.normal(0, 2, len(idx)),
                "floating_conductivity": base + rng.normal(0, 5, len(idx)),
            },
            index=idx,
        ).rename_axis("datetime")
    return out


# The roster comes from the inventory


def test_the_roster_is_read_off_the_inventory_not_a_list_in_the_model(inventory):
    assert nw.roster_from_inventory(inventory) == {s: list(v) for s, v in SITE_INVENTORY.items()}


def test_a_sensor_the_grid_marks_absent_yields_no_node(inventory):
    """north_fork_0 has no DO probe, so there is no north_fork_0.do_pct node."""
    roster = nw.roster_from_inventory(inventory)
    assert "do_pct" not in roster["north_fork_0"]
    assert "do_pct" in roster["footbridge"]


def test_a_roster_the_weights_cannot_take_is_refused_by_name():
    bad = {s: list(v) for s, v in SITE_INVENTORY.items()}
    bad["oxford"] = bad["oxford"] + ["do_pct"]
    with pytest.raises(nw.RosterMismatch) as exc:
        nw.check_roster(bad, {s: list(v) for s, v in SITE_INVENTORY.items()})
    assert "oxford" in str(exc.value)
    assert "18" in str(exc.value) and "17" in str(exc.value), "the node counts are not named"


def test_the_window_has_the_node_count_the_checkpoint_was_built_for(tables, inventory):
    win = nw.build_window(tables, tables["oxford"].index[0], tables["oxford"].index[-1], inventory)
    assert len(win["nodes"]) == sum(len(v) for v in SITE_INVENTORY.values()) == 17
    assert [n.key for n in win["nodes"]][0] == "north_fork_0.conductivity"


# sensor state goes through the mask


def test_a_sentinel_is_not_a_reading(tables, inventory):
    """-9999 is a wrong SDI-12 address. Left in it owns the node's variance."""
    dirty = {k: v.copy() for k, v in tables.items()}
    dirty["north_fork_0"].iloc[40:60, dirty["north_fork_0"].columns.get_loc("conductivity")] = -9999

    span = (tables["oxford"].index[0], tables["oxford"].index[-1])
    clean_std = nw.build_window(tables, *span, inventory)["scaler"].std[0]
    dirty_std = nw.build_window(dirty, *span, inventory)["scaler"].std[0]
    assert dirty_std == pytest.approx(clean_std, rel=0.5), (
        f"sentinels moved the scale from {clean_std:.2f} to {dirty_std:.2f}"
    )


def test_a_probe_that_was_never_installed_is_masked_not_zero_filled(tables, inventory):
    """A zero the model reads as real is worse than a hole."""
    win = nw.build_window(tables, tables["oxford"].index[0], tables["oxford"].index[-1], inventory)
    keys = [n.key for n in win["nodes"]]

    # south_fork_1's ctd went in 2025-06-04, so a window before that is absence.
    early = pd.date_range("2025-01-01", periods=len(tables["oxford"]), freq="15min", tz="UTC")
    shifted = {k: v.set_axis(early).rename_axis("datetime") for k, v in tables.items()}
    before = nw.build_window(shifted, early[0], early[-1], inventory)
    i = keys.index("south_fork_1.conductivity")
    assert not before["target_mask"][:, i].any(), "a pre-install probe produced targets"
    assert (before["staleness"][:, i] >= len(before["grid"])).all(), "it looks merely stale"


def test_a_switched_off_sensor_leaves_the_window(tables, inventory, tmp_path):
    import shutil

    copy = tmp_path / "inventory.yaml"
    shutil.copy(inv._yaml_path(), copy)
    inv.set_in_service("scnf010", "do", False, path=copy)
    off = inv.load(path=copy, reload=True)

    span = (tables["oxford"].index[0], tables["oxford"].index[-1])
    on_win = nw.build_window(tables, *span, inventory)
    keys = [n.key for n in on_win["nodes"]]
    i = keys.index("footbridge.do_pct")
    assert on_win["target_mask"][:, i].any(), "the probe was already silent"

    off_win = nw.build_window(tables, *span, off)
    assert not off_win["target_mask"][:, i].any(), "a switched off probe still reports"
    assert len(off_win["nodes"]) == len(on_win["nodes"]), "the node registry moved"


# no lookahead


def test_perturbing_the_future_does_not_move_the_present(tables, inventory):
    """
    The scored step is bit-identical when data after it changes.

    Fixed scaler on purpose. Fitting one on the window is itself lookahead,
    which is why build_window takes one, so holding it still measures the regrid.
    """
    span = (tables["oxford"].index[0], tables["oxford"].index[-1])
    base = nw.build_window(tables, *span, inventory)
    scaler = base["scaler"]
    anchor = 100

    later = {k: v.copy() for k, v in tables.items()}
    cutoff = base["grid"][anchor + 4]
    for frame in later.values():
        frame.loc[frame.index > cutoff] = frame.loc[frame.index > cutoff] * 7.0 + 1000.0

    after = nw.build_window(
        tables=later, start=span[0], end=span[1], inventory=inventory, scaler=scaler
    )

    np.testing.assert_array_equal(base["values"][: anchor + 1], after["values"][: anchor + 1])
    np.testing.assert_array_equal(base["target_val"][:anchor], after["target_val"][:anchor])
    np.testing.assert_array_equal(base["target_mask"][:anchor], after["target_mask"][:anchor])


def test_the_model_output_at_the_window_end_is_bit_identical(tables, inventory, cobble):
    """Same thing through the model, which is where it matters."""
    span = (tables["oxford"].index[0], tables["oxford"].index[-1])
    base = nw.build_window(tables, *span, inventory)
    scaler = base["scaler"]
    anchor = 100

    later = {k: v.copy() for k, v in tables.items()}
    cutoff = base["grid"][anchor + 4]
    for frame in later.values():
        frame.loc[frame.index > cutoff] = frame.loc[frame.index > cutoff] * 7.0 + 1000.0
    after = nw.build_window(later, *span, inventory, scaler=scaler)

    model, nulls = cobble
    a = model.score(nw.to_batch(base, anchor, WINDOW), nulls)
    b = model.score(nw.to_batch(after, anchor, WINDOW), nulls)
    assert a.tobytes() == b.tobytes(), "the future changed the score at the window end"


# the adapter and the generator agree


def test_the_two_paths_produce_the_same_node_tensors(inventory):
    """
    Same readings through both paths. They call the same regrid now, so what
    this really checks is that the adapter's name translation and masking are
    transparent.
    """
    idx = pd.date_range("2026-04-01", periods=120, freq="15min", tz="UTC")
    rng = np.random.default_rng(4)

    canonical, archive = {}, {}
    for site, variables in SITE_INVENTORY.items():
        table = nw.SITE_TO_TABLE.get(site, site)
        cols_c, cols_a = {}, {}
        for var in variables:
            raw = next(k for k, v in nw.VARIABLE_MAP.items() if v == var)
            series = 100.0 + rng.normal(0, 3, len(idx))
            cols_c[var] = series
            cols_a[raw] = series
        canonical[site] = pd.DataFrame(cols_c, index=idx).rename_axis("datetime")
        archive[table] = pd.DataFrame(cols_a, index=idx).rename_axis("datetime")

    nodes, _site, _var = nw.build_node_registry(
        {s: list(v) for s, v in SITE_INVENTORY.items()}, SITE_ORDER
    )
    grid = pd.date_range(idx[0], idx[-1], freq=nw.GRID_FREQ, tz="UTC")
    want = nw.regrid_to_nodes(canonical, grid, nodes)

    got = nw.build_window(archive, idx[0], idx[-1], inventory)

    # depth_dev is the site median removed; the scaler removes it again, so
    # compare the scaled arrays rather than the raw ones.
    scaler = nw.NodeScaler().fit(want[0], want[3])
    np.testing.assert_allclose(got["values"], scaler.transform(want[0]), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(got["staleness"], want[1])
    np.testing.assert_array_equal(got["target_mask"], want[3])


def test_a_synthetic_window_and_a_real_one_have_the_same_shape(tables, inventory):
    """Anything that scores one has to score the other without a branch."""
    real = nw.build_window(tables, tables["oxford"].index[0], tables["oxford"].index[-1], inventory)
    fake = cs.make_synthetic_creek(n_steps=144, seed=0)
    for key in ("values", "staleness", "target_val", "target_mask", "context"):
        assert np.asarray(real[key]).dtype == np.asarray(fake[key]).dtype, key
        assert np.asarray(real[key]).shape[1:] == np.asarray(fake[key]).shape[1:], key
    assert set(fake) - set(real) == {"truth"}, "the real window is missing a key"


def test_to_batch_matches_the_synthetic_builder():
    from tests.synthetic import nested_batch

    win = cs.make_synthetic_creek(n_steps=144, seed=0)
    win["roster"] = {s: list(v) for s, v in SITE_INVENTORY.items()}
    mine = nw.to_batch(win, 120, WINDOW)
    theirs = nested_batch.build_batch(seed=0)
    for key in ("values", "staleness", "context", "target", "target_mask", "obs_mask"):
        assert torch.equal(mine[key], theirs[key]), key


# Dusk Crayfish has not moved


def test_dusk_crayfish_weights_and_metadata_are_untouched():
    """
    Its weights are the audit baseline. Nothing in the adapter touches its path,
    and this asserts it instead of trusting it.
    """
    import pickle

    from strawberrywatch.paths import checkpoints_dir

    path = checkpoints_dir() / "dusk_crayfish_metadata.pkl"
    if not path.exists():
        pytest.skip("no Dusk Crayfish metadata")
    meta = pickle.loads(path.read_bytes())

    assert meta["feature_cols"] == [
        "conductivity",
        "depth",
        "temperature",
        "rain_mm",
        "air_temp_c",
        "shortwave_radiation",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]
    assert sorted(meta["node_thresholds"]) == [
        "north_fork_0",
        "oxford",
        "south_fork_1",
        "south_fork_2",
    ]
    assert meta["scaler"] is not None


def test_dusk_crayfish_scores_deterministically():
    """Two loads give the same numbers, or a comparison against it means nothing."""
    import pickle

    from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
    from strawberrywatch.paths import checkpoints_dir

    weights = checkpoints_dir() / "dusk_crayfish_weights.pt"
    meta_path = checkpoints_dir() / "dusk_crayfish_metadata.pkl"
    if not (weights.exists() and meta_path.exists()):
        pytest.skip("no Dusk Crayfish checkpoint")
    meta = pickle.loads(meta_path.read_bytes())

    def run():
        net = DuskCrayfish.from_metadata(meta)
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        net.eval()
        torch.manual_seed(0)
        x = torch.zeros(1, 12, 4, len(meta["feature_cols"]))
        edge = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            return net(x, edge, 1, 4)

    a, b = run(), run()
    assert torch.equal(a, b), "Dusk Crayfish is not deterministic"
