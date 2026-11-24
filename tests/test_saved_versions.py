"""Saved versions of a model's files, for switching to one and back later."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from strawberrywatch import paths
from strawberrywatch.serving import detector_class
from strawberrywatch.watch import model_updates as mu
from strawberrywatch.watch import runner

PACIFIC = timezone(timedelta(hours=-8))


def _save(base, title="production calibration", when=None):
    files = detector_class("water_strider").FILES
    return mu.save_version(
        "water_strider",
        title,
        paths.model_dir("water_strider"),
        files,
        "shipped",
        "d6ffed64ffcd5f9c4838f6f3be2c8b1082cb1c5e",
        base=base,
        now=when or datetime(2026, 11, 16, 23, 58, 42, tzinfo=PACIFIC),
    )


def test_a_saved_version_holds_the_files_and_a_manifest(tmp_path):
    folder = _save(tmp_path)
    assert folder == tmp_path / "saved" / "water_strider" / "2026-11-16_production-calibration"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["saved_at"] == "2026-11-16T23:58:42-08:00"
    assert manifest["origin"] == "shipped" and manifest["model"] == "water_strider"
    assert manifest["files"] == {
        "water_strider_calibration.json": mu.sha256(
            (folder / "water_strider_calibration.json").read_bytes()
        )
    }
    assert not list(folder.parent.glob(".*"))


def test_saving_again_makes_a_new_folder_and_lists_newest_first(tmp_path):
    first = _save(tmp_path)
    later = datetime(2026, 11, 17, 9, 0, tzinfo=PACIFIC)
    second = _save(tmp_path, when=later)
    third = _save(tmp_path, when=later)
    assert len({first, second, third}) == 3 and third.name.endswith("-2")
    names = [folder for folder, _ in mu.saved_versions("water_strider", base=tmp_path)]
    assert names[-1] == first and set(names[:2]) == {second, third}


def test_a_changed_file_drops_the_version_and_fails_to_load(tmp_path):
    folder = _save(tmp_path)
    assert runner.load_detector("water_strider", folder) is not None
    path = folder / "water_strider_calibration.json"
    path.write_text(path.read_text() + " ")
    assert mu.saved_versions("water_strider", base=tmp_path) == []
    with pytest.raises(runner.ModelFileError):
        runner.load_detector("water_strider", folder)


def test_the_manifest_keeps_details_only_when_given(tmp_path):
    assert "details" not in json.loads((_save(tmp_path) / "manifest.json").read_text())
    folder = mu.save_version(
        "water_strider",
        "with details",
        paths.model_dir("water_strider"),
        detector_class("water_strider").FILES,
        "trained",
        "abc",
        base=tmp_path,
        details={"rain_model": "ncep_nbm_conus"},
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["details"] == {"rain_model": "ncep_nbm_conus"}
    assert runner.load_detector("water_strider", folder) is not None


def test_an_unknown_origin_is_refused(tmp_path):
    with pytest.raises(ValueError, match="isn't one of"):
        mu.save_version("water_strider", "x", tmp_path, (), "somewhere", "abc", base=tmp_path)


def test_a_seed_gives_the_same_weights_and_the_manifest_keeps_it(tmp_path):
    import torch

    from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
    from strawberrywatch.training.trainer import TRAIN_THREADS, seed_everything

    def built(seed):
        seed_everything(seed)
        model = DuskCrayfish(num_node_features=10, num_nodes=4)
        return [p.detach().clone() for p in model.parameters()]

    first, again, other = built(7), built(7), built(8)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))
    # the saved NBM seed 0 weights were trained at 6 threads
    assert torch.get_num_threads() == TRAIN_THREADS == 6
    files = detector_class("water_strider").FILES
    folder = mu.save_version(
        "water_strider",
        "seeded",
        paths.model_dir("water_strider"),
        files,
        "trained",
        "abc",
        base=tmp_path,
        seed=7,
        threads=TRAIN_THREADS,
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["seed"] == 7 and manifest["threads"] == 6
