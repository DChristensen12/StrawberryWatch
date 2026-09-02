"""
Reading a trained checkpoint without needing this repository around it.

The training path writes two files per model, {name}_weights.pt and
{name}_metadata.pkl, into whatever directory paths.checkpoints_dir() resolves
to. That resolver walks up from the package looking for a pyproject.toml or a
.git, so it works in a checkout and raises in site-packages. Everything here
takes the directory as an argument instead.

The metadata pickle holds a live sklearn StandardScaler. Two problems with
that once the checkpoint leaves this repo. Unpickling it imports sklearn and
runs its constructor, and the pickle is tied to the sklearn version that wrote
it, which is not necessarily the one on the machine reading it. Normalizing is
subtract a mean and divide by a scale, so we pull those two arrays out once and
let a caller store them as JSON. After that the serving path never needs sklearn
at all.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

SIDECAR_SUFFIX = "_serving.json"

# Everything detect_anomalies looks up per site. Kept in one place because the
# sidecar and the pickle both have to carry all of it or neither is usable.
_CALIBRATION_KEYS = ("error_median", "error_iqr", "node_thresholds", "cond_median", "cond_iqr")


class CheckpointError(RuntimeError):
    """Something about the checkpoint on disk stops us building a model from it."""


class Checkpoint:
    """
    One trained model's weights path plus everything needed to run it.

    Deliberately not the model itself. Building the model needs torch, and there
    are useful things to do with a checkpoint (check which sites it covers, check
    which features it wants) that should not cost a torch import.
    """

    def __init__(
        self,
        weights_path,
        feature_cols,
        location_to_idx,
        means,
        scales,
        calibration,
        architecture=None,
        source=None,
    ):
        self.weights_path = Path(weights_path)
        self.feature_cols = list(feature_cols)
        self.location_to_idx = dict(location_to_idx)
        self.means = list(means)
        self.scales = list(scales)
        self.calibration = calibration
        self.architecture = architecture or {}
        self.source = source

    @property
    def sites(self):
        """Site names in node-index order, which is the order the model expects."""
        return [s for s, _ in sorted(self.location_to_idx.items(), key=lambda kv: kv[1])]

    def normalization(self):
        """{feature: (mean, scale)}, looked up by name rather than position."""
        return dict(zip(self.feature_cols, zip(self.means, self.scales, strict=True), strict=True))

    def detection_metadata(self):
        """The dict detect_anomalies reads. Same keys the training path pickles."""
        return {
            "feature_cols": self.feature_cols,
            "location_to_idx": self.location_to_idx,
            **self.calibration,
        }

    def as_dict(self):
        return {
            "feature_cols": self.feature_cols,
            "location_to_idx": self.location_to_idx,
            "means": self.means,
            "scales": self.scales,
            "architecture": self.architecture,
            **self.calibration,
        }


def _paths(checkpoint_dir, model_name):
    d = Path(checkpoint_dir).expanduser()
    return (
        d / f"{model_name}_weights.pt",
        d / f"{model_name}_metadata.pkl",
        d / f"{model_name}{SIDECAR_SUFFIX}",
    )


def _from_blob(blob, weights, source, scaler=None):
    feature_cols = blob.get("feature_cols")
    location_to_idx = blob.get("location_to_idx")
    if not feature_cols or not location_to_idx:
        raise CheckpointError(
            f"{source} has no feature_cols or no location_to_idx, so there is no way "
            f"to know what columns the model wants or what order its nodes are in. "
            f"Retrain, or point at a different checkpoint."
        )

    if scaler is not None:
        means, scales = list(scaler.mean_), list(scaler.scale_)
    else:
        means, scales = blob.get("means"), blob.get("scales")
    if means is None or scales is None:
        raise CheckpointError(f"{source} carries no normalization stats")
    if not (len(means) == len(scales) == len(feature_cols)):
        raise CheckpointError(
            f"{source} has {len(feature_cols)} features but {len(means)} means and "
            f"{len(scales)} scales. These line up by position, so a mismatch means "
            f"the file was written by something that disagreed about the feature set."
        )

    calibration = {k: blob.get(k, {}) for k in _CALIBRATION_KEYS}
    if not calibration["node_thresholds"]:
        raise CheckpointError(
            f"{source} has no node_thresholds. Detection compares each site against "
            f"its own calibrated bar and will not invent one, so every node would be "
            f"skipped and the run would look clean no matter what the water did."
        )

    return Checkpoint(
        weights_path=weights,
        feature_cols=feature_cols,
        location_to_idx=location_to_idx,
        means=means,
        scales=scales,
        calibration=calibration,
        architecture=blob.get("architecture"),
        source=source,
    )


def load_checkpoint(checkpoint_dir, model_name="dusk_crayfish"):
    """
    Read a checkpoint out of one directory.

    Prefers the JSON sidecar if it is there, because that path touches no pickle
    and no sklearn. Falls back to the metadata pickle, which is what every
    checkpoint trained so far actually has.
    """
    weights, meta_pickle, sidecar = _paths(checkpoint_dir, model_name)
    if not weights.exists():
        raise CheckpointError(
            f"no weights at {weights}. The checkpoint directory is passed in rather "
            f"than discovered, so this is usually a wrong path, not a missing model."
        )

    if sidecar.exists():
        return _from_blob(json.loads(sidecar.read_text()), weights, str(sidecar))

    if not meta_pickle.exists():
        raise CheckpointError(
            f"found weights at {weights} but neither {sidecar.name} nor "
            f"{meta_pickle.name} beside them. Weights alone do not say what features "
            f"the model wants, what order its nodes are in, or where its thresholds sit."
        )

    # Unpickling runs whatever the file references, so this is only safe on a
    # checkpoint you produced. That is the other reason the sidecar exists.
    with open(meta_pickle, "rb") as f:
        blob = pickle.load(f)
    return _from_blob(blob, weights, str(meta_pickle), scaler=blob.get("scaler"))


def export_sidecar(checkpoint_dir, model_name="dusk_crayfish"):
    """
    Write the JSON sidecar next to an existing pickle, and return its path.

    Run this once per checkpoint before shipping it anywhere that should not be
    unpickling our files or matching our sklearn version.
    """
    ckpt = load_checkpoint(checkpoint_dir, model_name)
    _, _, sidecar = _paths(checkpoint_dir, model_name)
    sidecar.write_text(json.dumps(ckpt.as_dict(), indent=2, sort_keys=True))
    return sidecar
