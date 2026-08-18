"""Newtwork Run: asks whether a flagged node's neighbours agree. Placeholder."""

from __future__ import annotations

from strawberrywatch.support_modules.base import SupportDetector


class NewtworkRun(SupportDetector):
    """Scores a flagged node against what its flow neighbours did"""

    name = "newtwork_run"

    def describe(self):
        return "newtwork_run: scores a flagged node against its flow neighbours (detector, unimplemented)"

    def score(self, window, nodes):
        raise NotImplementedError(
            "Newtwork Run will score a flagged node on whether the event "
            "travelled: a real spill shows up downstream after the travel lag "
            "and a failing sensor does not, so agreement along the flow edges "
            "separates the two. It runs after a detection and spends its share "
            "of the false alarm budget. Not implemented yet."
        )

    def null(self):
        raise NotImplementedError(
            "Newtwork Run's null will be fitted over fault-free windows at "
            "calibration time and loaded from the artifact beside the weights. "
            "Not implemented yet."
        )
