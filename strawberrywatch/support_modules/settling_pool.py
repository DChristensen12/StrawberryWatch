"""Settling Pool: screens readings before the model consumes them. Placeholder."""

from __future__ import annotations

from strawberrywatch.support_modules.base import SupportScreen


class SettlingPool(SupportScreen):
    """
    Drops readings that are unfit for the model to consume

    A screen and not one of the other two kinds. A modulator moves the
    threshold and leaves the score alone, which keeps the null valid; this
    changes what the model reads, so it changes the score. A detector adds a
    test and pays budget; this adds no test. So it takes no budget, but it does
    move the distribution the primary was calibrated against, and implementing
    it means recalibrating the primary on screened windows.
    """

    name = "settling_pool"

    def describe(self):
        return "settling_pool: screens unfit readings before the model reads them (screen, unimplemented)"

    def admit(self, window, nodes):
        raise NotImplementedError(
            "Settling Pool will screen readings before the model sees them, "
            "withholding those a sensor produced while it was settling, "
            "purging or otherwise not measuring the creek, so the model is "
            "never asked to explain a reading that was never a measurement. It "
            "runs before the model and takes no false alarm budget, but it "
            "moves the distribution the primary score is drawn from, so it "
            "needs the primary recalibrated against screened windows. Not "
            "implemented yet."
        )
