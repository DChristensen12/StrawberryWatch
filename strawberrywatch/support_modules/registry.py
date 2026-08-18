"""Which support modules exist, and the only way to get one by name."""

from __future__ import annotations

from strawberrywatch.support_modules.base import KINDS, SupportError, SupportModule
from strawberrywatch.support_modules.newtwork_run import NewtworkRun
from strawberrywatch.support_modules.settling_pool import SettlingPool
from strawberrywatch.support_modules.trial_bed import TrialBed

SUPPORT_REGISTRY = {
    "trial_bed": TrialBed,
    "newtwork_run": NewtworkRun,
    "settling_pool": SettlingPool,
}


class UnknownSupportModule(SupportError):
    """A name that is not in SUPPORT_REGISTRY."""


class SupportCollision(SupportError):
    """A module duplicates something the model already does internally."""


def available():
    """The valid --support names, sorted, for help text and error messages."""
    return sorted(SUPPORT_REGISTRY)


def support_class(name):
    """
    The class registered under one name, or raise listing the valid ones.

    Raising is the point. A name that silently resolved to nothing would give a
    run that took --support trail_bed, attached nothing, scored clean and
    exited zero with nobody the wiser.
    """
    try:
        return SUPPORT_REGISTRY[name]
    except KeyError:
        raise UnknownSupportModule(
            f"unknown support module {name!r}; valid options: {', '.join(available())}"
        ) from None


def builtin_support(model):
    """
    What a model says it already handles internally.

    Read off the class, so a model that grows an internal behaviour declares it
    in one place. A model with no attribute declares nothing, which has to be
    the default or no module could attach to anything until every model had
    one.
    """
    return tuple(getattr(model, "BUILTIN_SUPPORT", ()))


def check_collision(module, model):
    """
    Raise if the model already does internally what this module would do.

    The double-application guard. Dusk Crayfish applies a 2.0 rain multiplier
    inside anomaly_detector.py; a rain modulator on top would multiply the
    threshold twice, and nothing in the arithmetic would complain because both
    applications are individually correct. What comes out is a 4x bar during
    rain and a detector that has stopped alerting.

    In the loader rather than SupportModule.validate, because it applies to
    every module identically and a subclass overriding validate would drop it.
    """
    declared = builtin_support(model)
    if module.name in declared:
        model_name = model.__name__ if isinstance(model, type) else type(model).__name__
        raise SupportCollision(
            f"{model_name} already handles {module.name!r} internally "
            f"(BUILTIN_SUPPORT = {declared}); attaching the module as well would "
            f"apply it twice. Remove it from --support, or take the behaviour out "
            f"of the model."
        )


def load(names, model):
    """
    Instantiate the named modules and check every one of them against a model.

    names comes in however the operator typed it and goes out sorted, so
    command line ordering cannot reach anything downstream. Duplicates raise:
    --support trial_bed trial_bed is a typo, and it would spend the module's
    budget twice.

    Everything that can fail fails here, before the model runs. A module that
    cannot attach must not turn up halfway through a detection.
    """
    seen = []
    for name in names:
        if name in seen:
            raise SupportError(f"support module {name!r} given more than once")
        seen.append(name)

    modules = []
    for name in sorted(seen):
        module = support_class(name)()
        if module.kind not in KINDS:
            raise SupportError(f"{name} registered with unusable kind {module.kind!r}")
        if not isinstance(module, SupportModule):
            raise SupportError(f"{name} is not a SupportModule")
        module.validate(model)
        check_collision(module, model)
        modules.append(module)
    return modules


__all__ = [
    "SUPPORT_REGISTRY",
    "SupportCollision",
    "UnknownSupportModule",
    "available",
    "builtin_support",
    "check_collision",
    "load",
    "support_class",
]
