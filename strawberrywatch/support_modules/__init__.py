"""Optional layers on a detection run: screens, modulators, detectors and explainers."""

from strawberrywatch.support_modules.base import (
    DEFAULT_Q_TOTAL,
    KINDS,
    SUPPORT_KEYS,
    BudgetError,
    SupportDetector,
    SupportError,
    SupportExplainer,
    SupportModulator,
    SupportModule,
    SupportScreen,
    SupportStack,
    allocate,
    summarise,
)
from strawberrywatch.support_modules.registry import (
    SUPPORT_REGISTRY,
    SupportCollision,
    UnknownSupportModule,
    available,
    load,
    support_class,
)

__all__ = [
    "DEFAULT_Q_TOTAL",
    "KINDS",
    "SUPPORT_KEYS",
    "SUPPORT_REGISTRY",
    "BudgetError",
    "SupportCollision",
    "SupportDetector",
    "SupportError",
    "SupportExplainer",
    "SupportModulator",
    "SupportModule",
    "SupportScreen",
    "SupportStack",
    "UnknownSupportModule",
    "allocate",
    "available",
    "load",
    "summarise",
    "support_class",
]
