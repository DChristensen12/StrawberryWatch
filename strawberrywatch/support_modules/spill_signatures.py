"""
The pollutant signature table.

Every cell carries the mechanism that justifies its direction. The mechanism
sits on the row rather than in a header here because a direction without its
reason is not reviewable: the next person to argue with a cell needs to see
what claim they are arguing with.
"""

from __future__ import annotations

UP = "up"
DOWN = "down"
FLAT = "flat"
INDET = "indeterminate"

NORMAL = "normal"
MAJOR = "major"

# Channels that separate one pollutant from another. Conductivity and
# temperature alone collapse most types together.
DISCRIMINATING = ("dissolved_oxygen", "ph", "floating_conductivity")

NEW_TYPE_SCORE_FLOOR = 0.5
MIN_DISCRIMINATING_FOR_DIAGNOSIS = 1

# Berkeley tap water. Encoded as numbers rather than
# directions because "down" is weaker evidence than "down to about 13C" when the
# creek is already at 13.
TAPWATER_TEMP_C = 13.0
TAPWATER_PH = 9.4

# How close an event mean has to sit to the constants above to count as a match.
TAPWATER_TEMP_TOLERANCE_C = 1.5
TAPWATER_PH_TOLERANCE = 0.5

# Each cell is (direction, magnitude, mechanism).
SIGNATURES = {
    "rain": {
        "temperature": (INDET, NORMAL, "depends on the rain and the air temperature"),
        "dissolved_oxygen": (UP, NORMAL, "turbulence raises dissolved oxygen"),
        "ph": (DOWN, NORMAL, "rain is slightly acidic"),
        "conductivity": (DOWN, NORMAL, "increased volume dilutes solutes"),
        "floating_conductivity": (DOWN, NORMAL, "increased volume dilutes solutes"),
    },
    "tapwater": {
        "temperature": (DOWN, NORMAL, "Berkeley tap water is about 13 degrees Celsius"),
        "dissolved_oxygen": (
            DOWN,
            NORMAL,
            "tap water holds less dissolved oxygen than stream water, and carries "
            "chloramine, which is toxic to aquatic life",
        ),
        "ph": (UP, NORMAL, "Berkeley tap water is about pH 9.4"),
        "conductivity": (DOWN, NORMAL, "increased volume dilutes solutes"),
        "floating_conductivity": (DOWN, NORMAL, "increased volume dilutes solutes"),
    },
    "oil": {
        "temperature": (
            UP,
            NORMAL,
            "oil reduces evaporative cooling and lowers the creek's albedo",
        ),
        "dissolved_oxygen": (
            DOWN,
            MAJOR,
            "blocks gas exchange at the surface, kills plants, blocks photosynthesis "
            "and raises decomposer bacteria",
        ),
        "ph": (DOWN, NORMAL, "decomposition raises carbon dioxide, which lowers pH"),
        "conductivity": (DOWN, NORMAL, "oil is a poor conductor"),
        "floating_conductivity": (
            DOWN,
            MAJOR,
            "oil floats, so the surface falls further than the water column",
        ),
    },
    "sewage": {
        "temperature": (UP, NORMAL, "raw and treated sewage are warmer than creek water"),
        "dissolved_oxygen": (
            DOWN,
            MAJOR,
            "decomposing organic matter and decomposer bacteria consume oxygen",
        ),
        "ph": (
            INDET,
            NORMAL,
            "cleaners, industrial soap waste and detergents raise pH, often to between "
            "7.7 and 9.8, though high ammonia can lower it instead",
        ),
        "conductivity": (UP, NORMAL, "chlorides, phosphates and nitrates raise conductivity"),
        "floating_conductivity": (
            UP,
            NORMAL,
            "chlorides, phosphates and nitrates raise conductivity",
        ),
    },
    "fertilizer": {
        "temperature": (
            FLAT,
            NORMAL,
            "no significant effect; temperature depends on the water carrying it in",
        ),
        "dissolved_oxygen": (
            DOWN,
            NORMAL,
            "eutrophication raises dissolved oxygen briefly through algal blooms, then "
            "crashes it when the algae die and bacteria decompose them",
        ),
        "ph": (
            INDET,
            NORMAL,
            "nitrogen and phosphorus drive algae that consume dissolved carbon dioxide, "
            "raising pH; ammoniacal nitrogen or urea in initial runoff can lower it",
        ),
        "conductivity": (UP, NORMAL, "chlorides, phosphates and nitrates raise conductivity"),
        "floating_conductivity": (
            UP,
            NORMAL,
            "chlorides, phosphates and nitrates raise conductivity",
        ),
    },
}

PARAMETERS = [
    "temperature",
    "dissolved_oxygen",
    "ph",
    "conductivity",
    "floating_conductivity",
]

# Labelled events that contradict a row. Kept beside the table so nobody
# rediscovers them, and kept as exceptions so the row itself stays as sourced.
#
# The March 2026 hydrant break is the one that matters. A fire hydrant on Euclid
# Avenue ran down the street into a storm drain and into the North Fork, and
# north_fork_0 saw conductivity down and a depth spike with no meaningful
# temperature change. Tapwater says temperature down. Either street runoff warms
# on the way in, or the temperature row is wrong for delivery through a drain.
# That is a question for the group, not a reason to edit the row.
KNOWN_EXCEPTIONS = [
    {
        "event": "2026-03-20 Euclid Avenue fire hydrant break",
        "site": "north_fork_0",
        "row": "tapwater",
        "parameter": "temperature",
        "expected": DOWN,
        "observed": FLAT,
        "note": (
            "conductivity fell and depth spiked as the tapwater row predicts, but "
            "temperature did not move; the water reached the creek through a storm "
            "drain rather than directly"
        ),
    },
]

FEATURE_ALIASES = {
    "temperature": "temperature",
    "conductivity": "conductivity",
    "dissolved_oxygen": "dissolved_oxygen",
    "AtlasSci_DO": "dissolved_oxygen",
    "ph": "ph",
    "pH": "ph",
    "AtlasSci_pH": "ph",
    "floating_conductivity": "floating_conductivity",
    "AtlasSci_FloatCond": "floating_conductivity",
    "depth": "depth",
}

CHANGE_THRESHOLD_STD = 1.0


def resolve_parameter(column_name):
    """Map a dataframe column to a signature parameter name, or None."""
    return FEATURE_ALIASES.get(column_name)


def cell(pollutant, parameter, table=None):
    """One (direction, magnitude, mechanism) cell, defaulting to indeterminate."""
    signature = (SIGNATURES if table is None else table)[pollutant]
    return signature.get(parameter, (INDET, NORMAL, "not diagnostic for this type"))


def direction(pollutant, parameter, table=None):
    return cell(pollutant, parameter, table)[0]


def mechanism(pollutant, parameter, table=None):
    """Why the table claims that direction. The sourcing, attached to the row."""
    return cell(pollutant, parameter, table)[2]


def required_parameters(pollutant, table=None):
    """
    Parameters this row asserts a direction for, in PARAMETERS order.

    An INDET cell is not required. The row says the parameter does not
    discriminate, so its absence costs the row nothing.
    """
    return [param for param in PARAMETERS if direction(pollutant, param, table) != INDET]


def missing_parameters(pollutant, observed, table=None):
    """Which of a row's required parameters have no observed direction."""
    return [p for p in required_parameters(pollutant, table) if p not in observed]


def evaluable_on(available, table=None):
    """
    Which rows can be judged given a set of available parameters.

    Scoring a row on the parameters that happen to be there would report a
    partial signature as though it were the whole one.
    """
    have = set(available)
    rows = SIGNATURES if table is None else table
    return [p for p in rows if not set(required_parameters(p, rows)) - have]


def looks_like_tapwater_temperature(event_mean_c):
    """Whether an event mean sits at Berkeley tap water temperature."""
    if event_mean_c is None:
        return None
    return abs(float(event_mean_c) - TAPWATER_TEMP_C) <= TAPWATER_TEMP_TOLERANCE_C


def looks_like_tapwater_ph(event_mean_ph):
    """Whether an event mean sits at Berkeley tap water pH."""
    if event_mean_ph is None:
        return None
    return abs(float(event_mean_ph) - TAPWATER_PH) <= TAPWATER_PH_TOLERANCE


__all__ = [
    "CHANGE_THRESHOLD_STD",
    "DISCRIMINATING",
    "DOWN",
    "FEATURE_ALIASES",
    "FLAT",
    "INDET",
    "KNOWN_EXCEPTIONS",
    "MAJOR",
    "MIN_DISCRIMINATING_FOR_DIAGNOSIS",
    "NEW_TYPE_SCORE_FLOOR",
    "NORMAL",
    "PARAMETERS",
    "SIGNATURES",
    "TAPWATER_PH",
    "TAPWATER_PH_TOLERANCE",
    "TAPWATER_TEMP_C",
    "TAPWATER_TEMP_TOLERANCE_C",
    "UP",
    "cell",
    "direction",
    "evaluable_on",
    "looks_like_tapwater_ph",
    "looks_like_tapwater_temperature",
    "mechanism",
    "missing_parameters",
    "required_parameters",
    "resolve_parameter",
]
