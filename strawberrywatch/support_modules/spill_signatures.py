"""The pollutant signature table, its direction codes, and what each row needs."""

from __future__ import annotations

# Direction codes for the signature table and observed changes.
# UP/DOWN/FLAT are as expected. INDET means the parameter doesn't discriminate for
# this pollutant, so it's skipped during matching (no reward, no penalty).
UP = "up"
DOWN = "down"
FLAT = "flat"
INDET = "indeterminate"

# Magnitude tag. A few signature cells are marked MAJOR (e.g. oil crashing DO).
# The current matcher treats MAJOR the same as NORMAL; the tag is there for a
# future version that weights strong moves more heavily.
NORMAL = "normal"
MAJOR = "major"

# The channels that actually separate pollutant types. Conductivity and temp alone
# collapse most types together (rain/tapwater/oil all drop conductivity; sewage/fertilizer
# raise it), so without at least one of these the classifier won't name a type.
DISCRIMINATING = ("dissolved_oxygen", "ph", "floating_conductivity")

# If the best pollutant match scores below this fraction of agreeing parameters,
# it is a poor fit, reported as possible_new_type rather than as a named type.
NEW_TYPE_SCORE_FLOOR = 0.5

# Minimum discriminating channels needed to commit to a named pollutant.
# Below this it is a cond+temp guess, so the answer is cannot_evaluate instead.
MIN_DISCRIMINATING_FOR_DIAGNOSIS = 1


# Signature table from the Water Quality team's Table 1a, the one place to edit
# if it changes. Each pollutant maps a parameter to (direction, magnitude).
#
# Depth is left out: it tells you how the spill arrived, not which pollutant.
# Add "depth" to PARAMETERS to use it as a delivery hint.
SIGNATURES = {
    "rain": {
        "temperature": (INDET, NORMAL),  # depends on rain and air temp
        "dissolved_oxygen": (UP, NORMAL),  # increased turbulence
        "ph": (DOWN, NORMAL),  # rain is slightly acidic
        "conductivity": (DOWN, NORMAL),  # more volume dilutes solutes
        "floating_conductivity": (DOWN, NORMAL),  # more volume dilutes solutes
        # "depth":               (UP,    NORMAL),   # rain adds volume
    },
    "tapwater": {
        "temperature": (DOWN, NORMAL),  # Berkeley tap is ~13C, cooler
        "dissolved_oxygen": (DOWN, NORMAL),  # tap has less DO, has chloramine
        "ph": (UP, NORMAL),  # Berkeley tap is ~9.4
        "conductivity": (DOWN, NORMAL),  # more volume dilutes solutes
        "floating_conductivity": (DOWN, NORMAL),  # more volume dilutes solutes
        # "depth":               (UP,    NORMAL),   # adds volume, but may be small
    },
    "oil": {
        "temperature": (UP, NORMAL),  # reduces evaporative cooling
        "dissolved_oxygen": (DOWN, MAJOR),  # blocks gas exchange, kills plants
        "ph": (DOWN, NORMAL),  # CO2 from decomposition
        "conductivity": (DOWN, NORMAL),  # oil is a poor conductor
        "floating_conductivity": (DOWN, MAJOR),  # floats, hits surface hardest
        # "depth":               (INDET, NORMAL),   # delivery dependent
    },
    "sewage": {
        "temperature": (UP, NORMAL),  # sewage warmer than creek
        "dissolved_oxygen": (DOWN, MAJOR),  # decomposer bacteria consume DO
        "ph": (INDET, NORMAL),  # cleaners raise it, ammonia lowers it
        "conductivity": (UP, NORMAL),  # chlorides, phosphates, nitrates
        "floating_conductivity": (UP, NORMAL),  # chlorides, phosphates, nitrates
        # "depth":               (UP,    NORMAL),   # adds volume, often point source
    },
    "fertilizer": {
        "temperature": (FLAT, NORMAL),  # fertilizer itself does not move temp
        "dissolved_oxygen": (DOWN, NORMAL),  # algae die-off, bacteria consume DO
        "ph": (INDET, NORMAL),  # algae raise it, ammoniacal runoff lowers it
        "conductivity": (UP, NORMAL),  # chlorides, phosphates, nitrates
        "floating_conductivity": (UP, NORMAL),  # chlorides, phosphates, nitrates
        # "depth":               (UP,    NORMAL),   # arrives as runoff
    },
}

PARAMETERS = [
    "temperature",
    "dissolved_oxygen",
    "ph",
    "conductivity",
    "floating_conductivity",
    # "depth",
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

# How far the event mean has to move, in baseline standard deviations, to count.
CHANGE_THRESHOLD_STD = 1.0


def resolve_parameter(column_name):
    """Map a dataframe column to a signature parameter name, or None."""
    return FEATURE_ALIASES.get(column_name)


def required_parameters(pollutant, table=None):
    """
    The parameters a row asserts a direction for, in PARAMETERS order.

    INDET cells are not required: the row says the parameter does not
    discriminate, so its absence costs the row nothing.
    """
    signature = (SIGNATURES if table is None else table)[pollutant]
    required = []
    for param in PARAMETERS:
        direction, _magnitude = signature.get(param, (INDET, NORMAL))
        if direction != INDET:
            required.append(param)
    return required


def missing_parameters(pollutant, observed, table=None):
    """Which of a row's required parameters have no observed direction."""
    return [p for p in required_parameters(pollutant, table) if p not in observed]


def evaluable_on(available, table=None):
    """
    Which rows can be judged at all given a set of available parameters.

    A row needing pH cannot be evaluated without pH. Answering on the rest
    would score a partial signature as if it were the whole one.
    """
    have = set(available)
    rows = SIGNATURES if table is None else table
    return [p for p in rows if not set(required_parameters(p, rows)) - have]


__all__ = [
    "CHANGE_THRESHOLD_STD",
    "DISCRIMINATING",
    "DOWN",
    "FEATURE_ALIASES",
    "FLAT",
    "INDET",
    "MAJOR",
    "MIN_DISCRIMINATING_FOR_DIAGNOSIS",
    "NEW_TYPE_SCORE_FLOOR",
    "NORMAL",
    "PARAMETERS",
    "SIGNATURES",
    "UP",
    "evaluable_on",
    "missing_parameters",
    "required_parameters",
    "resolve_parameter",
]
