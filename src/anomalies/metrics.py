import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


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
_DISCRIMINATING = ("dissolved_oxygen", "ph", "floating_conductivity")

# If the best pollutant match scores below this fraction of agreeing parameters,
# it's a poor fit and we call possible_new_type instead of naming a pollutant.
_NEW_TYPE_SCORE_FLOOR = 0.5

# Minimum discriminating channels needed to commit to a named pollutant.
# Below this it's just a cond+temp guess, so we report undetermined instead.
_MIN_DISCRIMINATING_FOR_DIAGNOSIS = 1


# Signature table from the Water Quality team's Table 1a.
# Each pollutant maps each parameter to (direction, magnitude).
# This is the one place to edit if the table changes.
#
# Depth is deliberately left out. It tells you HOW the spill was delivered
# but not WHICH pollutant. Uncomment depth entries and add "depth" to
# _PARAMETERS if you want it as a delivery hint.
_SIGNATURES = {
    "rain": {
        "temperature":           (INDET, NORMAL),   # depends on rain and air temp
        "dissolved_oxygen":      (UP,    NORMAL),   # increased turbulence
        "ph":                    (DOWN,  NORMAL),   # rain is slightly acidic
        "conductivity":          (DOWN,  NORMAL),   # more volume dilutes solutes
        "floating_conductivity": (DOWN,  NORMAL),   # more volume dilutes solutes
        # "depth":               (UP,    NORMAL),   # rain adds volume
    },
    "tapwater": {
        "temperature":           (DOWN,  NORMAL),   # Berkeley tap is ~13C, cooler
        "dissolved_oxygen":      (DOWN,  NORMAL),   # tap has less DO, has chloramine
        "ph":                    (UP,    NORMAL),   # Berkeley tap is ~9.4
        "conductivity":          (DOWN,  NORMAL),   # more volume dilutes solutes
        "floating_conductivity": (DOWN,  NORMAL),   # more volume dilutes solutes
        # "depth":               (UP,    NORMAL),   # adds volume, but may be small
    },
    "oil": {
        "temperature":           (UP,    NORMAL),   # reduces evaporative cooling
        "dissolved_oxygen":      (DOWN,  MAJOR),    # blocks gas exchange, kills plants
        "ph":                    (DOWN,  NORMAL),   # CO2 from decomposition
        "conductivity":          (DOWN,  NORMAL),   # oil is a poor conductor
        "floating_conductivity": (DOWN,  MAJOR),    # floats, hits surface hardest
        # "depth":               (INDET, NORMAL),   # delivery dependent
    },
    "sewage": {
        "temperature":           (UP,    NORMAL),   # sewage warmer than creek
        "dissolved_oxygen":      (DOWN,  MAJOR),    # decomposer bacteria consume DO
        "ph":                    (INDET, NORMAL),   # cleaners raise it, ammonia lowers it
        "conductivity":          (UP,    NORMAL),   # chlorides, phosphates, nitrates
        "floating_conductivity": (UP,    NORMAL),   # chlorides, phosphates, nitrates
        # "depth":               (UP,    NORMAL),   # adds volume, often point source
    },
    "fertilizer": {
        "temperature":           (FLAT,  NORMAL),   # fertilizer itself does not move temp
        "dissolved_oxygen":      (DOWN,  NORMAL),   # algae die-off, bacteria consume DO
        "ph":                    (INDET, NORMAL),   # algae raise it, ammoniacal runoff lowers it
        "conductivity":          (UP,    NORMAL),   # chlorides, phosphates, nitrates
        "floating_conductivity": (UP,    NORMAL),   # chlorides, phosphates, nitrates
        # "depth":               (UP,    NORMAL),   # arrives as runoff
    },
}

_PARAMETERS = [
    "temperature",
    "dissolved_oxygen",
    "ph",
    "conductivity",
    "floating_conductivity",
    # "depth",
]

_FEATURE_ALIASES = {
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

_CHANGE_THRESHOLD_STD = 1.0


def _observed_direction(baseline_vals, event_vals):
    """
    Determines whether a parameter went UP, DOWN, or FLAT from baseline to event.

    Uses the baseline's own variability as the yardstick: the event mean has to move
    more than _CHANGE_THRESHOLD_STD standard deviations to count as a real move. This
    adapts to each parameter's natural noise rather than hardcoding one threshold for
    wildly different scales like temperature and conductivity.

    Returns UP, DOWN, FLAT, or None if there's not enough data.
    """
    b = pd.to_numeric(pd.Series(baseline_vals), errors="coerce").dropna()
    e = pd.to_numeric(pd.Series(event_vals), errors="coerce").dropna()
    if len(b) < 2 or len(e) < 1:
        return None

    baseline_mean = b.mean()
    baseline_std = b.std()
    event_mean = e.mean()

    if pd.isna(baseline_std) or baseline_std == 0:
        if baseline_mean == 0:
            return FLAT if event_mean == 0 else (UP if event_mean > 0 else DOWN)
        rel = (event_mean - baseline_mean) / abs(baseline_mean)
        if abs(rel) < 0.05:
            return FLAT
        return UP if rel > 0 else DOWN

    shift = (event_mean - baseline_mean) / baseline_std
    if abs(shift) < _CHANGE_THRESHOLD_STD:
        return FLAT
    return UP if shift > 0 else DOWN


def _resolve_parameter(column_name):
    """Map a dataframe column to a signature parameter name, or None."""
    return _FEATURE_ALIASES.get(column_name)


def classify_event(baseline_df, event_df):
    """
    Classifies a detected anomaly by matching observed parameter changes against
    the pollutant signature table. Returns one of three verdicts:

    "diagnosed": a named pollutant matched well enough to commit.
    "undetermined": not enough discriminating sensors (DO, pH, floating conductivity)
        to separate types. We decline to guess and give a hint instead.
    "possible_new_type": data is good enough to judge but fits no known signature,
        or the top score is a tie. Worth surfacing to a human.

    Works with whatever sensor columns are present; gets sharper as more come online.
    Returns a dict with verdict, named_type, ranked matches, and diagnostic context.
    """
    observed = {}
    for col in event_df.columns:
        param = _resolve_parameter(col)
        if param is None or param not in _PARAMETERS:
            continue
        if col not in baseline_df.columns:
            continue
        direction = _observed_direction(baseline_df[col].values, event_df[col].values)
        if direction is not None:
            observed[param] = direction

    available_params = sorted(observed.keys())
    discriminating_available = [p for p in available_params if p in _DISCRIMINATING]

    results = []
    for pollutant, signature in _SIGNATURES.items():
        comparable = 0
        agreements = 0
        per_param = {}
        for param in _PARAMETERS:
            sig_dir, _sig_mag = signature.get(param, (INDET, NORMAL))
            if sig_dir == INDET:
                per_param[param] = "skipped (not diagnostic)"
                continue
            if param not in observed:
                per_param[param] = "skipped (no data)"
                continue
            comparable += 1
            if observed[param] == sig_dir:
                agreements += 1
                per_param[param] = f"match ({observed[param]})"
            else:
                per_param[param] = f"differ (saw {observed[param]}, expected {sig_dir})"

        score = (agreements / comparable) if comparable > 0 else 0.0
        results.append({
            "pollutant": pollutant,
            "score": score,
            "agreements": agreements,
            "comparable": comparable,
            "per_param": per_param,
        })

    results.sort(key=lambda r: (r["score"], r["comparable"]), reverse=True)

    top_score = results[0]["score"]
    tied = [r["pollutant"] for r in results if r["score"] == top_score and top_score > 0]

    if len(discriminating_available) < _MIN_DISCRIMINATING_FOR_DIAGNOSIS:
        # Not enough channels to honestly name a type. Decline to diagnose.
        verdict = "undetermined"
        named_type = None
        verdict_note = (
            "Cannot name a pollutant type. None of the discriminating channels "
            "(dissolved oxygen, pH, floating conductivity) are populated, and "
            "conductivity with temperature alone cannot separate the candidates. "
            "Leading candidate below is a hint only, not a diagnosis."
        )
    elif top_score < _NEW_TYPE_SCORE_FLOOR:
        # Enough data to judge, but nothing fits. This is the case to surface.
        verdict = "possible_new_type"
        named_type = None
        verdict_note = (
            f"Does not match any known spill signature well (best score "
            f"{top_score:.2f}, below {_NEW_TYPE_SCORE_FLOOR:.2f}). This may be a "
            f"new or unclassified type of event and is worth a closer look."
        )
    elif len(tied) > 1:
        # Several pollutants fit equally and the data did not separate them.
        verdict = "possible_new_type"
        named_type = None
        verdict_note = (
            f"Top match is an unresolved tie between {', '.join(tied)} (all at "
            f"score {top_score:.2f}). The available channels did not separate "
            f"them, so no single type can be named; may also be a new type."
        )
    else:
        verdict = "diagnosed"
        named_type = results[0]["pollutant"]
        verdict_note = (
            f"Best match is {named_type} (score {top_score:.2f}, "
            f"{results[0]['agreements']}/{results[0]['comparable']} parameters agreed)."
        )

    if len(discriminating_available) == 0:
        confidence = "none (no discriminating channels populated)"
    elif len(discriminating_available) == 1:
        confidence = f"moderate (one discriminating channel: {discriminating_available[0]})"
    else:
        confidence = f"good ({len(discriminating_available)} discriminating channels)"

    return {
        "verdict": verdict,
        "named_type": named_type,
        "verdict_note": verdict_note,
        "ranked": results,
        "available_parameters": available_params,
        "discriminating_available": discriminating_available,
        "observed_directions": observed,
        "confidence": confidence,
        "top_candidates": tied,
    }


def format_classification(result):
    """Turn a classify_event result into a readable report."""
    lines = []
    lines.append("--- Spill Type Classification ---")
    lines.append(f"Verdict: {result['verdict'].upper().replace('_', ' ')}")
    if result["named_type"]:
        lines.append(f"Diagnosed type: {result['named_type']}")
    lines.append(result["verdict_note"])
    lines.append("")
    lines.append(f"Parameters available: {result['available_parameters'] or 'none'}")
    lines.append("Observed changes: " + (
        ", ".join(f"{p}={d}" for p, d in result["observed_directions"].items())
        if result["observed_directions"] else "none"
    ))
    lines.append(f"Channel confidence: {result['confidence']}")
    lines.append("")
    lines.append("Ranked matches:")
    for r in result["ranked"]:
        if r["comparable"] == 0:
            lines.append(f"  {r['pollutant']:12s} no comparable parameters")
            continue
        lines.append(
            f"  {r['pollutant']:12s} score {r['score']:.2f} "
            f"({r['agreements']}/{r['comparable']} parameters agreed)"
        )
    lines.append("")
    lines.append("Detail for leading candidate:")
    top = result["ranked"][0]
    for param, verdict in top["per_param"].items():
        lines.append(f"  {param:22s} {verdict}")
    lines.append("---------------------------------")
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Case A: conductivity + temperature only, no discriminating channels. Should be UNDETERMINED.
    print("CASE A: conductivity up, temperature flat, no discriminating channels")
    baseline = pd.DataFrame({
        "conductivity": rng.normal(300, 5, 50),
        "temperature":  rng.normal(15, 0.3, 50),
    })
    event = pd.DataFrame({
        "conductivity": rng.normal(360, 5, 20),
        "temperature":  rng.normal(15.1, 0.3, 20),
    })
    print(format_classification(classify_event(baseline, event)))
    print()

    # Case B: clean sewage signature with DO present. Should DIAGNOSE sewage.
    print("CASE B: sewage signature with DO present")
    baseline = pd.DataFrame({
        "conductivity":     rng.normal(300, 5, 50),
        "temperature":      rng.normal(15, 0.3, 50),
        "dissolved_oxygen": rng.normal(8, 0.2, 50),
    })
    event = pd.DataFrame({
        "conductivity":     rng.normal(360, 5, 20),  # up
        "temperature":      rng.normal(16.5, 0.3, 20), # up
        "dissolved_oxygen": rng.normal(4, 0.2, 20),   # major down
    })
    print(format_classification(classify_event(baseline, event)))
    print()

    # Case C: contradictory pattern with a discriminating channel. Should be POSSIBLE NEW TYPE.
    print("CASE C: contradictory pattern with a discriminating channel")
    baseline = pd.DataFrame({
        "conductivity":     rng.normal(300, 5, 50),
        "temperature":      rng.normal(15, 0.3, 50),
        "dissolved_oxygen": rng.normal(8, 0.2, 50),
    })
    event = pd.DataFrame({
        "conductivity":     rng.normal(360, 5, 20),  # up
        "temperature":      rng.normal(16.5, 0.3, 20), # up
        "dissolved_oxygen": rng.normal(11, 0.2, 20),  # up, which no up-conductivity type expects
    })
    print(format_classification(classify_event(baseline, event)))
