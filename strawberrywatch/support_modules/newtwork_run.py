"""
Newtwork Run: says whether a firing travelled, from the order nodes fired in.

Ordering alone. Upstream before downstream is consistent with transport, and
everything at once is consistent with a catchment wide cause like rain. That
needs no travel time measurement, which is the point: it works on the data that
exists today. An explainer, so it cannot change what fired.
"""

from __future__ import annotations

import numpy as np

from strawberrywatch import inventory as inv
from strawberrywatch.support_modules.base import SupportExplainer

TRANSPORT = "consistent_with_transport"
CATCHMENT_WIDE = "catchment_wide"
LOCAL = "local_to_this_site"
NOT_TRANSPORT = "not_transport"
CANNOT_EVALUATE = "cannot_evaluate"
NOTHING_TO_EXPLAIN = "nothing_to_explain"

# Per edge travel times, for when somebody measures them. Shape:
#
#   TRAVEL_TIMES[("south_fork_1", "south_fork_2")] = {
#       "low_flow":  {"minutes_to_peak": 45, "tolerance_minutes": 20},
#       "high_flow": {"minutes_to_peak": 15, "tolerance_minutes": 10},
#   }
#
# Empty on purpose. Nothing in this repo has measured a travel time and a
# plausible looking guess would turn into a number somebody trusts. Until it is
# filled, ordering carries the whole judgement, and travel_window returns None.
TRAVEL_TIMES = {}

DEFAULT_FLOW_CONDITION = "unknown"


def travel_window(upstream, downstream, flow_condition=DEFAULT_FLOW_CONDITION):
    """Measured travel time for one edge, or None where nobody has measured it."""
    edge = TRAVEL_TIMES.get((upstream, downstream))
    if not edge:
        return None
    return edge.get(flow_condition)


def first_firing(fired_column):
    """Index of the first timestep a node fired, or None."""
    hits = np.flatnonzero(np.asarray(fired_column, dtype=bool))
    return int(hits[0]) if len(hits) else None


def order_verdict(site, first, inventory, control_sites):
    """
    Read one node's firing order against its flow neighbours.

    first is {site: first firing index} over the sites that fired at all.
    """
    mine = first.get(site)
    if mine is None:
        return None

    upstream = [s for s in inventory.upstream_chain(site) if s in first]
    downstream = [s for s in inventory.downstream_chain(site) if s in first]

    earlier_upstream = [s for s in upstream if first[s] < mine]
    later_downstream = [s for s in downstream if first[s] > mine]
    simultaneous = [s for s, i in first.items() if i == mine and s != site]

    control_hits = [s for s in control_sites if s != site and first.get(s) == mine]

    if control_hits:
        return {
            "verdict": NOT_TRANSPORT,
            "reason": (
                f"{', '.join(control_hits)} fired at the same timestep, and it is "
                f"flow disconnected from Strawberry Creek, so nothing carried this "
                f"between them"
            ),
            "upstream_before": earlier_upstream,
            "downstream_after": later_downstream,
            "simultaneous": simultaneous,
            "control": control_hits,
        }

    if earlier_upstream or later_downstream:
        parts = []
        if earlier_upstream:
            parts.append(f"{', '.join(earlier_upstream)} fired first upstream")
        if later_downstream:
            parts.append(f"{', '.join(later_downstream)} fired later downstream")
        return {
            "verdict": TRANSPORT,
            "reason": "; ".join(parts),
            "upstream_before": earlier_upstream,
            "downstream_after": later_downstream,
            "simultaneous": simultaneous,
            "control": [],
        }

    if simultaneous:
        return {
            "verdict": CATCHMENT_WIDE,
            "reason": (
                f"fired at the same timestep as {', '.join(sorted(simultaneous))}, with "
                f"no upstream lead, which is what a catchment wide cause looks like"
            ),
            "upstream_before": [],
            "downstream_after": [],
            "simultaneous": simultaneous,
            "control": [],
        }

    return {
        "verdict": LOCAL,
        "reason": "no other site fired, so nothing suggests this moved through the creek",
        "upstream_before": [],
        "downstream_after": [],
        "simultaneous": [],
        "control": [],
    }


class NewtworkRun(SupportExplainer):
    """Says whether a firing travelled, from the order the nodes fired in"""

    name = "newtwork_run"

    def __init__(self, inventory=None):
        self.inventory = inventory or inv.load()

    def describe(self):
        return (
            "newtwork_run: reads firing order along the flow graph to separate "
            "transport from a catchment wide cause (explainer, no budget)"
        )

    def explain(self, window, nodes, fired):
        """
        One account per node, from the order every node fired in.

        window is unread. Ordering comes off the fired grid and the flow graph,
        so this needs no readings at all, which is why it works today.
        """
        fired = np.asarray(fired, dtype=bool)
        if fired.ndim != 2 or fired.shape[1] != len(nodes):
            raise ValueError(f"newtwork_run got a {fired.shape} grid for {len(nodes)} nodes")

        first = {}
        for j, node in enumerate(nodes):
            index = first_firing(fired[:, j])
            if index is not None:
                first[node] = index

        known = [n for n in nodes if n in self.inventory.sites]
        control_sites = [n for n in known if n in self.inventory.isolated()]

        out = []
        for node in nodes:
            out.append(self._account(node, first, control_sites))
        return out

    def _account(self, node, first, control_sites):
        if node not in first:
            return {
                "verdict": NOTHING_TO_EXPLAIN,
                "cause": None,
                "confidence": "not applicable",
                "evidence": {},
                "explanation": "nothing fired at this node, so there is nothing to explain",
                "note": "",
            }

        if node not in self.inventory.sites:
            return {
                "verdict": CANNOT_EVALUATE,
                "cause": None,
                "confidence": "none (site not in the inventory)",
                "evidence": {},
                "explanation": f"cannot evaluate: {node} is not a site the inventory describes",
                "note": "add the site to inventory.yaml before reading its neighbours",
            }

        result = order_verdict(node, first, self.inventory, control_sites)
        travel = {(up, node): travel_window(up, node) for up in self.inventory.upstream_of(node)}
        measured = {edge: w for edge, w in travel.items() if w is not None}

        return {
            "verdict": result["verdict"],
            "cause": None,
            "confidence": self._confidence(measured),
            "evidence": {
                "first_firing_index": first[node],
                "upstream_before": result["upstream_before"],
                "downstream_after": result["downstream_after"],
                "simultaneous": result["simultaneous"],
                "control_fired_with_it": result["control"],
            },
            "explanation": f"{result['verdict']}: {result['reason']}",
            "note": result["reason"],
        }

    def _confidence(self, measured):
        """Ordering alone is ordering alone. Say so rather than implying more."""
        if not measured:
            return "ordering only (no measured travel times for these edges)"
        return f"ordering plus {len(measured)} measured travel times"


__all__ = [
    "CANNOT_EVALUATE",
    "CATCHMENT_WIDE",
    "LOCAL",
    "NOTHING_TO_EXPLAIN",
    "NOT_TRANSPORT",
    "TRANSPORT",
    "TRAVEL_TIMES",
    "NewtworkRun",
    "first_firing",
    "order_verdict",
    "travel_window",
]
