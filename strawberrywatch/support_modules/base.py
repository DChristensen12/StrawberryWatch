"""Base classes, the budget split, the audit record and the stack."""

from __future__ import annotations

import numpy as np

# A module that does not care which contract it attaches to declares this.
ANY_CONTRACT = None

# Screens run before the model, modulators alongside it, detectors after it,
# explainers after the decision has been made. Reading the tuple in order is
# reading the pipeline in order.
KINDS = ("screen", "modulator", "detector", "explainer")

# Default total false alarm rate. Matches the operating_q the shipped
# calibration artifact carries.
DEFAULT_Q_TOTAL = 1e-4


class SupportError(RuntimeError):
    """A support module cannot attach, or was asked for something it is not."""


class BudgetError(ValueError):
    """A budget split that would leave the primary nothing, or is not a rate."""


# Base classes


class SupportModule:
    """
    The parts every kind shares.

    Subclass one of the four kinds below, not this. The registry, the budget
    split and the audit record all dispatch on `kind`, and a class that leaves
    it empty is refused at registration.

    This module doesn't import or name a model. A support module attaches to any
    model declaring the contract it needs, and that's how one module reaches
    both models by the same path.
    """

    # Short identifier used on the command line and as the audit record key.
    name: str = ""

    # Set by a kind, not by a concrete module
    kind: str = ""

    # The INPUT_CONTRACT a model must declare for this module to attach, or
    # ANY_CONTRACT. Compared as a plain string so this file needs no import
    # from the models package.
    requires_contract = ANY_CONTRACT

    def describe(self):
        """
        One line for the audit record.

        Falls back to the class docstring's first line so a module cannot ship
        with no description at all.
        """
        doc = (self.__class__.__doc__ or "").strip()
        first = doc.splitlines()[0].strip() if doc else ""
        return first or f"{self.name or self.__class__.__name__} ({self.kind})"

    def validate(self, model):
        """
        Raise if this module cannot attach to this model.

        Reads what the model declares about itself without asking which model
        it is. A subclass adding its own checks should call up to this first.

        The collision check against BUILTIN_SUPPORT lives in the loader instead,
        where a subclass overriding validate can't drop it.
        """
        if not self.name:
            raise SupportError(f"{type(self).__name__} has no name; it cannot be asked for by name")
        if self.kind not in KINDS:
            raise SupportError(
                f"{self.name} declares kind {self.kind!r}; expected one of {', '.join(KINDS)}"
            )
        if self.requires_contract is ANY_CONTRACT:
            return
        declared = getattr(model, "INPUT_CONTRACT", None)
        if declared != self.requires_contract:
            raise SupportError(
                f"{self.name} needs a model speaking {self.requires_contract!r}, but "
                f"{type(model).__name__ if not isinstance(model, type) else model.__name__} "
                f"declares {declared!r}"
            )

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r} kind={self.kind!r}>"


class SupportModulator(SupportModule):
    """
    Moves the alerting threshold. Adds no test and spends no budget.

    It's a separate class from SupportDetector because that keeps the budget
    divisible. With one class and a flag, three modules could each run at q
    while the real rate went to 3q.
    """

    kind = "modulator"

    def multipliers(self, window, nodes):
        """
        Per-timestep per-node multiplier on the threshold, shape (T, len(nodes)).

        Always >= 1 and not NaN. A timestep with no reading behind it gets 1.0,
        which leaves an uninformed modulator inert.

        We don't touch the score. Scaling it would invalidate the null it's
        judged against (anomalies/rain_gate.py has the full argument).
        """
        raise NotImplementedError(f"{self.name}.multipliers")


class SupportDetector(SupportModule):
    """
    Adds a score, therefore a test, therefore spends false alarm budget.

    The budget is what separates this from a modulator: a second test has a
    second tail to pay for, and allocate() below is where it gets paid.
    """

    kind = "detector"

    def score(self, window, nodes):
        """
        One raw score per node, shape (len(nodes),). Larger is more anomalous.

        NaN where we can't compute a score. NaN isn't zero and it isn't "did not
        exceed": it means this detector had no opinion, and the audit record
        shows it as unscorable instead of as an all-clear.
        """
        raise NotImplementedError(f"{self.name}.score")

    def null(self):
        """
        The fitted null this detector's raw scores are calibrated against.

        Anything with to_pvalue(score) will do, and the shipped one is
        anomalies.channel_scoring.ChannelNull. It's fitted at calibration time
        and loaded from a file. A detector that fit its null on the window it's
        judging would only be describing the data it just saw.
        """
        raise NotImplementedError(f"{self.name}.null")


class SupportScreen(SupportModule):
    """Screens readings before the model sees them. Fits neither kind above."""

    kind = "screen"

    def admit(self, window, nodes):
        """
        Per-timestep per-node bool, (T, len(nodes)), True where a reading is fit
        for the model. A screen adds no test and moves no threshold. It changes
        the model's input, and since the null and z_q were fitted on unscreened
        windows, the calibration no longer quite describes the score.
        """
        raise NotImplementedError(f"{self.name}.admit")


class SupportExplainer(SupportModule):
    """Names the likely cause of a firing. Adds no test and spends no budget."""

    kind = "explainer"

    def explain(self, window, nodes, fired):
        """
        One account per node, in `nodes` order, of what fired and why. fired is
        the (T, len(nodes)) grid of primary firings at the final threshold. An
        explainer only reads that decision, and the decision doesn't depend on
        what it says. It adds no test and moves no threshold, and takes no budget.

        A node we can't explain says so and names what was missing. No partial
        matches or defaults, because an operator would read a guess as a
        diagnosis.
        """
        raise NotImplementedError(f"{self.name}.explain")


# Budget


def _rate(name, value):
    """A false alarm rate has to sit strictly inside (0, 1)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise BudgetError(f"{name} must be a number, got {value!r}") from None
    if not (v == v) or v in (float("inf"), float("-inf")):
        raise BudgetError(f"{name} must be finite, got {value!r}")
    if not (0.0 < v < 1.0):
        raise BudgetError(f"{name} must be strictly between 0 and 1, got {v!r}")
    return v


def default_weights(n_detectors):
    """
    An even split across the primary and every detector.

    n+1 shares for n detectors. Each weight is 1/(n+1) and the sum is n/(n+1),
    below one for any n. Weighting a detector higher would be a claim about
    which test fires more often, and we haven't measured that.
    """
    if n_detectors < 0:
        raise BudgetError(f"cannot allocate for {n_detectors} detectors")
    if n_detectors == 0:
        return []
    return [1.0 / (n_detectors + 1)] * n_detectors


def allocate(q_total, detectors, weights=None):
    """
    Divides q_total across the primary and the detectors:

        allocate(q_total, detectors) -> (primary_q, [detector_q, ...])

    A primary at q plus a detector at q alerts at about 2q, with each part
    correct on its own. The operator sets one total and we split it. Modulators
    and screens take nothing. weights is one fraction per detector (even by
    default), and the primary keeps the rest:

        primary_q = q_total * (1 - sum(weights))

    The weights have to sum below one. At one the primary's threshold is
    infinite and above one the rate is negative. Both raise instead of
    clamping, because a clamp would switch the primary off without a word.
    It's a union bound, and the real combined rate comes in under q_total.
    Budgets come back in the order of `detectors`.
    """
    q = _rate("q_total", q_total)
    n = len(detectors)

    if weights is None:
        weights = default_weights(n)
    weights = [float(w) for w in weights]

    if len(weights) != n:
        raise BudgetError(f"{len(weights)} weights for {n} detectors")
    for i, w in enumerate(weights):
        if not (w == w) or w in (float("inf"), float("-inf")):
            raise BudgetError(f"weight {i} must be finite, got {w!r}")
        if w <= 0.0:
            raise BudgetError(
                f"weight {i} is {w!r}; a detector with no budget can never fire, "
                f"so switch it off instead of allocating it zero"
            )

    total = sum(weights)
    if total >= 1.0:
        raise BudgetError(
            f"detector weights sum to {total!r}, which leaves the primary "
            f"{1.0 - total!r} of the budget. They must sum below 1 so the "
            f"primary keeps a usable share."
        )

    return q * (1.0 - total), [q * w for w in weights]


def summarise(q_total, detectors, weights=None):
    """The split keyed by module name, for the audit record and for printing a plan."""
    primary_q, detector_q = allocate(q_total, detectors, weights)
    return {
        "q_total": float(q_total),
        "primary": primary_q,
        "detectors": {d.name: q for d, q in zip(detectors, detector_q, strict=True)},
    }


# Audit record

# Three verdicts, not two: a score that could not be computed is not the same
# as a score that did not exceed, and collapsing them is how an unscorable node
# reaches an operator as a quiet all-clear.
FIRED = "fired"
CLEAR = "clear"
NOT_SCORABLE = "not_scorable"

SUPPORT_KEYS = (
    "primary_q",
    "support_multiplier",
    "final_threshold",
    "primary_fired",
    "support",
    "verdict",
    "verdict_without",
)


def verdict(primary_fired, detector_fired, primary_scorable):
    """
    One node's verdict at one timestep, from the tests that ran.

    A union over the primary and the detectors, the same assumption the budget
    split makes. A node with no scorable test is reported as unscorable, not
    clear.
    """
    if primary_fired or any(detector_fired):
        return FIRED
    if not primary_scorable:
        return NOT_SCORABLE
    return CLEAR


def _fired(score, threshold):
    """
    NaN never fires, same rule as anomalies/rain_gate.py's fired().

    A scalar version, because that one takes arrays and this walks a record at
    a time. test_the_two_firing_rules_agree keeps the two in step.
    """
    return bool(np.isfinite(score) and score > threshold)


def primary_fired_grid(base_records, nodes, support_multiplier):
    """
    Which node-timesteps the primary fired on, at the modulated threshold.

    Built once, and both the record loop and the explainers read it.
    """
    node_pos = {node: j for j, node in enumerate(nodes)}
    grid = np.zeros(support_multiplier.shape, dtype=bool)
    for record in base_records:
        i = record["timestep"]
        j = node_pos[record["node"]]
        threshold = record["threshold"] * float(support_multiplier[i, j])
        grid[i, j] = _fired(record["score"], threshold)
    return grid


def module_entry(module, **fields):
    """
    One module's row inside a record.

    Name, kind and description are always there, including for a module that
    did nothing.
    """
    entry = {
        "name": module.name,
        "kind": module.kind,
        "describe": module.describe(),
        "ran": True,
        "fired": None,
        "budget_q": None,
    }
    entry.update(fields)
    return entry


def extend_records(
    base_records,
    nodes,
    modulators,
    detectors,
    support_multiplier,
    detector_scores,
    detector_pvalues,
    detector_budgets,
    primary_q,
    modulator_multipliers,
    explainers=(),
    explanations=None,
):
    """
    Add the support fields to the rain gate's records, in place.

    Every key rain_gate.decisions() writes keeps its meaning and the support
    fields go alongside. Two formats would mean two readers.

    base_records          what RainGate.decisions() returned, added to only
    support_multiplier    (T, N) product of every modulator's multiplier
    modulator_multipliers {name: (T, N)}, kept per module so the record can say
                          which modulator raised the bar and by how much
    detector_scores       {name: (N,)} one raw score per node for the window
    detector_pvalues      {name: (N,)} that score against the module's own null
    detector_budgets      {name: q} this module's share of q_total
    explanations          {name: [per-node result]} what each explainer said

    A detector scores a node once per window. We repeat that score on each of
    the node's rows instead of inventing a per-step breakdown. Explainer
    accounts are per node too.
    """
    node_pos = {node: j for j, node in enumerate(nodes)}
    explanations = explanations or {}
    fired_grid = primary_fired_grid(base_records, nodes, support_multiplier)

    for record in base_records:
        i = record["timestep"]
        j = node_pos[record["node"]]

        gate_threshold = record["threshold"]
        mult = float(support_multiplier[i, j])
        final_threshold = gate_threshold * mult
        score = record["score"]
        scorable = record["scorable"]
        primary_fired = bool(fired_grid[i, j])

        entries = []
        for m in modulators:
            entries.append(
                module_entry(
                    m,
                    multiplier=float(modulator_multipliers[m.name][i, j]),
                    budget_q=None,
                )
            )

        fired_flags = []
        for d in detectors:
            d_score = float(detector_scores[d.name][j])
            d_p = float(detector_pvalues[d.name][j])
            d_q = detector_budgets[d.name]
            # We fire on the p-value. Raw scores come in different units, and
            # the budget only means one thing across detectors this way.
            d_fired = bool(np.isfinite(d_p) and d_p < d_q)
            fired_flags.append(d_fired)
            entries.append(
                module_entry(
                    d,
                    score=d_score,
                    pvalue=d_p,
                    budget_q=d_q,
                    fired=d_fired,
                    scorable=bool(np.isfinite(d_score)),
                )
            )

        for x in explainers:
            result = explanations[x.name][j]
            entries.append(
                module_entry(
                    x,
                    cause=result.get("cause"),
                    evidence=result.get("evidence"),
                    confidence=result.get("confidence"),
                    explanation=result.get("explanation"),
                    budget_q=None,
                )
            )

        final = verdict(primary_fired, fired_flags, scorable)

        # What the verdict would have been with each module switched off, one
        # at a time. A modulator drops out by dividing its multiplier back out
        # of the threshold; a detector drops out by removing its vote.
        without = {}
        for m in modulators:
            m_mult = float(modulator_multipliers[m.name][i, j])
            reduced = final_threshold / m_mult if m_mult else final_threshold
            without[m.name] = verdict(_fired(score, reduced), fired_flags, scorable)
        for k, d in enumerate(detectors):
            others = [f for idx, f in enumerate(fired_flags) if idx != k]
            without[d.name] = verdict(primary_fired, others, scorable)
        for x in explainers:
            # An explainer casts no vote. We record that anyway, to show it's free.
            without[x.name] = final

        record.update(
            {
                "primary_q": primary_q,
                "support_multiplier": mult,
                "final_threshold": final_threshold,
                "primary_fired": primary_fired,
                "support": entries,
                "verdict": final,
                "verdict_without": without,
            }
        )

    return base_records


# The stack


class SupportStack:
    """
    The modules attached to one run, and the arithmetic that combines them.

    Built through load(). Name resolution, contract checks, collisions and the
    budget split all happen there, before anything runs.

    We sort modules by name. (a*b)*c and (a*c)*b can differ in floating point,
    and command line order shouldn't change a result. An empty stack is the
    identity, with no special case.
    """

    def __init__(self, modules=None, q_total=DEFAULT_Q_TOTAL, weights=None):
        # Sorted here as well as in registry.load, for stacks built directly in tests
        self.modules = sorted(modules or [], key=lambda m: m.name)
        self.q_total = q_total
        self.weights = weights
        # an impossible split raises now, not at the first detection
        self.primary_q, self._detector_q = allocate(q_total, self.detectors, weights)

    @classmethod
    def load(cls, names, model, q_total=DEFAULT_Q_TOTAL, weights=None):
        """
        Resolve names against SUPPORT_REGISTRY and attach them to a model.

        names may be None or empty, which is the no-support run and is not a
        special case downstream.
        """
        # imported here because registry imports this module
        from strawberrywatch.support_modules import registry

        return cls(registry.load(names or [], model), q_total=q_total, weights=weights)

    # Composition

    def of_kind(self, kind):
        return [m for m in self.modules if m.kind == kind]

    @property
    def screens(self):
        return self.of_kind("screen")

    @property
    def modulators(self):
        return self.of_kind("modulator")

    @property
    def detectors(self):
        return self.of_kind("detector")

    @property
    def explainers(self):
        return self.of_kind("explainer")

    @property
    def names(self):
        return [m.name for m in self.modules]

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def describe(self):
        """One line per module, for the audit record and for printing a plan."""
        return [m.describe() for m in self.modules]

    def budget(self):
        """The split, keyed by module name. Empty detectors means all primary."""
        return summarise(self.q_total, self.detectors, self.weights)

    def detector_budgets(self):
        return dict(zip([d.name for d in self.detectors], self._detector_q, strict=True))

    # Running

    def admit(self, window, nodes, shape):
        """
        Which readings the screens let through, as (T, N) bool.

        Intersection: a reading is admitted only if every screen admits it. An
        empty screen list admits everything, which is the identity.
        """
        keep = np.ones(shape, dtype=bool)
        for s in self.screens:
            keep = keep & np.asarray(s.admit(window, nodes), dtype=bool)
        return keep

    def multipliers(self, window, nodes, shape):
        """
        Per-modulator multipliers and their product, both as (T, N).

        Returns (product, {name: multiplier}). The per-module arrays are kept
        because the audit record has to say which modulator raised the bar, not
        just that something did.
        """
        per_module = {}
        product = np.ones(shape, dtype=float)
        for m in self.modulators:
            mult = np.asarray(m.multipliers(window, nodes), dtype=float)
            if mult.shape != shape:
                raise SupportError(f"{m.name}.multipliers returned {mult.shape}, expected {shape}")
            if np.any(~np.isfinite(mult)):
                raise SupportError(f"{m.name}.multipliers returned a non-finite multiplier")
            if np.any(mult < 1.0):
                raise SupportError(
                    f"{m.name}.multipliers returned a multiplier below 1, which would "
                    f"lower the alerting bar; this layer only raises it"
                )
            per_module[m.name] = mult
            product = product * mult
        return product, per_module

    def detector_results(self, window, nodes):
        """
        Every detector's raw score and its p-value against its own null.

        Returns ({name: (N,) score}, {name: (N,) pvalue}). Raw scores aren't
        comparable across detectors. Firing is decided on the p-value.
        """
        scores, pvalues = {}, {}
        n = len(nodes)
        for d in self.detectors:
            raw = np.asarray(d.score(window, nodes), dtype=float)
            if raw.shape != (n,):
                raise SupportError(
                    f"{d.name}.score returned {raw.shape}, expected {(n,)}: one score per node"
                )
            scores[d.name] = raw
            pvalues[d.name] = np.asarray(d.null().to_pvalue(raw), dtype=float)
        return scores, pvalues

    def explanations(self, window, nodes, fired):
        """
        Every explainer's per-node account, keyed by module name.

        Returns {name: [result, ...]} in `nodes` order. We refuse a wrong
        length here, before it gets filed against the wrong node.
        """
        out = {}
        n = len(nodes)
        for x in self.explainers:
            results = list(x.explain(window, nodes, fired))
            if len(results) != n:
                raise SupportError(
                    f"{x.name}.explain returned {len(results)} results, expected {n}: one per node"
                )
            out[x.name] = results
        return out

    def decisions(self, gate, scores, pvalues, rain_mm, nodes, window=None, timestamps=None):
        """
        The full audit trail: the rain gate's record for every timestep and
        node, extended with every support module that ran.

        gate is an anomalies.rain_gate.RainGate. We take its record first,
        unmodified, and add the support fields on top. With no modules attached
        the extra fields describe an empty stack and every verdict matches the
        gate's own `fired`.
        """
        records = gate.decisions(scores, pvalues, rain_mm, timestamps=timestamps, nodes=nodes)
        shape = (len(records) // len(nodes), len(nodes))

        product, per_module = self.multipliers(window, nodes, shape)
        det_scores, det_pvalues = self.detector_results(window, nodes)

        # Explainers run last, on the firings the rest of the stack produced
        fired_grid = primary_fired_grid(records, nodes, product)
        accounts = self.explanations(window, nodes, fired_grid)

        return extend_records(
            records,
            nodes,
            self.modulators,
            self.detectors,
            product,
            det_scores,
            det_pvalues,
            self.detector_budgets(),
            self.primary_q,
            per_module,
            explainers=self.explainers,
            explanations=accounts,
        )

    def __repr__(self):
        return f"<SupportStack {self.names} q_total={self.q_total!r}>"


__all__ = [
    "ANY_CONTRACT",
    "CLEAR",
    "DEFAULT_Q_TOTAL",
    "FIRED",
    "KINDS",
    "NOT_SCORABLE",
    "SUPPORT_KEYS",
    "BudgetError",
    "SupportDetector",
    "SupportError",
    "SupportExplainer",
    "SupportModulator",
    "SupportModule",
    "SupportScreen",
    "SupportStack",
    "allocate",
    "default_weights",
    "extend_records",
    "module_entry",
    "primary_fired_grid",
    "summarise",
    "verdict",
]
