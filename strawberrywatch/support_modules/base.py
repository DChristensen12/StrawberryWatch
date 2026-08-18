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

    Subclass one of the four kinds below, not this. `kind` is what the
    registry, the budget split and the audit record all dispatch on, so a class
    that leaves it empty is refused at registration rather than silently
    treated as a modulator.

    Nothing in this module imports a model or names one. A module attaches to
    anything declaring the contract it needs, which is what lets one module
    reach both models by the same path.
    """

    # Short identifier used on the command line and as the audit record key.
    name: str = ""

    # Set by a kind, never by a concrete module.
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

        Reads what the model declares about itself and never asks which model
        it is. A subclass adding its own checks should call up to this first.

        Collision against a model's BUILTIN_SUPPORT is deliberately not here.
        That applies to every module equally, so it lives in the loader where a
        subclass overriding validate cannot drop it.
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

    Separate from SupportDetector because that is what makes the budget
    divisible. One class with a flag and nothing stops three modules running at
    q each while the real rate goes to 3q.
    """

    kind = "modulator"

    def multipliers(self, window, nodes):
        """
        Per-timestep per-node multiplier on the threshold, shape (T, len(nodes)).

        Always >= 1, never NaN. A timestep with no reading behind it is 1.0, so an
        uninformed modulator is inert rather than suppressing.

        The score is never touched. Scaling scores instead would invalidate the
        null they are judged against, argued in full in anomalies/rain_gate.py.
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

        NaN where the score cannot honestly be computed. NaN is not zero and it
        is not "did not exceed": it means this detector had nothing to say, and
        the framework carries it through to the audit record as unscorable
        rather than as a quiet all-clear.
        """
        raise NotImplementedError(f"{self.name}.score")

    def null(self):
        """
        The fitted null this detector's raw scores are calibrated against.

        Anything exposing to_pvalue(score) will do; the shipped one is
        anomalies.channel_scoring.ChannelNull. Fitted at calibration time and
        loaded from an artifact, never fitted here. A detector that fits its
        null on the window it is judging has no null, it has a description of
        the data it just saw.
        """
        raise NotImplementedError(f"{self.name}.null")


class SupportScreen(SupportModule):
    """Screens readings before the model sees them. Fits neither kind above."""

    kind = "screen"

    def admit(self, window, nodes):
        """
        Per-timestep per-node bool, shape (T, len(nodes)). True where the
        reading is fit for the model to consume.

        Deliberately not multipliers() and not score(). A screen adds no test,
        so it is not a detector and takes no budget; it moves no threshold, so
        it is not a modulator. What it changes is the model's input, and that
        has a consequence neither of the other kinds carries: the calibrated
        null and z_q were fitted on unscreened windows, so screening changes
        the distribution the primary score is drawn from and the calibration
        no longer describes it.
        """
        raise NotImplementedError(f"{self.name}.admit")


class SupportExplainer(SupportModule):
    """Names the likely cause of a firing. Adds no test and spends no budget."""

    kind = "explainer"

    def explain(self, window, nodes, fired):
        """
        One account per node, in `nodes` order, of what fired and why.

        fired is the (T, len(nodes)) bool grid of primary firings at the final
        threshold. An explainer reads the decision rather than remaking it, so
        it runs last and nothing downstream of it depends on what it says.

        Deliberately not score(), multipliers() or admit(). A detector adds a
        second tail and allocate() charges it; this adds no test, so charging
        it would cost the primary sensitivity for a module that cannot fire. A
        modulator moves the threshold and a screen moves the model's input;
        this moves neither, and tests/test_support_modules.py pins that.

        A node that cannot be explained comes back saying so, naming what was
        missing. Never a partial match and never a default: a guess with the
        evidence filed off reaches an operator as a diagnosis.
        """
        raise NotImplementedError(f"{self.name}.explain")


# Budget


def _rate(name, value):
    """A false alarm rate has to be strictly inside (0, 1) to mean anything."""
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

    n+1 shares for n detectors, so each weight is 1/(n+1) and the sum is
    n/(n+1), below one for every n. Even is the honest default: weighting a
    detector higher is a claim about which test is more likely to fire, and
    nothing has measured that yet.
    """
    if n_detectors < 0:
        raise BudgetError(f"cannot allocate for {n_detectors} detectors")
    if n_detectors == 0:
        return []
    return [1.0 / (n_detectors + 1)] * n_detectors


def allocate(q_total, detectors, weights=None):
    """
    Divide q_total across the primary and the detectors.

        allocate(q_total, detectors) -> (primary_q, [detector_q, ...])

    Primary at q plus a detector at q alerts at about 2q, and every component
    is individually correct while it happens. So the operator sets one total
    and it gets split, never handed out fresh per module. Modulators and
    screens add no test and take nothing.

    detectors is the list of detector modules (or anything len() works on; only
    the count is read). weights is one fraction of q_total per detector,
    defaulting to an even split. The primary keeps whatever is left:

        primary_q = q_total * (1 - sum(weights))

    which is why the weights must sum below one. At a sum of exactly one the
    primary is left a budget of zero, meaning a threshold at infinity and a
    detector that can never fire; above one it is negative, which is not a rate
    at all. Both raise rather than clamping, because a clamp here would produce
    a working-looking run whose primary had been silently switched off.

    The split is a union bound, so the real combined rate lands under q_total
    rather than on it. Detector budgets come back in the same order as
    `detectors`.
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

    Union over the primary and the detectors, which is the same assumption the
    budget split is built on. A node with nothing scorable is reported as such
    rather than as clear.
    """
    if primary_fired or any(detector_fired):
        return FIRED
    if not primary_scorable:
        return NOT_SCORABLE
    return CLEAR


def _fired(score, threshold):
    """
    NaN never fires, same rule as anomalies/rain_gate.py's fired().

    A scalar version because that one takes arrays and this walks a record at a
    time. Written twice, so test_the_two_firing_rules_agree pins them together.
    """
    return bool(np.isfinite(score) and score > threshold)


def primary_fired_grid(base_records, nodes, support_multiplier):
    """
    Which node-timesteps the primary fired on, at the modulated threshold.

    Built once so the record loop and the explainers read the same firings
    rather than each applying the rule to the same numbers separately.
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

    Name, kind and description are always present so a reader can account for
    every attached module even when it did nothing.
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

    A detector scores a node once for the whole window, so the same score
    appears on every timestep of that node's rows. Spreading it over timesteps
    would put a breakdown in the record that no module ever computed. An
    explainer's account is per node for the same reason and is carried the
    same way.
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
            # Fires on the p-value, not the raw score, so detectors with
            # different score units stay comparable and the budget means the
            # same thing for all of them.
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
            # An explainer casts no vote, so switching it off cannot move the
            # verdict. Written down anyway, so the record shows it costs nothing.
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

    Built through load() so that name resolution, contract validation, builtin
    collision and the budget split all happen before anything runs. A stack
    that exists is a stack that has already been checked.

    Modules come in as a list and get sorted by name, so command line order
    cannot reach anything downstream. Sorted rather than trusting
    commutativity: a*b == b*a holds, but (a*b)*c and (a*c)*b need not, and
    three modulators reaches that. An empty stack is the identity with no
    special case, so zero modules walk the same code as three.
    """

    def __init__(self, modules=None, q_total=DEFAULT_Q_TOTAL, weights=None):
        # sorted here as well as in registry.load, so the ordering guarantee
        # holds for a stack built directly in a test
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

        Returns ({name: (N,) score}, {name: (N,) pvalue}). Raw scores are not
        comparable across detectors, which is exactly why the firing decision
        downstream is made on the p-value and never on the score.
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

        Returns {name: [result, ...]} in `nodes` order. A wrong length is
        refused here rather than written into the record against the wrong node.
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

        gate is an anomalies.rain_gate.RainGate. Its record is produced first
        and unmodified, so the primary decision is written down exactly as the
        gate made it, and the support fields are added on top. With no modules
        attached the added fields describe an empty stack and the verdict
        matches the gate's own `fired` for every row.
        """
        records = gate.decisions(scores, pvalues, rain_mm, timestamps=timestamps, nodes=nodes)
        shape = (len(records) // len(nodes), len(nodes))

        product, per_module = self.multipliers(window, nodes, shape)
        det_scores, det_pvalues = self.detector_results(window, nodes)

        # Explainers run last and read the firings the rest of the stack
        # produced, which is what keeps them out of the decision entirely.
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
