"""
The registry, the kinds, the stack, the budget split and the audit record.

Newtwork Run and Settling Pool are placeholders and every method raises.
Reaching the raise is the point: it proves the call got all the way to the
module. Trial Bed is implemented, so it is exercised for real, and what it
says is pinned in tests/test_trial_bed.py. Anything needing a number uses a
stub defined here.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from strawberrywatch.anomalies.rain_gate import RainGate, fired
from strawberrywatch.models.Cobble_Shoal import CobbleShoal
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
from strawberrywatch.support_modules import (
    SUPPORT_REGISTRY,
    SupportStack,
    allocate,
    base,
    registry,
)
from strawberrywatch.support_modules.newtwork_run import NewtworkRun
from strawberrywatch.support_modules.settling_pool import SettlingPool
from strawberrywatch.support_modules.trial_bed import TrialBed

BASE_THRESHOLD = 39.13047989749433
NODES = ["n1", "n2", "n3"]
T = 12


# Stubs, for the parts that need a module to return something


class ConstantNull:
    """A null that maps a raw score to a fixed p-value, so firing is decidable."""

    def __init__(self, p):
        self.p = p

    def to_pvalue(self, score):
        s = np.asarray(score, dtype=float)
        return np.where(np.isfinite(s), self.p, np.nan)


class StubModulator(base.SupportModulator):
    """Raises the bar by a fixed factor everywhere"""

    def __init__(self, name, factor):
        self.name = name
        self.factor = factor

    def multipliers(self, window, nodes):
        return np.full((T, len(nodes)), self.factor, dtype=float)


class StubDetector(base.SupportDetector):
    """Scores every node the same and reports a fixed p-value"""

    def __init__(self, name, score_value=1.0, p=1e-9):
        self.name = name
        self.score_value = score_value
        self.p = p

    def score(self, window, nodes):
        return np.full(len(nodes), self.score_value, dtype=float)

    def null(self):
        return ConstantNull(self.p)


class StubExplainer(base.SupportExplainer):
    """Reports the same account for every node, and keeps what it was shown"""

    def __init__(self, name, cause="sewage"):
        self.name = name
        self.cause = cause
        self.seen = None

    def explain(self, window, nodes, fired):
        self.seen = np.asarray(fired)
        return [
            {
                "cause": self.cause,
                "evidence": {"conductivity": "up"},
                "confidence": "good (2 discriminating channels)",
                "explanation": f"consistent with {self.cause}: conductivity up",
            }
            for _ in nodes
        ]


class StubScreen(base.SupportScreen):
    """Admits everything"""

    name = "stub_screen"

    def admit(self, window, nodes):
        return np.ones((T, len(nodes)), dtype=bool)


def a_gate():
    return RainGate(base_threshold=BASE_THRESHOLD, mode="off", sample_hours=1.0)


def a_run(scores=None):
    """Scores, p-values and rain for one small decision grid."""
    s = np.full((T, len(NODES)), 0.5 * BASE_THRESHOLD) if scores is None else scores
    p = np.full_like(s, 0.5)
    rain = np.zeros(T)
    return s, p, rain


# The registry


def test_registry_holds_the_three_named_modules():
    assert SUPPORT_REGISTRY == {
        "trial_bed": TrialBed,
        "newtwork_run": NewtworkRun,
        "settling_pool": SettlingPool,
    }


def test_unknown_name_raises_and_lists_the_valid_options():
    """
    The worst outcome is a typo that attaches nothing, scores with no support
    and reports success. It has to raise, and the message has to say what would
    have worked.
    """
    with pytest.raises(registry.UnknownSupportModule) as exc:
        registry.support_class("trail_bed")
    message = str(exc.value)
    assert "trail_bed" in message
    for name in SUPPORT_REGISTRY:
        assert name in message, f"the error does not offer {name}"

    with pytest.raises(registry.UnknownSupportModule):
        registry.load(["trial_bed", "nope"], DuskCrayfish)


def test_the_same_module_twice_is_refused():
    """Running a detector twice would spend its budget twice."""
    with pytest.raises(base.SupportError, match="more than once"):
        registry.load(["trial_bed", "trial_bed"], CobbleShoal)


# The kinds


def test_modulators_and_detectors_are_different_kinds():
    """
    The distinction that stops the alarm rate rising silently: a detector
    spends budget and a modulator does not.
    """
    assert base.SupportModulator.kind == "modulator"
    assert base.SupportDetector.kind == "detector"
    assert base.SupportModulator.kind != base.SupportDetector.kind


def test_every_module_exposes_name_describe_and_validate():
    for name, cls in SUPPORT_REGISTRY.items():
        module = cls()
        assert module.name == name, f"{cls.__name__}.name does not match its registry key"
        assert isinstance(module.describe(), str) and module.describe().strip()
        assert module.kind in base.KINDS
        module.validate(CobbleShoal)


def test_placeholders_raise_and_say_what_they_will_do():
    """
    Each placeholder raises from its one required method, and the message names
    the job. A bare NotImplementedError would leave the next person reading the
    class name to guess.
    """
    window, nodes = {}, NODES

    with pytest.raises(NotImplementedError) as exc:
        NewtworkRun().score(window, nodes)
    assert "downstream" in str(exc.value) and "budget" in str(exc.value)

    with pytest.raises(NotImplementedError) as exc:
        SettlingPool().admit(window, nodes)
    assert "before the model" in str(exc.value)


def test_trial_bed_is_an_explainer_because_it_fits_no_other_kind():
    """
    An explainer runs after the decision and reads it. It adds no test, so it
    is not a detector and takes no budget; it moves no threshold, so it is not
    a modulator; it runs after the model rather than before, so it is not a
    screen. Charging it budget would cost the primary sensitivity for a module
    that can never fire.
    """
    bed = TrialBed()
    assert bed.kind == "explainer"
    assert not isinstance(bed, base.SupportDetector)
    assert not isinstance(bed, base.SupportModulator)
    assert not isinstance(bed, base.SupportScreen)
    assert isinstance(bed, base.SupportExplainer)

    # It has none of the other kinds' required methods, so it could not have
    # been registered as one of them without inventing an arithmetic for it.
    assert not hasattr(bed, "multipliers")
    assert not hasattr(bed, "score")
    assert not hasattr(bed, "admit")

    stack = SupportStack.load(["trial_bed"], CobbleShoal)
    assert stack.explainers == list(stack)
    assert stack.detectors == [] and stack.modulators == [] and stack.screens == []
    # An explainer takes no budget: the primary keeps all of it.
    assert stack.budget()["primary"] == pytest.approx(stack.q_total)
    assert stack.budget()["detectors"] == {}


def test_settling_pool_is_a_screen_because_it_fits_neither_other_kind():
    """
    A screen runs before the model. It adds no test, so it is not a detector
    and takes no budget; it moves no threshold, so it is not a modulator. What
    it does change is the model's input, which moves the distribution the
    primary's calibration was fitted against.
    """
    pool = SettlingPool()
    assert pool.kind == "screen"
    assert not isinstance(pool, base.SupportDetector)
    assert not isinstance(pool, base.SupportModulator)
    assert isinstance(pool, base.SupportScreen)

    # It has neither of the other kinds' required methods, so it could not have
    # been registered as one of them without inventing an arithmetic for it.
    assert not hasattr(pool, "multipliers")
    assert not hasattr(pool, "score")

    stack = SupportStack.load(["settling_pool"], CobbleShoal)
    assert stack.screens == list(stack)
    assert stack.detectors == [] and stack.modulators == []
    # A screen takes no budget: the primary keeps all of it.
    assert stack.budget()["primary"] == pytest.approx(stack.q_total)


# Stacking


def test_the_three_call_shapes_take_the_same_path():
    """
    No support, one module and two modules are the same code with a different
    list length. The no-support stack is the identity.
    """
    none_stack = SupportStack.load(None, CobbleShoal)
    empty_stack = SupportStack.load([], CobbleShoal)
    one = SupportStack.load(["trial_bed"], CobbleShoal)
    two = SupportStack.load(["trial_bed", "newtwork_run"], CobbleShoal)

    assert len(none_stack) == 0 and len(empty_stack) == 0
    assert len(one) == 1 and len(two) == 2
    assert none_stack.names == empty_stack.names == []
    assert set(two.names) == {"trial_bed", "newtwork_run"}

    # Modules attach as a list, not a tuple.
    assert isinstance(two.modules, list)


def test_an_empty_stack_leaves_the_gate_decision_untouched():
    """
    Attaching nothing must reproduce the rain gate's own verdict exactly, or
    "no support" is a different code path rather than the same one at length
    zero.
    """
    scores = np.linspace(0.2, 2.0, T * len(NODES)).reshape(T, len(NODES)) * BASE_THRESHOLD
    s, p, rain = a_run(scores)
    gate = a_gate()
    records = SupportStack.load([], CobbleShoal).decisions(gate, s, p, rain, NODES, window={})

    for r in records:
        assert r["support"] == []
        assert r["support_multiplier"] == 1.0
        assert r["final_threshold"] == r["threshold"]
        assert r["primary_fired"] is r["fired"]
        assert r["verdict"] == (base.FIRED if r["fired"] else base.CLEAR)
    assert any(r["fired"] for r in records), "vacuous if nothing fires at all"


@pytest.mark.parametrize("kind", ["modulator", "detector"])
def test_order_in_the_list_does_not_change_the_result(kind):
    """
    Every permutation of three same-kind modules, compared bit for bit.

    The stack sorts by name and combines in that order, so the sequence of
    floating point operations is the same whatever the operator typed. Relying
    on multiplication commuting would not cover this: a*b == b*a holds, but
    (a*b)*c and (a*c)*b need not.
    """
    if kind == "modulator":
        made = {n: StubModulator(n, f) for n, f in (("m_a", 1.7), ("m_b", 1.3), ("m_c", 2.9))}
    else:
        made = {
            n: StubDetector(n, score_value=v, p=q)
            for n, v, q in (("d_a", 1.0, 1e-9), ("d_b", 2.0, 0.5), ("d_c", 3.0, 1e-7))
        }

    s, p, rain = a_run()
    reference = None
    for order in itertools.permutations(made):
        stack = SupportStack([made[n] for n in order])
        records = stack.decisions(a_gate(), s, p, rain, NODES, window={})

        signature = [
            (
                r["timestep"],
                r["node"],
                repr(r["final_threshold"]),
                repr(r["support_multiplier"]),
                r["verdict"],
                tuple(sorted((e["name"], repr(e.get("score")), e["fired"]) for e in r["support"])),
                tuple(sorted(r["verdict_without"].items())),
            )
            for r in records
        ]
        if reference is None:
            reference = signature
        else:
            assert signature == reference, f"order {order} changed the result"

    assert reference, "no permutations were compared"


def test_a_modulator_may_not_lower_the_bar():
    """This layer raises the alerting threshold. A factor below 1 is a bug."""
    s, p, rain = a_run()
    stack = SupportStack([StubModulator("down", 0.5)])
    with pytest.raises(base.SupportError, match="below 1"):
        stack.decisions(a_gate(), s, p, rain, NODES, window={})


# Budget


def test_budget_is_divided_not_handed_out_fresh():
    """
    The failure this exists to stop: primary at q, each detector at q, and the
    real rate is about q times the number of tests while every component
    reports q.
    """
    q = 1e-4
    primary_q, detector_q = allocate(q, [StubDetector("a"), StubDetector("b")])
    assert primary_q + sum(detector_q) == pytest.approx(q)
    assert primary_q < q, "the primary was handed the whole budget as well"
    for d in detector_q:
        assert 0 < d < q

    # No detectors: the primary keeps all of it.
    assert allocate(q, []) == (q, [])


def test_weights_must_sum_below_one():
    """At a sum of 1 the primary is left nothing, which is a threshold at infinity."""
    detectors = [StubDetector("a"), StubDetector("b")]
    with pytest.raises(base.BudgetError, match="sum below 1"):
        allocate(1e-4, detectors, weights=[0.5, 0.5])
    with pytest.raises(base.BudgetError, match="sum below 1"):
        allocate(1e-4, detectors, weights=[0.9, 0.4])

    primary_q, detector_q = allocate(1e-4, detectors, weights=[0.25, 0.25])
    assert primary_q == pytest.approx(0.5e-4)
    assert detector_q == pytest.approx([0.25e-4, 0.25e-4])


def test_a_rate_outside_zero_to_one_is_refused():
    for bad in (0.0, 1.0, -0.1, 2.0, float("nan"), float("inf")):
        with pytest.raises(base.BudgetError):
            allocate(bad, [])


def test_modulators_screens_and_explainers_take_no_budget():
    """Only detectors spend. Every other kind leaves the primary whole."""
    stack = SupportStack(
        [
            StubModulator("m_a", 1.5),
            StubModulator("m_b", 2.0),
            StubScreen(),
            StubExplainer("x_a"),
        ]
    )
    split = stack.budget()
    assert split["detectors"] == {}
    assert split["primary"] == pytest.approx(stack.q_total)


def test_budget_shrinks_the_primary_as_detectors_are_added():
    q = 1e-4
    seen = []
    for n in range(4):
        detectors = [StubDetector(f"d{i}") for i in range(n)]
        seen.append(allocate(q, detectors)[0])
    assert seen == sorted(seen, reverse=True), "adding a detector did not cost the primary"
    assert seen[0] == q


# The audit record


def test_the_record_extends_the_rain_gates_own_format():
    """
    Every key the rain gate writes survives with its meaning, and the support
    fields are added beside them. Two formats would mean two readers.
    """
    s, p, rain = a_run()
    gate = a_gate()
    gate_keys = set(gate.decisions(s, p, rain, nodes=NODES)[0])

    stack = SupportStack([StubModulator("m_a", 2.0), StubDetector("d_a", p=1e-9)])
    record = stack.decisions(a_gate(), s, p, rain, NODES, window={})[0]

    assert gate_keys <= set(record), "the support record dropped a rain gate field"
    assert set(base.SUPPORT_KEYS) <= set(record)


def test_the_record_answers_why_did_this_fire():
    """Every module that ran is present with its own budget and its own verdict."""
    s, p, rain = a_run()
    stack = SupportStack([StubModulator("m_a", 2.0), StubDetector("d_a", p=1e-9)])
    record = stack.decisions(a_gate(), s, p, rain, NODES, window={})[0]

    by_name = {e["name"]: e for e in record["support"]}
    assert set(by_name) == {"m_a", "d_a"}

    assert by_name["m_a"]["kind"] == "modulator"
    assert by_name["m_a"]["budget_q"] is None, "a modulator was charged budget"
    assert by_name["m_a"]["multiplier"] == 2.0

    assert by_name["d_a"]["kind"] == "detector"
    assert by_name["d_a"]["budget_q"] == pytest.approx(stack.q_total / 2)
    assert by_name["d_a"]["fired"] is True
    assert record["verdict"] == base.FIRED

    for entry in record["support"]:
        assert entry["describe"].strip(), "a module ran without describing itself"

    # And the modulator did move the bar it is recorded as having moved.
    assert record["final_threshold"] == pytest.approx(2.0 * record["threshold"])


def test_the_record_answers_what_changes_if_i_turn_that_off():
    """
    verdict_without, per module, without re-running anything. The detector here
    is the only reason the node fires, so switching it off must clear it.
    """
    s, p, rain = a_run()
    stack = SupportStack([StubDetector("d_a", p=1e-9), StubDetector("d_b", p=0.9)])
    record = stack.decisions(a_gate(), s, p, rain, NODES, window={})[0]

    assert record["primary_fired"] is False, "the primary should not fire on its own here"
    assert record["verdict"] == base.FIRED
    assert record["verdict_without"]["d_a"] == base.CLEAR, "d_a was the only reason it fired"
    assert record["verdict_without"]["d_b"] == base.FIRED, "d_b was not the reason"


def test_an_explainer_is_in_the_record_and_changes_no_verdict():
    """
    Its account rides in the same record beside the modules that decided, and
    verdict_without says in the record itself that switching it off is free.
    """
    s, p, rain = a_run()
    stack = SupportStack([StubDetector("d_a", p=1e-9), StubExplainer("x_a")])
    record = stack.decisions(a_gate(), s, p, rain, NODES, window={})[0]

    by_name = {e["name"]: e for e in record["support"]}
    assert set(by_name) == {"d_a", "x_a"}

    entry = by_name["x_a"]
    assert entry["kind"] == "explainer"
    assert entry["budget_q"] is None, "an explainer was charged budget"
    assert entry["fired"] is None, "an explainer cast a vote"
    assert entry["cause"] == "sewage"
    assert entry["evidence"] == {"conductivity": "up"}
    assert entry["confidence"].startswith("good")
    assert "consistent with sewage" in entry["explanation"]

    assert record["verdict"] == base.FIRED
    assert record["verdict_without"]["x_a"] == record["verdict"]


def test_an_explainer_reads_the_firings_the_rest_of_the_stack_made():
    """
    It is handed the primary's firings at the final threshold, so the modulator
    above it has already moved the bar by the time it looks.
    """
    scores = np.linspace(0.2, 2.0, T * len(NODES)).reshape(T, len(NODES)) * BASE_THRESHOLD
    s, p, rain = a_run(scores)

    explainer = StubExplainer("x_a")
    SupportStack([explainer]).decisions(a_gate(), s, p, rain, NODES, window={})
    plain = explainer.seen.copy()

    raised = StubExplainer("x_a")
    SupportStack([raised, StubModulator("m_a", 2.0)]).decisions(
        a_gate(), s, p, rain, NODES, window={}
    )

    assert plain.shape == (T, len(NODES))
    assert plain.any(), "vacuous if nothing fired at all"
    assert raised.seen.sum() < plain.sum(), "the raised bar did not reach the explainer"


def test_an_explainer_returning_the_wrong_number_of_accounts_is_refused():
    """A short list would file one node's account against another node."""

    class ShortExplainer(base.SupportExplainer):
        name = "short"

        def explain(self, window, nodes, fired):
            return [{"cause": None}]

    stack = SupportStack([ShortExplainer()])
    with pytest.raises(base.SupportError, match="one per node"):
        stack.explanations({}, NODES, np.zeros((T, len(NODES)), dtype=bool))


def test_an_unscorable_score_is_not_reported_as_clear():
    """NaN never fires, and it is not the same answer as "did not exceed"."""
    s, p, rain = a_run()
    s = s.copy()
    s[0, 0] = np.nan
    stack = SupportStack([StubDetector("d_a", p=0.9)])
    records = stack.decisions(a_gate(), s, p, rain, NODES, window={})

    first = records[0]
    assert first["node"] == "n1" and first["timestep"] == 0
    assert first["scorable"] is False
    assert first["primary_fired"] is False
    assert first["verdict"] == base.NOT_SCORABLE
    assert records[1]["verdict"] == base.CLEAR


def test_the_two_firing_rules_agree():
    """
    base._fired walks a record at a time and rain_gate.fired takes arrays, so
    the rule is written twice. There were three rain formulas in this codebase
    once and decisions() carried its own copy of fired(); this is the same
    defect class, so the two are pinned against each other rather than trusted.
    """
    rng = np.random.default_rng(0)
    scores = rng.normal(BASE_THRESHOLD, 10.0, size=200)
    scores[::7] = np.nan
    thresholds = rng.normal(BASE_THRESHOLD, 2.0, size=200)

    array_path = fired(scores, thresholds)
    record_path = [base._fired(s, t) for s, t in zip(scores, thresholds, strict=True)]
    assert list(array_path) == record_path
    assert any(record_path) and not all(record_path), "vacuous if every case agrees trivially"
    assert not any(base._fired(np.nan, t) for t in thresholds), "NaN fired somewhere"


def test_one_record_per_timestep_per_node():
    s, p, rain = a_run()
    records = SupportStack([StubDetector("d_a")]).decisions(a_gate(), s, p, rain, NODES, window={})
    assert len(records) == T * len(NODES)
    assert {(r["timestep"], r["node"]) for r in records} == {
        (i, n) for i in range(T) for n in NODES
    }


# Modularity sweep


@pytest.mark.parametrize(
    "model", [DuskCrayfish, CobbleShoal], ids=["dusk_crayfish", "cobble_shoal"]
)
@pytest.mark.parametrize("name,phrase", [("newtwork_run", "Newtwork Run")])
def test_the_same_module_reaches_both_models(model, name, phrase):
    """
    The point of the framework: one module, two models with different input
    contracts, one path to the module's method.

    NotImplementedError is the pass condition. It is raised by the placeholder
    itself, which means the call travelled from the stack through the module
    and arrived, and it arrived identically for a sequence-contract model and a
    nested-node-contract one. The message is matched so that the raise is known
    to come from the module rather than from anything on the way.
    """
    stack = SupportStack.load([name], model)
    assert stack.names == [name]

    with pytest.raises(NotImplementedError, match=phrase):
        stack.detector_results({}, NODES)


@pytest.mark.parametrize(
    "model", [DuskCrayfish, CobbleShoal], ids=["dusk_crayfish", "cobble_shoal"]
)
def test_the_explainer_reaches_both_models_too(model):
    """
    Trial Bed is implemented, so arriving is not enough: it has to answer, and
    answer the same way for a sequence-contract model and a nested-node one.
    Nothing about either model is threaded through to get there.
    """
    stack = SupportStack.load(["trial_bed"], model)
    fired = np.zeros((T, len(NODES)), dtype=bool)
    accounts = stack.explanations({}, NODES, fired)["trial_bed"]

    assert len(accounts) == len(NODES)
    for account in accounts:
        assert account["verdict"] == "cannot_evaluate"
        assert account["cause"] is None
        assert "no readings" in account["explanation"]


@pytest.mark.parametrize(
    "model", [DuskCrayfish, CobbleShoal], ids=["dusk_crayfish", "cobble_shoal"]
)
def test_the_screen_reaches_both_models_too(model):
    stack = SupportStack.load(["settling_pool"], model)
    with pytest.raises(NotImplementedError, match="Settling Pool"):
        stack.admit({}, NODES, (T, len(NODES)))


def test_the_framework_names_no_model():
    """
    The base classes must not import a model, name one, or branch on one.
    Checked against the source so a future edit trips it.
    """
    import inspect

    forbidden = ("DuskCrayfish", "CobbleShoal", "Dusk_Crayfish", "Cobble_Shoal", "dusk_crayfish")
    for module in (base, registry):
        src = inspect.getsource(module)
        for token in forbidden:
            assert token not in src, f"{module.__name__} references the model {token}"
        assert "strawberrywatch.models" not in src, f"{module.__name__} imports from models"


def test_models_do_not_know_which_modules_exist():
    """
    A model declares what it already handles and nothing else. It must not
    import the support package or name a module class.

    Imports are read off the parsed tree rather than the text, because a
    comment saying which package reads BUILTIN_SUPPORT is documentation and a
    coupling is an import. The class-name check stays textual: naming a module
    class anywhere in a model is the coupling, whether it is imported or not.
    """
    import ast
    import inspect

    for model_module in (
        inspect.getmodule(DuskCrayfish),
        inspect.getmodule(CobbleShoal),
    ):
        src = inspect.getsource(model_module)
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any("support_modules" in m for m in imported), (
            f"{model_module.__name__} imports the support package: {sorted(imported)}"
        )
        for cls in ("TrialBed", "NewtworkRun", "SettlingPool", "SupportStack"):
            assert cls not in src, f"{model_module.__name__} names {cls}"


# Builtin collision


def test_each_model_declares_what_it_already_handles():
    assert DuskCrayfish.BUILTIN_SUPPORT == ("rain",)
    assert CobbleShoal.BUILTIN_SUPPORT == ()


def test_a_module_colliding_with_a_builtin_is_refused():
    """
    Dusk Crayfish applies a 2.0 rain multiplier inside anomaly_detector.py.
    A rain modulator on top would raise the bar twice, and both applications
    would be individually correct, so nothing in the arithmetic would complain.
    """
    rain_module = StubModulator("rain", 2.0)
    with pytest.raises(registry.SupportCollision, match="already handles"):
        registry.check_collision(rain_module, DuskCrayfish)

    # And it is allowed on the model that does not handle rain internally.
    registry.check_collision(rain_module, CobbleShoal)


def test_the_collision_check_runs_on_every_load():
    """
    In the loader, not in validate(), so a module that overrides validate
    cannot drop it.
    """
    import inspect

    src = inspect.getsource(registry.load)
    assert "check_collision" in src

    class RainDetector(base.SupportDetector):
        name = "rain"

        def score(self, window, nodes):
            return np.zeros(len(nodes))

    original = SUPPORT_REGISTRY.get("rain")
    SUPPORT_REGISTRY["rain"] = RainDetector
    try:
        with pytest.raises(registry.SupportCollision):
            registry.load(["rain"], DuskCrayfish)
        assert registry.load(["rain"], CobbleShoal)[0].name == "rain"
    finally:
        if original is None:
            del SUPPORT_REGISTRY["rain"]
        else:
            SUPPORT_REGISTRY["rain"] = original


def test_a_module_that_needs_a_contract_is_checked_against_it():
    """
    validate() reads what the model declares. It never asks which model it is,
    which is what keeps the same module working on both.
    """

    class NestedOnly(base.SupportDetector):
        name = "nested_only"
        requires_contract = "nested_node_batch"

        def score(self, window, nodes):
            return np.zeros(len(nodes))

    NestedOnly().validate(CobbleShoal)
    with pytest.raises(base.SupportError, match="nested_node_batch"):
        NestedOnly().validate(DuskCrayfish)
