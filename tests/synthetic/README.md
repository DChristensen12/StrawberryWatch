# Synthetic creek

Everything in this folder makes data. Nothing in it asserts anything. The tests
that consume it live one level up in `tests/`, and the split is there so that
reading `tests/` tells you what the project claims, while reading `tests/synthetic/`
tells you what the claims were measured on.

`creek_synthetic.py` builds a fault-free creek with known ground truth: five
sites on the real flow graph, per-site baselines near the levels the network
actually sits at, a diurnal cycle, noise, and optional spills that attenuate and
lag downstream along the edges. It then injects one of seven fault shapes into a
chosen node over a fixed window — `stuck` (frozen and the prediction target
frozen with it), `stale` (frozen history, live target), `partial` (frozen plus a
fraction of the node's own normal variance), `drift` (a ramp), `spike` (a step
change landing exactly on the scored observation), `decouple` (reflection about
the window mean, which preserves mean and variance exactly and inverts only the
correlation with siblings), and `slow_all` (every node ramping together). The
returned `truth` dict is the only thing scoring is allowed to compare against.
`nested_batch.py` turns one generated window into the batch dict the per-node
encoder reads; `baseline_sweep.py` runs the whole shape × node × magnitude grid
through any detector exposing `score(batch, nulls)`, so a model and a control
chart are compared by one code path rather than two.

Treat what comes out of here as secondary evidence. A synthetic fault is
controlled — you know which node broke, when, and by how much, which is what
makes ranking and false-alarm rates measurable at all — but the fault shapes are
ones somebody chose, and a detector can only be measured against faults that
were thought of. The primary evidence is `tests/events.yaml`: real labelled
events from the creek, scored in `tests/test_anomaly_detection.py`. Where the
two disagree, the real events win. This folder is load-bearing anyway and must
not be deleted: `tests/test_node_adapter.py` asserts that a synthetic window and
a real one produce identical node tensors, so the generator is what pins the
real ingestion path to a known-correct shape.
