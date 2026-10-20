"""
Measures what model checks cost on this server. The website: limits in
settings.yaml come from these numbers instead of guesses.

    python scripts/load_test.py
    python scripts/load_test.py --models dusk_crayfish cobble_shoal --event anomaly_2026_03_20_hydrant_nf0

Each run gets its own process, like the website's checks: one with just the
imports, one per model, and one with every model, each scoring a past event.
Run it on the SCRP test account (same machine as the live site), and again
after a server move or a new model.

It prints each run's cost and the website: block those numbers suggest. The
memory floor is the largest check plus at least half again, the model cap is
the most models that fit in the usual free memory above that floor, and the
time limit is half the 15 minute interval.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_EVENT = "anomaly_2025_06_12_spill_sf"
INTERVAL_MINUTES = 15


def child(models, event):
    """One measured run: load each model and score the replay with it, in this process."""
    began = time.time()
    import torch

    from strawberrywatch.watch import runner, sources

    torch.set_num_threads(1)
    replay = sources.ReplaySource(event)
    start, end = replay.window(None)
    weather = replay.weather(start, end)
    for name in models:
        detector = runner.load_detector(name)
        tables = replay.tables(detector.table_names, start, end)
        detector.scores(tables, weather)
    return {
        "models": models,
        "seconds": round(time.time() - began, 1),
        "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def measure(models, event):
    """One child process, and what it reported."""
    done = subprocess.run(
        [sys.executable, __file__, "--child", "--event", event, "--models", *models],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise SystemExit(f"the run with {models or 'no models'} failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def suggest(base, singles, together, free_mb):
    """The website: block the measurements suggest."""
    largest = max([together["peak_mb"], *(r["peak_mb"] for r in singles)])
    floor = largest * 1.5
    costs = sorted(max(0.0, r["peak_mb"] - base["peak_mb"]) for r in singles)
    cap, used = 0, base["peak_mb"]
    for cost in costs:
        if free_mb is None or used + cost > free_mb - floor:
            break
        used += cost
        cap += 1
    return {
        "check_timeout_minutes": INTERVAL_MINUTES / 2,
        "min_free_memory_mb": round(floor),
        "max_models_on": max(cap, 1) if free_mb is None else cap,
    }


def main(argv=None):
    from strawberrywatch.watch import settings as ws

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        print(json.dumps(child(args.models or [], args.event)), flush=True)
        return 0

    from strawberrywatch.serving import available

    models = args.models or available()
    free = ws.free_memory_mb()
    print(f"Server: {free:,.0f} MB free now" if free is not None else "Server: free memory unknown")
    base = measure([], args.event)
    print(f"  imports only: {base['seconds']:6.1f} s  {base['peak_mb']:8,.0f} MB")
    singles = []
    for name in models:
        run = measure([name], args.event)
        singles.append(run)
        print(f"  {name:<14}{run['seconds']:6.1f} s  {run['peak_mb']:8,.0f} MB")
    together = measure(models, args.event) if len(models) > 1 else singles[0]
    print(f"  all together: {together['seconds']:6.1f} s  {together['peak_mb']:8,.0f} MB")
    block = suggest(base, singles, together, free)
    print("\nSuggested website: block for settings.yaml:\n\nwebsite:")
    for key, value in block.items():
        print(f"  {key}: {value:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
