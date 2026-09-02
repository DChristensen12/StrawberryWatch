"""
Retrain Cobble Shoal on the real archive instead of the synthetic generator.

The shipped weights were fitted on generated windows, and refit_cobble_calibration
only moved the null and the threshold on top of them. This moves the weights.

Two variants, so the comparison means something:

    --context clock     hour_sin, hour_cos. The shipped contract, real corpus.
    --context weather   the same two plus five Open-Meteo channels.

Training one and reporting it against the synthetic checkpoint would confound
two changes at once (new corpus, new inputs), so run both and diff them.

    python scripts/train_cobble_shoal_real.py --context weather
    python scripts/train_cobble_shoal_real.py --context clock

--split chronological (the default) holds out the tail of the corpus, which is
the right shape for a detector deployed forward in time. On this archive it also
puts the dry season in training and most of the rain in validation, so
--split blocked alternates whole months when the question is about the weather
channels rather than about forward generalisation.

Writes checkpoints/cobble_shoal_temp_{tag}.pt and its calibration beside it,
where tag defaults to the context name. Use --tag to keep several around.
Nothing shipped is overwritten. Grade the result with:

    python scripts/run_audit_comparison.py --model weather

Caveat worth reading before trusting the weather variant: the shipped design
took weather out of the model on purpose (see Cobble_Shoal.CONTEXT_FEATURES and
anomalies/rain_gate.py) because rain handling lives in the web application and
the model has no weather at inference. A weather-fed model can only be served if
the serving path fetches Open-Meteo too. That is why these land under temp_.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from strawberrywatch import inventory as inv  # noqa: E402
from strawberrywatch.anomalies import channel_scoring as scoring  # noqa: E402
from strawberrywatch.ingest import weather_archive  # noqa: E402
from strawberrywatch.ingest.raw_data_loader import load_archive_by_table  # noqa: E402
from strawberrywatch.models.Cobble_Shoal import (  # noqa: E402
    CANONICAL_VARS,
    CONTEXT_FEATURES,
    SITE_ORDER,
    build_cobble_shoal,
    build_node_registry,
    build_site_matrix,
    build_variable_adjacency,
    inventory_matrix,
)
from strawberrywatch.paths import checkpoints_dir  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402

# loss() calls float() on tensors that still carry grad, just to build its
# reporting dict. Harmless, and one warning per step drowns the log.
warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")

WINDOW = 24
GRID_PER_HOUR = 60 // nw.GRID_MINUTES

# Every site in the model's roster reports across this span. north_fork_0 opens
# in Aug 2024 and scnf010 in 2021, but the two south forks and oxford do not
# start until June 2025, and a corpus that starts earlier is mostly holes.
DEFAULT_START = "2025-06-04"
DEFAULT_END = "2026-04-29"
# Chronological, not shuffled. A random split leaks: adjacent 15-min windows
# overlap by 23 of 24 steps, so a shuffled val set is the training set.
DEFAULT_VAL_START = "2026-02-01"

WEATHER_CHANNELS = ["rain_mm", "rain_6h", "rain_72h", "air_temp_c", "shortwave_radiation"]
# Rain is mostly zero with a long tail, so z-scoring it raw leaves a channel
# that is flat for weeks and then spikes past 40 sigma.
LOG1P_CHANNELS = ("rain_mm", "rain_6h", "rain_72h")

CONTEXT_SETS = {
    "clock": [],
    "weather": WEATHER_CHANNELS,
}


# Physical bounds on a Berkeley creek reading, in inventory variable names.
# Not a quality claim: -9999 is the only sentinel the inventory knows, and
# north_fork_0 logged 332 C and 113.6 C on 2026-04-22, which sail straight past
# it into the scaler. Wide enough that nothing real is clipped (the archive's
# actual extremes are 5.2..34.7 C, 0..1318 uS, -1420..1458 mm).
PLAUSIBLE_RANGE = {
    "temperature": (0.0, 40.0),
    "conductivity": (0.0, 5000.0),
    "floating_conductivity": (0.0, 5000.0),
    "depth": (-5000.0, 5000.0),
    "dissolved_oxygen": (-5.0, 200.0),
}


def drop_implausible(tables):
    """NaN out readings no creek produces, before they reach the regrid."""
    cleaned, dropped = {}, {}
    for name, df in tables.items():
        out = df.copy()
        for column, (lo, hi) in PLAUSIBLE_RANGE.items():
            if column not in out.columns:
                continue
            bad = out[column].notna() & ((out[column] < lo) | (out[column] > hi))
            if bad.any():
                dropped[f"{name}.{column}"] = int(bad.sum())
                out.loc[bad, column] = np.nan
        cleaned[name] = out
    return cleaned, dropped


def hold_offline(win, node_index):
    """
    Force a node to look like a dead sensor for the whole corpus.

    For a node the training span never observed there is no scale, so NodeScaler
    leaves it at mean 0 / std 1 and its raw value goes into the encoder
    untouched. footbridge.float_cond does that at 457, which is not a
    z-score. The model already has a word for a node it knows nothing about, so
    say that instead: zero value, staleness at the ceiling, nothing to score.
    """
    ceiling = float(len(win["grid"]))
    for k in node_index:
        win["values"][:, k] = 0.0
        win["staleness"][:, k] = ceiling
        win["target_val"][:, k] = 0.0
        win["target_mask"][:, k] = False
    return win


def weather_frame(grid, cache_dir=None):
    """
    The Open-Meteo channels on the model's grid, all of them trailing.

    rain_6h and rain_72h are backward-looking sums: storm intensity and how wet
    the catchment already was. A centred window would be lookahead, and the
    thing that makes conductivity dilution predictable is what already fell.
    """
    # Pulled from 3 days before the grid so rain_72h is a full sum at the first
    # step instead of ramping up out of nothing.
    warmup = pd.Timedelta(days=3)
    raw = weather_archive.load_weather(grid[0] - warmup, grid[-1], cache_dir=cache_dir)
    full = pd.date_range(grid[0] - warmup, grid[-1], freq=nw.GRID_FREQ, tz="UTC")
    on_grid = raw.reindex(full).ffill().bfill()

    out = pd.DataFrame(index=full)
    out["rain_mm"] = on_grid["rain_mm"]
    out["rain_6h"] = on_grid["rain_mm"].rolling(6 * GRID_PER_HOUR, min_periods=1).sum()
    out["rain_72h"] = on_grid["rain_mm"].rolling(72 * GRID_PER_HOUR, min_periods=1).sum()
    out["air_temp_c"] = on_grid["air_temp_c"]
    out["shortwave_radiation"] = on_grid["shortwave_radiation"]
    return out.reindex(grid)


class ContextScaler:
    """
    Z-score for the weather channels, log1p first on the rain ones.

    Same discipline as NodeScaler and for the same reason: fit on the training
    span, save it, load it. Refitting on the window being scored is how the
    conductivity mean walked out of training space last time.
    """

    def __init__(self, columns=(), mean=None, std=None):
        self.columns = list(columns)
        self.mean = mean
        self.std = std

    @staticmethod
    def _pre(frame, columns):
        out = frame[columns].to_numpy(dtype=np.float64).copy()
        for i, name in enumerate(columns):
            if name in LOG1P_CHANNELS:
                out[:, i] = np.log1p(np.clip(out[:, i], 0.0, None))
        return out

    def fit(self, frame, rows=None):
        x = self._pre(frame, self.columns)
        if rows is not None:
            x = x[rows]
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = np.maximum(x.std(axis=0), 1e-3).astype(np.float32)
        return self

    def transform(self, frame):
        if not self.columns:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return ((self._pre(frame, self.columns) - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {
            "columns": self.columns,
            "mean": [] if self.mean is None else self.mean.tolist(),
            "std": [] if self.std is None else self.std.tolist(),
            "log1p": [c for c in self.columns if c in LOG1P_CHANNELS],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["columns"],
            np.array(d["mean"], np.float32),
            np.array(d["std"], np.float32),
        )


def attach_context(win, extra, scaler):
    """
    Swap the window's context block for clock plus scaled weather.

    node_windows.add_context_features deliberately builds clock only, so this
    widens the array here rather than editing that. build_window's output is a
    plain dict and to_batch just slices context, so nothing else has to know.
    """
    if not scaler.columns:
        return win
    win["context"] = np.concatenate([win["context"], scaler.transform(extra)], axis=1).astype(
        np.float32
    )
    return win


def graph_tensors(win):
    """The batch entries that do not vary with the anchor. Built once, reused."""
    roster = win["roster"]
    node_site = win["node_site"]
    return {
        "node_site": node_site,
        "node_var": win["node_var"],
        "site_inventory": inventory_matrix(list(roster), roster),
        "a_var": build_variable_adjacency(node_site),
        "site_matrix": build_site_matrix(node_site, len(roster)),
        "nodes": win["nodes"],
    }


def stack_batch(win, anchors, graph, window=WINDOW):
    """to_batch for many anchors at once. Only predict_all_masked needs B == 1."""
    take = lambda key: np.stack([win[key][a - window : a] for a in anchors])  # noqa: E731
    stale = torch.from_numpy(take("staleness"))
    return {
        "values": torch.from_numpy(take("values")),
        "staleness": stale,
        "context": torch.from_numpy(take("context")),
        "target": torch.from_numpy(np.stack([win["target_val"][a] for a in anchors])),
        "target_mask": torch.from_numpy(np.stack([win["target_mask"][a] for a in anchors])),
        "obs_mask": stale < 1.0,
        **graph,
    }


def split_mask(grid, mode, val_start):
    """
    Boolean over the grid, True where a step belongs to the training side.

    chronological is the honest split for a detector that will be deployed
    forward in time, but on this corpus it also splits the seasons: training
    lands in the dry half of the year and most of the rain lands in val, which
    is exactly the wrong shape for judging weather channels. blocked alternates
    whole months so both sides span the dry season and the wet one.
    """
    if mode == "chronological":
        return np.asarray(grid < pd.Timestamp(val_start, tz="UTC"))
    month = np.asarray(grid.year) * 12 + np.asarray(grid.month)
    return (month - month.min()) % 2 == 0


def assign_anchors(anchors, train_rows, window=WINDOW):
    """
    Send each anchor to the side its whole window sits on, and drop the rest.

    An anchor straddling the boundary is scored on inputs from both sides, so it
    belongs to neither. With month blocks that costs 6 hours per boundary.
    """
    train, val = [], []
    for a in anchors:
        segment = train_rows[a - window : a + 1]
        if segment.all():
            train.append(a)
        elif not segment.any():
            val.append(a)
    return train, val


def evaluate(net, win, anchors, graph, batch_size, seed):
    """Mean val loss. Fixed generator so the LOO sampling is the same every epoch."""
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(anchors), batch_size):
            chunk = anchors[i : i + batch_size]
            loss, _ = net.loss(stack_batch(win, chunk, graph), generator=gen)
            total += float(loss) * len(chunk)
            n += len(chunk)
    return total / max(n, 1)


def train(net, win, train_anchors, val_anchors, graph, args):
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    best_epoch, since = 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(train_anchors))
        t0, running, steps = time.time(), 0.0, 0
        for i in range(0, len(order), args.batch):
            chunk = [train_anchors[j] for j in order[i : i + args.batch]]
            if len(chunk) < 2:
                continue
            opt.zero_grad()
            loss, _parts = net.loss(stack_batch(win, chunk, graph), generator=gen)
            if not torch.isfinite(loss):
                print(f"  epoch {epoch}: non-finite loss, batch skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            running += float(loss)
            steps += 1

        train_loss = running / max(steps, 1)
        val_loss = evaluate(net, win, val_anchors, graph, args.batch, args.seed)
        sched.step(val_loss)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        flag = ""
        if val_loss < best - args.min_delta:
            best, best_epoch, since = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            flag = "  *"
        else:
            since += 1
        print(
            f"  epoch {epoch:>3}  train {train_loss:8.4f}  val {val_loss:8.4f}  "
            f"lr {opt.param_groups[0]['lr']:.2e}  [{time.time() - t0:.0f}s]{flag}"
        )
        if since >= args.patience:
            print(f"  no val improvement in {args.patience} epochs, stopping")
            break

    net.load_state_dict(best_state)
    return {"best_val": best, "best_epoch": best_epoch, "history": history}


def channel_dicts(net, win, anchors, graph, label=""):
    """Per-window channel scores. channels() goes through predict_all_masked, so B is 1."""
    out, t0 = [], time.time()
    for n, anchor in enumerate(anchors, 1):
        chans = net.channels(stack_batch(win, [anchor], graph))
        out.append({k: np.asarray(v).ravel() for k, v in chans.items()})
        if n % 200 == 0:
            print(f"    {label}{n}/{len(anchors)}  [{time.time() - t0:.0f}s]")
    return out


def calibrate(net, win, fit_anchors, held_anchors, graph, q):
    """
    Fault-free nulls and POT for the weights that just came out of training.

    Fitted on the training span, empirical rate measured on the held-out one, so
    the rate reported is not the rate on the sample the fit saw. Channels are
    kept rather than recomputed: score() would run the whole forward again just
    to rebuild what the null fit already had.
    """
    fit_chans = channel_dicts(net, win, fit_anchors, graph, "null ")
    nulls = scoring.ChannelNulls().fit(
        {name: np.concatenate([c[name] for c in fit_chans]) for name in net.channel_names}
    )

    def combined(chans):
        parts = [
            np.asarray(scoring.combine(pvalues=nulls.pvalues(c), rule="fisher")).ravel()
            for c in chans
        ]
        s = np.concatenate(parts) if parts else np.array([])
        return s[np.isfinite(s)]

    fit_scores = combined(fit_chans)
    held_scores = combined(channel_dicts(net, win, held_anchors, graph, "held "))
    # q is in the set even when it is not one of the three standard rates, or
    # the artifact would carry a z_q at a rate nobody fitted.
    pot = {
        f"combined_fisher@{rate:g}": scoring.pot_diagnostics(
            fit_scores, q=rate, held_out=held_scores if held_scores.size else None
        )
        for rate in sorted({1e-3, 1e-4, 1e-5, q})
    }
    return nulls, pot, fit_scores, held_scores


class TempModel:
    """A retrained checkpoint plus everything needed to score a window with it."""

    def __init__(self, tag, net, nulls, calibration, meta, node_scaler, ctx_scaler):
        self.tag = tag
        self.net = net
        self.nulls = nulls
        self.calibration = calibration
        self.meta = meta
        self.node_scaler = node_scaler
        self.ctx_scaler = ctx_scaler
        self.offline = tuple(calibration["corpus"].get("offline_nodes", ()))
        self.features = calibration["context"]["features"]
        self.z_q = float(calibration["z_q"])
        self.operating_q = float(calibration["operating_q"])

    def prepare(self, win):
        """
        Put a freshly built window into the space these weights were trained in.

        Context first, then the nodes the training span never saw. Both were true
        of every window the model learned from, and a window missing either is
        scored by weights that were never shown anything like it.
        """
        if self.ctx_scaler.columns:
            win = attach_context(win, weather_frame(win["grid"]), self.ctx_scaler)
        idx = [k for k, node in enumerate(win["nodes"]) if node.key in self.offline]
        return hold_offline(win, idx) if idx else win


def temp_paths(tag, checkpoint_dir=None):
    root = Path(checkpoint_dir or checkpoints_dir())
    return root / f"cobble_shoal_temp_{tag}.pt", root / f"cobble_shoal_temp_{tag}_calibration.json"


def load_temp_model(tag, checkpoint_dir=None):
    """Load a checkpoint this script wrote. Raises if either half is missing."""
    weights, cal_path = temp_paths(tag, checkpoint_dir)
    for path in (weights, cal_path):
        if not path.exists():
            raise FileNotFoundError(
                f"no {path.name} in {path.parent}. Train it with "
                f"scripts/train_cobble_shoal_real.py --context {tag}"
            )
    blob = torch.load(weights, map_location="cpu", weights_only=True)
    meta = blob["meta"]
    cal = json.loads(cal_path.read_text())

    net = build_cobble_shoal(
        len(SITE_ORDER),
        len(CANONICAL_VARS),
        int(meta["num_context"]),
        len(build_node_registry()[0]),
        int(meta["seed"]),
        int(meta["window"]),
    )
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return TempModel(
        tag,
        net,
        scoring.ChannelNulls.from_dict(cal["channel_nulls"]),
        cal,
        meta,
        nw.NodeScaler.from_dict(cal["node_scaler"]),
        ContextScaler.from_dict(cal["context"]["scaler"]),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=sorted(CONTEXT_SETS), default="weather")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--val-start", default=DEFAULT_VAL_START)
    ap.add_argument("--split", choices=("chronological", "blocked"), default="chronological")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--max-val", type=int, default=3000)
    ap.add_argument("--null-windows", type=int, default=400)
    ap.add_argument("--operating-q", type=float, default=1e-4)
    ap.add_argument("--tag", default=None, help="checkpoint suffix, defaults to the context name")
    args = ap.parse_args(argv)

    tag = args.tag or args.context
    columns = CONTEXT_SETS[args.context]
    features = list(CONTEXT_FEATURES) + columns

    inventory = inv.load()
    tables = load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    tables, dropped = drop_implausible(tables)
    as_of = max(f.index.max() for f in tables.values() if len(f))
    span = (pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"))
    build = lambda scaler: nw.build_window(  # noqa: E731
        tables, *span, inventory=inventory, scaler=scaler, as_of=as_of
    )

    win = build(None)
    grid = win["grid"]
    print(f"corpus {grid[0]} .. {grid[-1]}  ({len(grid)} steps, {len(win['nodes'])} nodes)")
    if dropped:
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(dropped.items()))
        print(f"out-of-range readings dropped: {detail}")

    # The NodeScaler has to see training data only. build_window above fitted one
    # over the whole span, which puts val statistics into the training space, so
    # undo it and refit on the train side. inverse() is exact, and the fit reads
    # observed rows only, so the carried-forward fill it recovers never counts.
    train_rows = split_mask(grid, args.split, args.val_start)
    raw = win["scaler"].inverse(win["values"])
    scaler = nw.NodeScaler().fit(raw[train_rows], win["target_mask"][train_rows])

    seen = win["target_mask"][train_rows].sum(axis=0)
    offline = [k for k in range(len(win["nodes"])) if seen[k] == 0]
    win = hold_offline(build(scaler), offline)
    if offline:
        names = ", ".join(win["nodes"][k].key for k in offline)
        print(
            f"untrained nodes held offline: {names}\n"
            f"  never observed on the training side, so there is no scale for them "
            f"and nothing for the loss to learn from"
        )

    anchors = nw.fault_free_anchors(win, 10**9, WINDOW, args.min_coverage)
    train_anchors, val_anchors = assign_anchors(anchors, train_rows)
    where = args.val_start if args.split == "chronological" else "alternating months"
    print(
        f"fault-free anchors: {len(anchors)}  ->  train {len(train_anchors)}, "
        f"val {len(val_anchors)}, {len(anchors) - len(train_anchors) - len(val_anchors)} "
        f"straddling ({args.split} split, {where})"
    )
    if not train_anchors or not val_anchors:
        raise SystemExit("empty split, move --val-start inside the corpus")

    ctx_scaler = ContextScaler(columns)
    extra = None
    if columns:
        extra = weather_frame(grid)
        ctx_scaler.fit(extra, rows=np.array(train_anchors))
        win = attach_context(win, extra, ctx_scaler)
        wet = float((extra["rain_mm"].to_numpy()[train_anchors] > 0.1).mean())
        print(f"weather channels: {', '.join(columns)}  ({wet:.1%} of train anchors wet)")
    print(f"context features ({len(features)}): {', '.join(features)}")

    train_anchors = _thin(train_anchors, args.max_train)
    val_anchors = _thin(val_anchors, args.max_val)
    print(f"training on {len(train_anchors)} anchors, validating on {len(val_anchors)}\n")

    torch.manual_seed(args.seed)
    net = build_cobble_shoal(
        len(SITE_ORDER),
        len(CANONICAL_VARS),
        len(features),
        len(build_node_registry()[0]),
        args.seed,
        WINDOW,
    )
    graph = graph_tensors(win)
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    result = train(net, win, train_anchors, val_anchors, graph, args)
    print(f"\nbest val {result['best_val']:.4f} at epoch {result['best_epoch']}")

    print(f"\nfitting nulls on {min(args.null_windows, len(train_anchors))} fault-free windows")
    nulls, pot, fit_scores, held_scores = calibrate(
        net,
        win,
        _thin(train_anchors, args.null_windows),
        _thin(val_anchors, args.null_windows // 2),
        graph,
        args.operating_q,
    )
    z_q = float(pot[f"combined_fisher@{args.operating_q:g}"]["threshold"])

    weights = checkpoints_dir() / f"cobble_shoal_temp_{tag}.pt"
    cal_path = checkpoints_dir() / f"cobble_shoal_temp_{tag}_calibration.json"
    meta = {
        "seed": args.seed,
        "window": WINDOW,
        "context_features": features,
        "num_context": len(features),
        "roster": {k: list(v) for k, v in win["roster"].items()},
        "train_span": [str(grid[train_anchors[0]]), str(grid[train_anchors[-1]])],
        "val_span": [str(grid[val_anchors[0]]), str(grid[val_anchors[-1]])],
        "offline_nodes": [win["nodes"][k].key for k in offline],
        "split": args.split,
        "n_train": len(train_anchors),
        "n_val": len(val_anchors),
        "best_val": result["best_val"],
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "lr": args.lr,
        "batch": args.batch,
    }
    torch.save({"state_dict": net.state_dict(), "meta": meta}, weights)

    cal_path.write_text(
        json.dumps(
            {
                "model": "cobble_shoal",
                "source_checkpoint": weights.name,
                "seed": args.seed,
                "channel_nulls": nulls.to_dict(),
                "pot": pot,
                "operating_q": args.operating_q,
                "z_q": z_q,
                "n_null_windows": len(_thin(train_anchors, args.null_windows)),
                "node_scaler": scaler.to_dict(),
                "context": {"features": features, "scaler": ctx_scaler.to_dict()},
                "corpus": {
                    "start": str(grid[0]),
                    "end": str(grid[-1]),
                    "val_start": args.val_start,
                    "split": args.split,
                    "min_coverage": args.min_coverage,
                    "offline_nodes": [win["nodes"][k].key for k in offline],
                    "plausible_range": {k: list(v) for k, v in PLAUSIBLE_RANGE.items()},
                    "excluded_spans": list(nw.EVENT_SPANS),
                },
                "note": (
                    f"Weights retrained on data/raw_data with context={args.context}. "
                    f"Experimental: not the shipped Cobble Shoal artifact."
                ),
            },
            indent=1,
        )
    )

    print(f"\nz_q at q={args.operating_q:g}: {z_q:.4f}")
    for rate in (1e-3, 1e-4, 1e-5):
        d = pot[f"combined_fisher@{rate:g}"]
        print(
            f"  q={rate:<7g} threshold={d['threshold']:9.4f}  fitted={d['fitted']}  "
            f"empirical={d['empirical_rate']:.2e}  gamma={d['gamma']:+.3f}"
        )
    print(f"  null scores: n={fit_scores.size}, held-out n={held_scores.size}")
    print(f"\nwrote {weights}\nwrote {cal_path}")
    return 0


def _thin(items, count):
    """Evenly spaced subsample, so a cap keeps the whole span rather than its front."""
    if count is None or len(items) <= count:
        return list(items)
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


if __name__ == "__main__":
    raise SystemExit(main())
