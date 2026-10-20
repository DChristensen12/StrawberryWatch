# Trains and calibrates Flame Skimmer on the real archive

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_cobble_shoal_real import (  # noqa: E402
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_VAL_START,
    PLAUSIBLE_RANGE,
    WINDOW,
    ContextScaler,
    TempModel,
    _thin,
    assign_anchors,
    calibrate,
    drop_implausible,
    graph_tensors,
    hold_offline,
    split_mask,
    stack_batch,
)

from strawberrywatch import inventory as inv  # noqa: E402
from strawberrywatch.anomalies import channel_scoring as scoring  # noqa: E402
from strawberrywatch.ingest.raw_data_loader import load_archive_by_table  # noqa: E402
from strawberrywatch.models.Cobble_Shoal import (  # noqa: E402
    CANONICAL_VARS,
    CONTEXT_FEATURES,
    MAX_ANCHOR_STALENESS_STEPS,
    SITE_ORDER,
    build_node_registry,
)
from strawberrywatch.models.Flame_Skimmer import build_flame_skimmer  # noqa: E402
from strawberrywatch.paths import checkpoints_dir  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402

# Share of the optimiser steps spent warming beta from 0 to 1
BETA_WARMUP_FRACTION = 0.10

# Anchors per side used for the epistemic share diagnostic each epoch. Enough
# for a stable mean, few enough that it costs seconds rather than minutes.
SHARE_ANCHORS = 600


def build_corpus(args):
    """
    The training window, scaled on the training side only, with unseen nodes
    held offline. Same steps and helpers as train_cobble_shoal_real.main, so
    these are the anchors the Cobble Shoal clock model trained on. Returns
    (win, train_rows, train_anchors, val_anchors, offline, scaler).
    """
    inventory = inv.load()
    tables = load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    tables, dropped = drop_implausible(tables)
    as_of = max(f.index.max() for f in tables.values() if len(f))
    span = (pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"))

    def build(scaler):
        return nw.build_window(tables, *span, inventory=inventory, scaler=scaler, as_of=as_of)

    win = build(None)
    grid = win["grid"]
    print(f"corpus {grid[0]} .. {grid[-1]}  ({len(grid)} steps, {len(win['nodes'])} nodes)")
    if dropped:
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(dropped.items()))
        print(f"out-of-range readings dropped: {detail}")

    # refit the scaler on the training side, see train_cobble_shoal_real.main
    train_rows = split_mask(grid, args.split, args.val_start)
    raw = win["scaler"].inverse(win["values"])
    scaler = nw.NodeScaler().fit(raw[train_rows], win["target_mask"][train_rows])

    seen = win["target_mask"][train_rows].sum(axis=0)
    offline = [k for k in range(len(win["nodes"])) if seen[k] == 0]
    win = hold_offline(build(scaler), offline)
    if offline:
        print(f"untrained nodes held offline: {', '.join(win['nodes'][k].key for k in offline)}")

    anchors = nw.fault_free_anchors(win, 10**9, WINDOW, args.min_coverage)
    train_anchors, val_anchors = assign_anchors(anchors, train_rows)
    print(
        f"fault-free anchors: {len(anchors)}  ->  train {len(train_anchors)}, "
        f"val {len(val_anchors)} ({args.split} split)"
    )
    if not train_anchors or not val_anchors:
        raise SystemExit("empty split, move --val-start inside the corpus")
    return win, train_rows, train_anchors, val_anchors, offline, scaler


def supervised_targets(win, anchors):
    """
    How many (anchor, node) pairs the loss actually scores, which is N_train for
    the KL. Same mask as FlameSkimmer.loss: a target at the anchor, and a recent
    enough reading to predict a delta from.
    """
    a = np.asarray(anchors)
    mask = win["target_mask"][a] & (win["staleness"][a - 1] <= MAX_ANCHOR_STALENESS_STEPS)
    return int(mask.sum())


def beta_at(step, warmup_steps):
    """Linear from 0 at step 0 to 1 at warmup_steps, then 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def evaluate(net, win, anchors, graph, batch_size, seed):
    """
    Mean validation NLL and negative ELBO at beta = 1, per supervised target,
    as (val_nll, val_loss). The generator is fixed. The change between epochs
    is the weights and not the draw.
    """
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    nll_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(anchors), batch_size):
            chunk = anchors[i : i + batch_size]
            batch = stack_batch(win, chunk, graph)
            _loss, parts = net.loss(batch, beta=1.0, generator=gen)
            count = int((batch["target_mask"] & net.anchor_fresh(batch)).sum())
            nll_sum += parts["nll"] * count
            n += count
        kl_per_target = float(net.kl()) / net.n_train
    val_nll = nll_sum / max(n, 1)
    return val_nll, val_nll + kl_per_target


def epistemic_share(net, win, anchors, graph, batch_size):
    """epistemic / total for every supervised target, under the fixed serving noise."""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(anchors), batch_size):
            batch = stack_batch(win, anchors[i : i + batch_size], graph)
            pred = net.predictive(batch)
            mask = batch["target_mask"] & net.anchor_fresh(batch)
            out.append((pred.epistemic / pred.total)[mask].numpy())
    return np.concatenate(out) if out else np.array([])


def train(net, win, train_anchors, val_anchors, graph, args):
    """
    Adam with gradient clipping, beta warmup and train_cobble_shoal_real's early
    stopping. Returns {"best_val", "best_epoch", "history"}.
    """
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    steps_per_epoch = sum(
        1
        for i in range(0, len(train_anchors), args.batch)
        if len(train_anchors[i : i + args.batch]) >= 2
    )
    warmup_steps = int(round(BETA_WARMUP_FRACTION * steps_per_epoch * args.epochs))
    share_train = _thin(train_anchors, SHARE_ANCHORS)
    share_val = _thin(val_anchors, SHARE_ANCHORS)

    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    best_epoch, since, step = 0, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(train_anchors))
        t0 = time.time()
        sums = {"loss": 0.0, "nll": 0.0, "kl_term": 0.0}
        steps = 0
        for i in range(0, len(order), args.batch):
            chunk = [train_anchors[j] for j in order[i : i + args.batch]]
            if len(chunk) < 2:
                continue
            beta = beta_at(step, warmup_steps)
            opt.zero_grad()
            loss, parts = net.loss(stack_batch(win, chunk, graph), beta=beta, generator=gen)
            step += 1
            if not torch.isfinite(loss):
                print(f"  epoch {epoch}: non-finite loss, batch skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            sums["loss"] += float(loss.detach())
            sums["nll"] += parts["nll"]
            sums["kl_term"] += parts["kl"] / net.n_train
            steps += 1

        means = {k: v / max(steps, 1) for k, v in sums.items()}
        val_nll, val_loss = evaluate(net, win, val_anchors, graph, args.batch, args.seed)
        share_tr = epistemic_share(net, win, share_train, graph, args.batch)
        share_va = epistemic_share(net, win, share_val, graph, args.batch)
        sched.step(val_loss)
        record = {
            "epoch": epoch,
            "train_loss": means["loss"],
            "train_nll": means["nll"],
            "kl_per_target": means["kl_term"],
            "beta_end": beta_at(step, warmup_steps),
            "val_nll": val_nll,
            "val_loss": val_loss,
            "epistemic_share_train": float(np.mean(share_tr)) if share_tr.size else float("nan"),
            "epistemic_share_val": float(np.mean(share_va)) if share_va.size else float("nan"),
        }
        history.append(record)
        flag = ""
        if val_loss < best - args.min_delta:
            best, best_epoch, since = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            flag = "  *"
        else:
            since += 1
        print(
            f"  epoch {epoch:>3}  nll {record['train_nll']:8.4f}  kl/n {record['kl_per_target']:.4f}  "
            f"beta {record['beta_end']:.2f}  val nll {val_nll:8.4f}  val loss {val_loss:8.4f}  "
            f"epi share train {record['epistemic_share_train']:.4f} val {record['epistemic_share_val']:.4f}  "
            f"lr {opt.param_groups[0]['lr']:.2e}  [{time.time() - t0:.0f}s]{flag}"
        )
        if since >= args.patience:
            print(f"  no val improvement in {args.patience} epochs, stopping")
            break

    net.load_state_dict(best_state)
    return {
        "best_val": best,
        "best_epoch": best_epoch,
        "history": history,
        "warmup_steps": warmup_steps,
    }


def temp_paths(tag, checkpoint_dir=None):
    root = Path(checkpoint_dir or checkpoints_dir())
    return (
        root / f"flame_skimmer_temp_{tag}.pt",
        root / f"flame_skimmer_temp_{tag}_calibration.json",
    )


def load_temp_flame(tag, checkpoint_dir=None):
    """
    Loads a checkpoint this script wrote, as the same TempModel the Cobble Shoal
    loader returns. Raises if either file is missing.
    """
    weights, cal_path = temp_paths(tag, checkpoint_dir)
    for path in (weights, cal_path):
        if not path.exists():
            raise FileNotFoundError(
                f"no {path.name} in {path.parent}. Train it with "
                f"scripts/train_flame_skimmer.py --tag {tag}"
            )
    blob = torch.load(weights, map_location="cpu", weights_only=True)
    meta = blob["meta"]
    cal = json.loads(cal_path.read_text())
    net = build_flame_skimmer(
        len(SITE_ORDER),
        len(CANONICAL_VARS),
        int(meta["num_context"]),
        len(build_node_registry()[0]),
        int(meta["seed"]),
        int(meta["window"]),
        head_width=int(meta["head_width"]),
        prior_sigma=float(meta["prior_sigma"]),
        rho_init=float(meta["rho_init"]),
        n_train=int(meta["n_train_targets"]),
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
    ap.add_argument("--operating-q", type=float, default=1e-3)
    ap.add_argument("--head-width", type=int, default=32)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--rho-init", type=float, default=-5.0)
    ap.add_argument("--tag", default="clock", help="checkpoint suffix")
    args = ap.parse_args(argv)

    features = list(CONTEXT_FEATURES)
    win, _train_rows, train_anchors, val_anchors, offline, scaler = build_corpus(args)
    grid = win["grid"]

    train_anchors = _thin(train_anchors, args.max_train)
    val_anchors = _thin(val_anchors, args.max_val)
    n_train_targets = supervised_targets(win, train_anchors)
    print(
        f"training on {len(train_anchors)} anchors ({n_train_targets} supervised targets, "
        f"N_train for the KL), validating on {len(val_anchors)}\n"
    )

    torch.manual_seed(args.seed)
    net = build_flame_skimmer(
        len(SITE_ORDER),
        len(CANONICAL_VARS),
        len(features),
        len(build_node_registry()[0]),
        args.seed,
        WINDOW,
        head_width=args.head_width,
        prior_sigma=args.prior_sigma,
        rho_init=args.rho_init,
        n_train=n_train_targets,
    )
    graph = graph_tensors(win)
    head_params = sum(p.numel() for p in net.head.parameters())
    print(
        f"parameters: {sum(p.numel() for p in net.parameters()):,} ({head_params:,} in the Bayesian head)"
    )

    result = train(net, win, train_anchors, val_anchors, graph, args)
    print(f"\nbest val loss {result['best_val']:.4f} at epoch {result['best_epoch']}")

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

    weights, cal_path = temp_paths(args.tag)
    meta = {
        "seed": args.seed,
        "window": WINDOW,
        "context_features": features,
        "num_context": len(features),
        "head_width": args.head_width,
        "prior_sigma": args.prior_sigma,
        "rho_init": args.rho_init,
        "n_train_targets": n_train_targets,
        "beta_warmup_steps": result["warmup_steps"],
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
                "model": "flame_skimmer",
                "source_checkpoint": weights.name,
                "seed": args.seed,
                "channel_nulls": nulls.to_dict(),
                "pot": pot,
                "operating_q": args.operating_q,
                "z_q": z_q,
                "n_null_windows": len(_thin(train_anchors, args.null_windows)),
                "node_scaler": scaler.to_dict(),
                "context": {"features": features, "scaler": ContextScaler([]).to_dict()},
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
                "note": "Flame Skimmer trained on data/raw_data, clock context. Experimental, not shipped.",
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


if __name__ == "__main__":
    raise SystemExit(main())
