# Grades Flame Skimmer against the Cobble Shoal clock model on the same split

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_audit_comparison import event_tables  # noqa: E402
from train_cobble_shoal_real import (  # noqa: E402
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_VAL_START,
    WINDOW,
    _thin,
    graph_tensors,
    load_temp_model,
    stack_batch,
    weather_frame,
)
from train_flame_skimmer import build_corpus, load_temp_flame  # noqa: E402

from strawberrywatch import inventory as inv  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402
from tests import event_catalog  # noqa: E402

# The rate the group verified as the operating point.
EVAL_Q = 1e-3

# A step counts as rain when more than this fell over the trailing 6 hours.
# Same 0.1 mm cut train_cobble_shoal_real uses to call an anchor wet.
RAIN_6H_MM = 0.1

# Median epistemic share on val rain steps must beat train dry steps by this.
GATE_RATIO = 1.5

INTERVALS = (0.50, 0.90, 0.95)

EPISTEMIC_EVENTS = (
    "anomaly_2025_11_13_rain_nf1",
    "anomaly_2026_04_01_rainfall",
    "anomaly_2026_01_botanical_actuator",
)

BATCH = 128


def crps_gaussian(y_pred, y_std, y_true):
    """
    Closed form CRPS of a Gaussian, per point. Smaller is better.

    From crps_gaussian in uncertainty-toolbox, kept per point so it splits by node.

    returns np.ndarray, same shape as the inputs
    """
    z = (y_true - y_pred) / y_std
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    return y_std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def nll_gaussian(y_pred, y_std, y_true):
    """Gaussian negative log likelihood per point, with the constant, unlike gaussian_nll."""
    z = (y_true - y_pred) / y_std
    return 0.5 * z**2 + np.log(y_std) + 0.5 * math.log(2.0 * math.pi)


def _erf(x):
    return torch.erf(torch.as_tensor(x, dtype=torch.float64)).numpy()


def _z_for(level):
    # two sided normal quantile for a central interval, via the inverse error function
    return float(math.sqrt(2.0) * torch.erfinv(torch.tensor(level, dtype=torch.float64)))


def forecasts(model_name, net, win, anchors, graph):
    """
    Forecast mean, standard deviation and target on every supervised (anchor, node).

    Flame Skimmer reads its predictive total, Cobble Shoal its forecast head.

    returns dict of 1D arrays: pred, std, target, node, anchor, plus epistemic and total for Flame
    """
    out = {k: [] for k in ("pred", "std", "target", "node", "anchor", "epistemic", "total")}
    with torch.no_grad():
        for i in range(0, len(anchors), BATCH):
            chunk = anchors[i : i + BATCH]
            batch = stack_batch(win, chunk, graph)
            if model_name == "flame":
                p = net.predictive(batch)
                mu, std = p.mean, torch.sqrt(p.total)
                mask = batch["target_mask"] & net.anchor_fresh(batch)
                out["epistemic"].append(p.epistemic[mask].numpy())
                out["total"].append(p.total[mask].numpy())
            else:
                _h, mu, std, mask = net.forecast_terms(batch)
            rows, cols = torch.nonzero(mask, as_tuple=True)
            out["pred"].append(mu[mask].numpy())
            out["std"].append(std[mask].numpy())
            out["target"].append(batch["target"][mask].numpy())
            out["node"].append(cols.numpy())
            out["anchor"].append(np.asarray(chunk)[rows.numpy()])
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}


def forecast_table(f, node_names):
    """Per node NLL, CRPS, MAE and interval coverage, plus an all-nodes row."""
    rows = {}
    for k, name in [(None, "all")] + list(enumerate(node_names)):
        sel = np.ones_like(f["node"], dtype=bool) if k is None else f["node"] == k
        if not sel.any():
            continue
        pred, std, y = f["pred"][sel], f["std"][sel], f["target"][sel]
        z = np.abs(y - pred) / std
        rows[name] = {
            "n": int(sel.sum()),
            "nll": float(nll_gaussian(pred, std, y).mean()),
            "crps": float(crps_gaussian(pred, std, y).mean()),
            "mae": float(np.abs(y - pred).mean()),
            **{f"cov{int(level * 100)}": float((z <= _z_for(level)).mean()) for level in INTERVALS},
        }
    return rows


def node_scores(model_name, net, nulls, win, anchors, graph):
    """
    Combined Fisher score per (anchor, node), finite entries only.

    Cobble Shoal's channels go through predict_all_masked, which wants one
    window per batch, so it is scored one anchor at a time.

    returns (scores, anchor index per score)
    """
    scores, which = [], []
    step = BATCH if model_name == "flame" else 1
    for i in range(0, len(anchors), step):
        chunk = anchors[i : i + step]
        s = np.asarray(net.score(stack_batch(win, chunk, graph), nulls))
        for row, a in zip(s, chunk, strict=True):
            keep = np.isfinite(row)
            scores.append(row[keep])
            which.append(np.full(int(keep.sum()), a))
    return np.concatenate(scores), np.concatenate(which)


def threshold_at(calibration, q=EVAL_Q):
    return float(calibration["pot"][f"combined_fisher@{q:g}"]["threshold"])


def describe(shares, epistemic=None):
    """
    Quartiles of the epistemic share, and the median epistemic standard deviation when given.

    The share falls whenever aleatoric rises, so a storm that raises both can
    still lower it. The absolute value is printed beside it to tell those apart.
    """
    if shares.size == 0:
        return "n=0"
    q25, q50, q75 = np.percentile(shares, [25, 50, 75])
    text = (
        f"n={shares.size:6d}  median {q50:.4f}  iqr {q25:.4f}..{q75:.4f}  mean {shares.mean():.4f}"
    )
    if epistemic is not None:
        text += f"  epistemic sd median {np.median(np.sqrt(epistemic)):.4f}"
    return text


def score_event(model, folder, inventory, site=None):
    """
    Peak score at the graded node over every anchor in one event window, plus epistemic shares.

    returns dict with peak, peak_any, epistemic (Flame only) and rain flags
    """
    tables = event_tables(folder)
    spans = [df.index for df in tables.values() if len(df)]
    start = min(s.min() for s in spans).floor("15min")
    end = max(s.max() for s in spans).ceil("15min")
    win = nw.build_window(
        tables, start, end, inventory=inventory, scaler=model.node_scaler, as_of=end
    )
    win = model.prepare(win)
    anchors = list(range(WINDOW, len(win["grid"]) - 1))
    graph = graph_tensors(win)
    keys = [n.key for n in win["nodes"]]
    target = keys.index(f"{site}.conductivity") if site else None

    peak, peak_any = -np.inf, -np.inf
    step = BATCH if model.kind == "flame" else 1
    for i in range(0, len(anchors), step):
        s = np.asarray(model.net.score(stack_batch(win, anchors[i : i + step], graph), model.nulls))
        if np.isfinite(s).any():
            peak_any = max(peak_any, float(np.nanmax(s)))
        if target is not None and np.isfinite(s[:, target]).any():
            peak = max(peak, float(np.nanmax(s[:, target])))

    out = {"peak": peak, "peak_any": peak_any}
    if model.kind == "flame":
        f = forecasts("flame", model.net, win, anchors, graph)
        rain = weather_frame(win["grid"])["rain_6h"].to_numpy() > RAIN_6H_MM
        out["share"] = f["epistemic"] / f["total"]
        out["rain"] = rain[f["anchor"]]
    return out


def load_models(args):
    """Every Flame Skimmer tag asked for, then the Cobble Shoal it is compared against."""
    models = []
    for tag in [t for t in args.flame.split(",") if t]:
        temp = load_temp_flame(tag)
        temp.kind, temp.label = "flame", f"flame_skimmer[{tag}] seed {temp.meta['seed']}"
        models.append(temp)
    temp = load_temp_model(args.cobble)
    temp.kind, temp.label = "cobble", f"cobble_shoal[{args.cobble}] seed {temp.meta['seed']}"
    models.append(temp)
    return models


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--flame", default="clock", help="comma separated flame_skimmer_temp tags, one per seed"
    )
    ap.add_argument("--cobble", default="clock", help="cobble_shoal_temp tag to compare against")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--val-start", default=DEFAULT_VAL_START)
    ap.add_argument("--split", default="chronological")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--max-val", type=int, default=3000)
    args = ap.parse_args(argv)

    models = load_models(args)
    win, _rows, train_all, val_all, _offline, scaler = build_corpus(args)
    train_anchors = _thin(train_all, args.max_train)
    val_anchors = _thin(val_all, args.max_val)
    for m in models:
        # every model has to be reading the data in the space this window is in
        if not np.allclose(m.node_scaler.mean, scaler.mean) or not np.allclose(
            m.node_scaler.std, scaler.std
        ):
            raise SystemExit(f"{m.label} was scaled differently from this corpus, not comparable")
    graph = graph_tensors(win)
    node_names = [n.key for n in win["nodes"]]
    rain_6h = weather_frame(win["grid"])["rain_6h"].to_numpy() > RAIN_6H_MM
    print(
        f"val anchors {len(val_anchors)}, {rain_6h[val_anchors].mean():.1%} with rain in the previous 6 h; "
        f"train anchors {len(train_anchors)}, {rain_6h[train_anchors].mean():.1%}\n"
    )

    print("A  forecast quality on val (standardised units)")
    tables = {}
    for m in models:
        tables[m.label] = forecast_table(
            forecasts(m.kind, m.net, win, val_anchors, graph), node_names
        )
    header = f"  {'node':<26}{'model':<34}{'n':>6}{'nll':>9}{'crps':>8}{'mae':>8}{'cov50':>7}{'cov90':>7}{'cov95':>7}"
    print(header)
    for node in ["all"] + node_names:
        for label, rows in tables.items():
            r = rows.get(node)
            if r is None:
                continue
            print(
                f"  {node:<26}{label:<34}{r['n']:>6}{r['nll']:>9.4f}{r['crps']:>8.4f}{r['mae']:>8.4f}"
                f"{r['cov50']:>7.3f}{r['cov90']:>7.3f}{r['cov95']:>7.3f}"
            )

    print("\nB  epistemic share, epistemic / total, per supervised target")
    gate = {}
    for m in [m for m in models if m.kind == "flame"]:
        tr = forecasts("flame", m.net, win, train_anchors, graph)
        va = forecasts("flame", m.net, win, val_anchors, graph)
        share_tr, share_va = tr["epistemic"] / tr["total"], va["epistemic"] / va["total"]
        rain_tr, rain_va = rain_6h[tr["anchor"]], rain_6h[va["anchor"]]
        print(f"  {m.label}")
        print(f"    train all   {describe(share_tr, tr['epistemic'])}")
        print(f"    train dry   {describe(share_tr[~rain_tr], tr['epistemic'][~rain_tr])}")
        print(f"    train rain  {describe(share_tr[rain_tr], tr['epistemic'][rain_tr])}")
        print(f"    val all     {describe(share_va, va['epistemic'])}")
        print(f"    val dry     {describe(share_va[~rain_va], va['epistemic'][~rain_va])}")
        print(f"    val rain    {describe(share_va[rain_va], va['epistemic'][rain_va])}")
        for k, name in enumerate(node_names):
            sel_tr, sel_va = (tr["node"] == k) & ~rain_tr, (va["node"] == k) & rain_va
            if sel_tr.any() and sel_va.any():
                print(
                    f"      {name:<26} train dry median {np.median(share_tr[sel_tr]):.4f}  "
                    f"val rain median {np.median(share_va[sel_va]):.4f}"
                )
        base = np.median(share_tr[~rain_tr]) if (~rain_tr).any() else float("nan")
        wet = np.median(share_va[rain_va]) if rain_va.any() else float("nan")
        gate[m.label] = (wet, base, wet / base if base > 0 else float("nan"))
        for folder in EPISTEMIC_EVENTS:
            ev = score_event(m, folder, inv.load())
            print(f"    {folder:<36} all {describe(ev['share'])}")
            if ev["rain"].any():
                print(f"    {'':<36} rain {describe(ev['share'][ev['rain']])}")

    print(f"\nC  detection on events.yaml at q={EVAL_Q:g}")
    inventory = inv.load()
    events = event_catalog.load()
    for m in models:
        z_q = threshold_at(m.calibration)
        print(f"  {m.label}  z_q={z_q:.3f}")
        for ev in events:
            if ev.label == "true_negative" or not ev.path.exists():
                continue
            r = score_event(m, ev.folder, inventory, site=ev.site)
            crossed = r["peak"] > z_q
            verdict = ("PASS" if crossed else "MISS") if ev.label == "anomaly" else "reported"
            print(
                f"    {ev.group:<18}{ev.site:<14}{ev.label:<15}peak {r['peak']:9.3f}  "
                f"any node {r['peak_any']:9.3f}  {verdict}"
            )

    print(f"\nD  false alarm rate on fault-free val anchors at q={EVAL_Q:g}")
    for m in models:
        z_q = threshold_at(m.calibration)
        s, which = node_scores(m.kind, m.net, m.nulls, win, val_anchors, graph)
        wet = rain_6h[which]
        over = s > z_q
        print(
            f"  {m.label:<34} all {over.mean():.2e} (n={s.size})  dry {over[~wet].mean():.2e} (n={(~wet).sum()})  "
            f"rain {over[wet].mean() if wet.any() else float('nan'):.2e} (n={wet.sum()})  nominal {EVAL_Q:.0e}"
        )

    print(f"\nGATE  median epistemic share, val rain / train dry, needs at least {GATE_RATIO}x")
    for label, (wet, base, ratio) in gate.items():
        verdict = "passes" if ratio >= GATE_RATIO else "FAILS, run the ensemble reference"
        print(f"  {label}: {wet:.4f} / {base:.4f} = {ratio:.2f}x  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
