"""The winter verdict on Dusk Crayfish's storm-mode check: the mixing and depth-fit methods against the current model, fits frozen in checkpoints/dusk_crayfish/storm_mode/frozen.json."""

import json
import pickle
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WINTER = ROOT / "checkpoints" / "dusk_crayfish" / "storm_mode"
# Water Strider as served when the rule was frozen, the baseline for rain false alarms
WATER_STRIDER = ROOT / "checkpoints" / "water_strider" / "storm_mode"
WORK = ROOT / "data" / "processed_data" / "winter_verdict"
SITES = ("north_fork_0", "south_fork_2", "south_fork_1", "oxford")
CORE = ("north_fork_0", "south_fork_1", "south_fork_2")
COND, DEPTH, TEMP = "Meter_Hydros21_Cond", "Meter_Hydros21_Depth", "Meter_Hydros21_Temp"
STEP = pd.Timedelta(minutes=15)
WEEK = 7 * 96  # steps
SEEDS = range(5)
METHODS = ("current", "depth_fit", "mixing")
NAMES = {"current": "current model", "depth_fit": "depth fit", "mixing": "mixing"}
# Open-Meteo's name for each rain product. Every one counts when picking dry hours
RAIN_MODELS = {
    "hrrr": "best_match",
    "nbm": "ncep_nbm_conus",
    "gfs013": "ncep_gfs013",
    "ecmwf": "ecmwf_ifs025",
    "icon": "icon_seamless",
    "gem": "gem_seamless",
    "meteofrance": "meteofrance_seamless",
    "jma": "jma_seamless",
    "ukmo": "ukmo_seamless",
    "metno": "metno_seamless",
}
WET = 0.1  # mm in a quarter hour, the rain gate's floor

# Fake faults on conductivity, after tests/synthetic/creek_synthetic.py
TAPER = 4  # steps back to the real readings, one hour
SIZES = {
    "spike": (60.0, 120.0, 240.0),
    "drift": (60.0, 120.0, 240.0),
    "spill": (60.0, 120.0, 240.0),
    "partial": (0.5, 0.2, 0.05),
    "stuck": (12, 24, 48),
    "decouple": (12, 24, 48),
}
LENGTH = {"spike": 24, "drift": 48, "spill": 16, "partial": 32}  # steps before the taper
TRAVEL = 4  # steps from south_fork_1 to south_fork_2
DOWNSTREAM = {"south_fork_1": "south_fork_2"}
# readings rebuilt before a fault, for the band and the mode's state
HISTORY = pd.Timedelta(days=10)
GRADED_LABEL = "anomaly"


def load_frozen(path=None):
    return json.loads(Path(path or WINTER / "frozen.json").read_text())


# Readings, rain and weather


def pull_readings(lo, hi, sites=SITES):
    from strawberrywatch.ingest.api_client import fetch_creek_data
    from strawberrywatch.watch import sources

    raw = {}
    for site in sites:
        frame = fetch_creek_data(
            site, lo.to_pydatetime(), hi.to_pydatetime(), variables=[COND, DEPTH, TEMP], token=""
        )
        if frame.empty:
            continue
        frame = frame.drop(
            columns=[c for c in ("station_id", "uuid", "site_code") if c in frame.columns]
        )
        raw[site] = sources._indexed(frame, f"The API's answer for {site}")
        print(
            f"{site}: {len(raw[site])} readings from {raw[site].index.min()} to {raw[site].index.max()}"
        )
    return raw


def pull_rain(model, lo, hi):
    """Open-Meteo's 15 minute precipitation for Berkeley from one model, a month per request."""
    import requests

    from strawberrywatch.ingest import historical_weather_client as client

    parts = []
    for start in pd.date_range(
        lo.tz_localize(None).normalize().replace(day=1), hi.tz_localize(None), freq="MS"
    ):
        end = min(start + pd.offsets.MonthEnd(1), hi.tz_localize(None))
        params = {
            "latitude": client._BERKELEY_LAT,
            "longitude": client._BERKELEY_LON,
            "start_date": f"{start:%Y-%m-%d}",
            "end_date": f"{end:%Y-%m-%d}",
            "minutely_15": "precipitation",
            "timezone": "UTC",
            "models": model,
        }
        for attempt in range(5):
            try:
                r = requests.get(client._OPEN_METEO_URL, params=params, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Open-Meteo didn't answer for {model} from {start:%Y-%m-%d}")
        q = r.json().get("minutely_15", {})
        parts.append(
            pd.DataFrame({"time": q.get("time", []), "rain_mm": q.get("precipitation", [])})
        )
        time.sleep(1)
    df = pd.concat(parts)
    s = pd.Series(
        pd.to_numeric(df["rain_mm"], errors="coerce").to_numpy(),
        pd.to_datetime(df["time"], utc=True),
    )
    return s[~s.index.duplicated()]


def on_rain_grid(s, lo, hi):
    """mm in the quarter hour ending at each stamp, NaN unknown."""
    return s.reindex(pd.date_range(lo + STEP, hi, freq="15min"))


def pull_weather(lo, hi):
    from strawberrywatch.ingest import weather_archive

    return weather_archive.load_weather(lo - pd.Timedelta(days=3), hi)


def weather_with(weather, rain):
    w = weather.copy()
    w["rain_mm"] = rain.reindex(w.index)
    return w


def screen(raw, lo, hi):
    """Dusk Crayfish's and Water Strider's screened tables, and each core site's depth flags."""
    from strawberrywatch import inventory as inv
    from strawberrywatch.ingest import levels
    from strawberrywatch.watch import runner

    flags = levels.screen(raw, inv.load(), lo, hi, hi.timestamp())
    dusk = runner._screened(raw, flags, runner._policy("dusk_crayfish"))
    ws = runner._screened(raw, flags, runner._policy("water_strider"))
    depth = {s: flags[s][DEPTH]["flag"] for s in CORE if s in flags and DEPTH in flags[s]}
    return dusk, ws, depth


# Real events and storm groups


def real_events(lo, hi, field_log_path=None):
    """Graded and true-negative events in [lo, hi]: tests/events.yaml's, and the field log's confirmed discharges."""
    import yaml

    from strawberrywatch import field_log

    spans = []
    for e in yaml.safe_load((ROOT / "tests" / "events.yaml").read_text())["events"]:
        df = pd.read_csv(ROOT / "data" / "anomalies" / e["folder"] / f"{e['site']}.csv")
        col = next(c for c in ("DateTimeUTC", "timestamp", "datetime", "time") if c in df.columns)
        t = pd.to_datetime(df[col], utc=True)
        spans.append(
            {
                "group": e["group"],
                "site": e["site"],
                "label": e["label"],
                "start": t.min(),
                "end": t.max(),
            }
        )
    log = field_log.load(field_log_path)
    for row in log[(log["event"] == "discharge") & log["confirmed"]].itertuples():
        end = row.end if pd.notna(row.end) else field_log.day_end(row.start)
        spans.append(
            {
                "group": f"field_log_{row.line}",
                "site": row.site,
                "label": GRADED_LABEL,
                "start": row.start,
                "end": end,
            }
        )
    return [e for e in spans if e["end"] >= lo and e["start"] <= hi]


def creek_series(tables):
    out = {}
    for s in CORE:
        if s not in tables:
            continue
        f = tables[s]
        grid = pd.date_range(
            f.index.min().floor("15min"), f.index.max().ceil("15min"), freq="15min"
        )
        f = f[[COND, DEPTH]].set_axis(["cond", "depth"], axis=1)
        out[s] = f.groupby(f.index.floor("15min")).mean().reindex(grid)
    return out


def responding(f, z):
    """Conductivity under 90% of its 12 h median an hour ago, and depth up on its 12 h low by the trigger."""
    base = f["cond"].rolling("12h", min_periods=16).median().shift(4)
    low = f["depth"].rolling("12h", min_periods=16).min().shift(4)
    rise = f["depth"] - low >= np.maximum(z["trigger_mm"], z["trigger_share"] * low.abs())
    return (f["cond"] < 0.9 * base) & rise


def onsets(flag, merge_hours=6):
    hits = flag.index[flag.fillna(False).to_numpy(dtype=bool)]
    if len(hits) == 0:
        return pd.DatetimeIndex([])
    keep = np.concatenate([[True], (hits[1:] - hits[:-1]) > pd.Timedelta(hours=merge_hours)])
    return hits[keep]


def creek_answers(data, z, within_hours=6):
    """Times every core site with readings answered within 6 h of each other, at least 2 sites."""
    starts = {s: onsets(responding(f, z)) for s, f in data.items()}
    events = []
    for t in (
        pd.DatetimeIndex([])
        .append([starts.get(s, pd.DatetimeIndex([])) for s in CORE])
        .sort_values()
    ):
        if events and t - events[-1] < pd.Timedelta(hours=within_hours):
            continue
        live = [
            s
            for s, f in data.items()
            if f["cond"]
            .reindex([t], method="nearest", tolerance=pd.Timedelta(hours=1))
            .notna()
            .all()
        ]
        near = [
            s
            for s in live
            if (
                (starts[s] >= t - pd.Timedelta(hours=within_hours))
                & (starts[s] <= t + pd.Timedelta(hours=within_hours))
            ).any()
        ]
        if len(live) >= 2 and set(near) == set(live):
            events.append(t)
    return events


def storm_groups(tables, events, z, lo, hi):
    """Creek answers in [lo, hi], each from 3 h before to 30 h after, overlapping ones joined."""
    groups = []
    for t in creek_answers(creek_series(tables), z):
        if not lo <= t <= hi:
            continue
        a, b = t - pd.Timedelta(hours=3), t + pd.Timedelta(hours=30)
        if groups and a <= groups[-1]["end"]:
            groups[-1]["end"] = b
            groups[-1]["answers"].append(t)
        else:
            groups.append({"start": a, "end": b, "answers": [t]})
    for g in groups:
        g["known_event"] = [
            e["group"] for e in events if e["start"] <= g["end"] and e["end"] >= g["start"]
        ]
    return groups


def where(stamps, groups, events):
    """Group index for each stamp, -1 for dry, -2 within a day of a known event."""
    stamps = pd.DatetimeIndex(stamps)
    out = np.full(len(stamps), -1)
    for i, g in enumerate(groups):
        out[(stamps >= g["start"]) & (stamps <= g["end"])] = i
    for e in events:
        out[
            (stamps >= e["start"] - pd.Timedelta(days=1))
            & (stamps <= e["end"] + pd.Timedelta(days=1))
            & (out < 0)
        ] = -2
    return out


# Fake faults


def fault_length(shape, size):
    return int(size) if shape in ("stuck", "decouple") else LENGTH[shape]


def blend(real, fake):
    """fake for the body, then a straight hour back to real."""
    out = fake.copy()
    n = len(real)
    w = np.linspace(1.0, 0.0, TAPER + 2)[1:-1]
    out[n - TAPER :] = w * fake[n - TAPER :] + (1 - w) * real[n - TAPER :]
    return out


def apply_fault(frame, start, shape, size, sign, rng, scale=1.0):
    body = fault_length(shape, size)
    stamps = pd.date_range(start, periods=body + TAPER, freq=STEP)
    f = frame.copy()
    rows = f.index.floor("15min").isin(stamps) & f[COND].notna().to_numpy()
    if rows.sum() < body // 2:
        return None, stamps
    real = f.loc[rows, COND].to_numpy(dtype=float)
    k = np.arange(len(real))
    if shape in ("spike", "spill"):
        fake = real + sign * size * scale
    elif shape == "drift":
        fake = real + sign * size * np.minimum(k / max(body - 1, 1), 1.0)
    elif shape == "stuck":
        fake = np.full_like(real, real[0])
    elif shape == "partial":
        fake = real[0] + rng.normal(0.0, np.std(real) * size, len(real))
    elif shape == "decouple":
        fake = 2.0 * real.mean() - real
    f.loc[rows, COND] = blend(real, fake)
    return f, stamps


def inject(tables, node, start, shape, size, sign=1, seed=0):
    """Faulted copies of the tables and {site: on stamps}, or (None, None) with too few readings."""
    rng = np.random.default_rng(seed)
    out = dict(tables)
    f, on = apply_fault(tables[node], start, shape, size, sign, rng)
    if f is None:
        return None, None
    out[node] = f
    marks = {node: on}
    if shape == "spill" and node in DOWNSTREAM and DOWNSTREAM[node] in tables:
        down = DOWNSTREAM[node]
        g, on2 = apply_fault(tables[down], start + TRAVEL * STEP, shape, size, sign, rng, scale=0.5)
        if g is not None:
            out[down] = g
            marks[down] = on2
    return out, marks


def placements(tables, groups, events, wet, seed=0):
    """Where each fake fault goes: 18 per storm group, each matched by one at a random dry hour."""
    rng = np.random.default_rng(seed)
    have = {s: tables[s][COND].notna().groupby(tables[s].index.floor("15min")).any() for s in CORE}

    def live(node, t, steps):
        idx = pd.date_range(t - pd.Timedelta(hours=36), t + steps * STEP, freq="15min")
        return have[node].reindex(idx).fillna(False).mean() > 0.8

    # dry: no model wet from 60 h before to 24 h after, no creek answer near, no known event
    near_wet = wet.astype(int)[::-1].rolling(96 + 1, min_periods=1).max()[::-1].astype(
        bool
    ) | wet.astype(int).rolling(240 + 1, min_periods=1).max().astype(bool)
    pool = near_wet.index[~near_wet.to_numpy()]
    for g in groups:
        pool = pool[
            (pool < g["start"] - pd.Timedelta(hours=60))
            | (pool > g["end"] + pd.Timedelta(hours=24))
        ]
    for e in events:
        pool = pool[
            (pool < e["start"] - pd.Timedelta(days=1)) | (pool > e["end"] + pd.Timedelta(days=1))
        ]
    pool = pool[pool.minute == 0]
    rows, k = [], 0
    for gi, g in enumerate(groups):
        for shape in SIZES:
            for si, size in enumerate(SIZES[shape]):
                steps = fault_length(shape, size) + TAPER
                sign = 1 if k % 2 == 0 else -1
                # storm: the fault starts in the creek's answer, at a site with readings
                t0 = g["answers"][0] - pd.Timedelta(hours=1) + int(rng.integers(0, 13)) * STEP
                nodes = (
                    ["south_fork_1"]
                    if shape == "spill"
                    else list(CORE[k % 3 :]) + list(CORE[: k % 3])
                )
                node = next((n for n in nodes if live(n, t0, steps)), None)
                if node is not None:
                    rows.append(
                        {
                            "id": f"s{k}",
                            "kind": "storm",
                            "group": gi,
                            "node": node,
                            "shape": shape,
                            "size_i": si,
                            "size": size,
                            "sign": sign,
                            "start": t0,
                        }
                    )
                for _ in range(50):
                    if not len(pool):
                        break
                    t1 = pool[int(rng.integers(0, len(pool)))]
                    node = next((n for n in nodes if live(n, t1, steps)), None)
                    if node is not None:
                        rows.append(
                            {
                                "id": f"d{k}",
                                "kind": "dry",
                                "group": -1,
                                "node": node,
                                "shape": shape,
                                "size_i": si,
                                "size": size,
                                "sign": sign,
                                "start": t1,
                            }
                        )
                        break
                k += 1
    return pd.DataFrame(
        rows, columns=["id", "kind", "group", "node", "shape", "size_i", "size", "sign", "start"]
    )


# The models


def load_dusk(folder):
    from strawberrywatch.serving import detector_class

    det = detector_class("dusk_crayfish").load(str(folder), device="cpu")
    # the gate is applied offline, so the detector judges against its dry lines
    det.rain_params = {"rain_threshold_multiplier": 1.0}
    return det


def dusk_scores(det, tables, weather, lo, hi, chunk_days=30, overlap_days=2):
    """[site, time, score, line] at every judged step, in chunks that overlap so no window is cut."""
    rows, start = [], lo
    while start < hi:
        end = min(start + pd.Timedelta(days=chunk_days), hi)
        a = start - pd.Timedelta(days=overlap_days)
        part = {s: f[(f.index >= a) & (f.index < end)] for s, f in tables.items()}
        out = det.scores(
            part, weather[(weather.index >= a - pd.Timedelta(days=2)) & (weather.index < end)]
        )
        for sc in out:
            if sc.rule == "forecast_residual" and start <= sc.time < end:
                rows.append((sc.site, sc.time, sc.score, sc.threshold))
        start = end
    df = pd.DataFrame(rows, columns=["site", "time", "score", "line"])
    return df.drop_duplicates(["site", "time"]).sort_values(["site", "time"]).reset_index(drop=True)


_POOL = {}


def _pool_init(folder, tables, weather):
    import torch

    torch.set_num_threads(1)
    _POOL.update(det=load_dusk(folder), tables=tables, weather=weather)


def _one_fault(row):
    tables, det, w = _POOL["tables"], _POOL["det"], _POOL["weather"]
    if row["node"] not in tables:
        return row["id"], None
    faulted, marks = inject(
        tables,
        row["node"],
        row["start"],
        row["shape"],
        row["size"],
        row["sign"],
        seed=zlib.crc32(row["id"].encode()),
    )
    if faulted is None:
        return row["id"], None
    lo = row["start"] - pd.Timedelta(hours=36)
    hi = max(m[-1] for m in marks.values()) + pd.Timedelta(hours=2)
    part = {s: f[(f.index >= lo) & (f.index < hi)] for s, f in faulted.items()}
    on_steps = set().union(*[set(m) for m in marks.values()])
    out = {}
    for sc in det.scores(part, w[(w.index >= lo - pd.Timedelta(days=2)) & (w.index < hi)]):
        # every site at the fault's steps
        if sc.rule == "forecast_residual" and sc.time in on_steps:
            out.setdefault(sc.site, []).append((sc.time, sc.score, sc.threshold))
    frames = {s: pd.DataFrame(v, columns=["time", "score", "line"]) for s, v in out.items()}
    return row["id"], {"marks": {s: list(m) for s, m in marks.items()}, "sites": frames}


def fault_scores(folder, tables, weather, place, workers=4):
    with Pool(workers, initializer=_pool_init, initargs=(folder, tables, weather)) as pool:
        return dict(pool.map(_one_fault, place.to_dict("records"), chunksize=8))


def water_strider_scores(folder, tables, weather, lo, hi):
    """Water Strider as served (decay gate, x2), every score kept."""
    from strawberrywatch.serving import detector_class

    det = detector_class("water_strider").cached(str(folder), device="cpu")
    det.set_rain_gate({"mode": "decay", "multiplier": 2.0})
    if hasattr(det, "set_sensitivity"):
        det.set_sensitivity(None, None)
    rows, start = [], lo
    while start < hi:
        end = min(start + pd.Timedelta(days=30), hi)
        a = start - pd.Timedelta(days=2)
        part = {
            s: f[(f.index >= a) & (f.index < end)]
            for s, f in tables.items()
            if s in det.table_names
        }
        for sc in det.scores(
            part, weather[(weather.index >= a - pd.Timedelta(days=2)) & (weather.index < end)]
        ):
            if start <= sc.time < end:
                rows.append(
                    (sc.site, sc.variable, sc.rule, sc.time, sc.score, sc.threshold, sc.base)
                )
        start = end
    df = pd.DataFrame(rows, columns=["site", "variable", "rule", "time", "score", "line", "base"])
    return df.drop_duplicates(["site", "variable", "rule", "time"])


# The rain gate and alerts


def gate(times, rain, z):
    """Production's x2 gate on final rain: full while the last 12 h hold rain, then a 36 h taper. Unknown rain keeps 1."""
    from strawberrywatch.anomalies.anomaly_detector import rain_known

    multiplier, lookback, taper, wet = (
        z["gate_multiplier"],
        z["gate_lookback_hours"],
        z["gate_taper_hours"],
        z["gate_wet_mm"],
    )
    times = pd.DatetimeIndex(times)
    out = np.ones(len(times))
    if len(times) == 0:
        return out, out.astype(bool)
    r = rain.groupby(rain.index.floor("15min")).first()
    grid = pd.date_range(
        (times.min() - pd.Timedelta(hours=lookback + taper + 1)).floor("15min"),
        times.max().ceil("15min"),
        freq="15min",
    )
    r = r.reindex(grid)
    filled = r.fillna(0.0).to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    hi = np.searchsorted(grid, times.floor("15min"), side="right")
    lo = np.searchsorted(grid, (times - pd.Timedelta(hours=lookback)).ceil("15min"), side="left")
    recent = np.round(csum[hi] - csum[lo], 9)
    known = rain_known(times, rain, lookback)
    full = known & (recent > wet)
    wet_at = np.where(filled > wet, np.arange(len(grid)), -1)
    last = np.maximum.accumulate(wet_at)
    before = np.clip(lo - 1, 0, len(grid) - 1)
    idx = np.where(lo > 0, last[before], -1)
    has = (idx >= 0) & (grid[np.clip(idx, 0, None)] >= times - pd.Timedelta(hours=lookback + taper))
    hours = (times - grid[np.clip(idx, 0, None)]).total_seconds().to_numpy() / 3600 - lookback
    frac = np.clip(hours / taper, 0, 1)
    tapered = 1 + (multiplier - 1) * (1 - frac)
    out = np.where(known & has & ~full, tapered, out)
    out = np.where(full, multiplier, out)
    return out, out > 1.0


def gate_cols(df, rain, z):
    if "mult" in df.columns:
        return df
    times = pd.DatetimeIndex(sorted(df["time"].unique()))
    m, f = gate(times, rain, z)
    return df.merge(pd.DataFrame({"time": times, "mult": m, "flag": f}), on="time", how="left")


def alerts(times, over, min_steps=3, lookback_days=2, repeat_hours=6.0):
    """What a check every 15 minutes raises: [(check time, first step over)], as decide.notice_window does."""
    times = pd.DatetimeIndex(times)
    over = np.asarray(over, dtype=bool)
    if not over.any():
        return []
    hits = times[over]
    checks = pd.date_range(
        hits[0].floor("15min"),
        min(hits[-1] + pd.Timedelta(days=lookback_days), times[-1]).ceil("15min"),
        freq=STEP,
    )
    lo = np.searchsorted(hits, checks - pd.Timedelta(days=lookback_days), side="left")
    count = np.searchsorted(hits, checks, side="right") - lo
    out, prev_first, prev_check = [], None, None
    for j in np.flatnonzero(count >= max(1, min_steps)):
        c, fo = checks[j], hits[lo[j]]
        if prev_first is not None and fo == prev_first:
            continue
        if prev_check is not None and (c - prev_check) < pd.Timedelta(hours=repeat_hours):
            continue
        out.append((c, fo))
        prev_first, prev_check = fo, c
    return out


def ws_rates(ws, sites, groups, events):
    """Water Strider's alerts a sensor-week in rain-only storm groups and in dry weather, at these sites."""
    ws = ws[ws["site"].isin(sites)]
    usable = np.array([not g["known_event"] for g in groups], dtype=bool)
    storm_n, dry_n, storm_w, dry_w = 0, 0, 0.0, 0.0
    for (_, _), f in ws.groupby(["site", "variable"]):
        stamps = pd.DatetimeIndex(sorted(f["time"].unique()))
        w = where(stamps, groups, events)
        storm_w += ((w >= 0) & np.array([x >= 0 and usable[x] for x in w], dtype=bool)).sum() / WEEK
        dry_w += (w == -1).sum() / WEEK
        for _, g in f.groupby("rule"):
            g = g.sort_values("time")
            for _, fo in alerts(g["time"], (g["score"] > g["line"]).to_numpy(), min_steps=1):
                x = where([fo], groups, events)[0]
                if x >= 0 and usable[x]:
                    storm_n += 1
                elif x == -1:
                    dry_n += 1
    return {
        "storm_rate": storm_n / storm_w if storm_w else np.nan,
        "dry_rate": dry_n / dry_w if dry_w else np.nan,
        "storm_alerts": storm_n,
        "storm_weeks": storm_w,
    }


# Storm mode: the switch


def depth_mode(d, z):
    """On at a depth rise of the trigger on the 12 h low an hour ago, held until depth is back in its 7-day band."""
    low = (
        d.rolling(f"{z['low_hours']}h", min_periods=z["low_min_steps"]).min().shift(z["lag_steps"])
    )
    up = (d - low >= np.maximum(z["trigger_mm"], z["trigger_share"] * low.abs())).to_numpy()
    base = (
        d.rolling(f"{z['band_days']}D", min_periods=z["band_min_steps"])
        .quantile(z["band_quantile"])
        .shift(z["lag_steps"])
    )
    band = (base + np.maximum(z["trigger_mm"], z["trigger_share"] * base.abs())).to_numpy()
    on = np.zeros(len(d), dtype=bool)
    state, missing = False, 0
    vals = d.to_numpy()
    for i in range(len(d)):
        if np.isnan(vals[i]):
            missing += 1
            if missing > z["missing_steps"]:
                state = False
            on[i] = state
            continue
        missing = 0
        if not state and up[i]:
            state = True
        elif state and np.isfinite(band[i]) and vals[i] < band[i]:
            state = False
        on[i] = state
    return pd.Series(on, d.index)


def trigger(tables, z):
    out = {}
    for s, f in tables.items():
        d = f[DEPTH].groupby(f.index.floor("15min")).mean()
        out[s] = depth_mode(d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="15min")), z)
    return out


def depth_passes(flags, z):
    """Depth flagged pass or suspect by the screen, on the grid."""
    ok = pd.Series(np.isin(flags.to_numpy(), z["depth_passes"]), flags.index)
    return ok.groupby(ok.index.floor("15min")).all()


def grid(tables, sites=CORE):
    out = {}
    for s in sites:
        if s not in tables:
            continue
        f = tables[s][[COND, DEPTH]].apply(pd.to_numeric, errors="coerce")
        f = f.groupby(f.index.floor("15min")).mean()
        out[s] = f.reindex(pd.date_range(f.index.min(), f.index.max(), freq="15min"))
    return out


def state(g, ok, mode, gate_flag, z):
    """Storm mode (trigger and gate), depth ok, limb (1 rising, 0 falling), log C and log D at each step."""
    d, c = g[DEPTH], g[COND]
    on = mode.reindex(g.index).fillna(False).to_numpy(dtype=bool) & gate_flag.reindex(
        g.index
    ).fillna(False).to_numpy(dtype=bool)
    earlier = d.shift(z["limb_steps"])
    limb = np.where(d.notna() & earlier.notna(), (d > earlier).astype(float), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(c.where(c > 0))
        logd = np.log(d.where(d > 0))
    return pd.DataFrame(
        {
            "mode": on,
            "ok": ok.reindex(g.index).fillna(False).to_numpy(dtype=bool),
            "limb": limb,
            "logc": logc,
            "logd": logd,
        },
        index=g.index,
    )


def episodes(mode):
    """Position of each step's episode start, -1 outside storm mode."""
    mode = np.asarray(mode, dtype=bool)
    first = mode & ~np.concatenate([[False], mode[:-1]])
    start = np.maximum.accumulate(np.where(first, np.arange(len(mode)), -1))
    return np.where(mode, start, -1)


def pre_median(values, starts, pre):
    out = {}
    for p in np.unique(starts[starts >= 0]):
        w = values[max(p - pre, 0) : max(p, 0)]
        w = w[np.isfinite(w)]
        out[p] = np.median(w) if len(w) else np.nan
    return out


def frame(g, ok, mode, gate_flag, z):
    st = state(g, ok, mode, gate_flag, z)
    c, d = g[COND].to_numpy(dtype=float), g[DEPTH].to_numpy(dtype=float)
    start = episodes(st["mode"].to_numpy())
    cp, dp = pre_median(c, start, z["pre_steps"]), pre_median(d, start, z["pre_steps"])
    with np.errstate(divide="ignore", invalid="ignore"):
        lcpre = np.log(np.array([cp.get(p, np.nan) if p >= 0 else np.nan for p in start]))
        ldpre = np.log(np.array([dp.get(p, np.nan) if p >= 0 else np.nan for p in start]))
    return st.assign(c=np.where(c > 0, c, np.nan), start=start, lcpre=lcpre, ldpre=ldpre)


def clean_states(tables, ok, mode, gate_flag, z):
    g = grid(tables)
    lo, hi = min(f.index.min() for f in g.values()), max(f.index.max() for f in g.values())
    index = pd.date_range(lo, hi, freq="15min")
    return {s: frame(f.reindex(index), ok[s], mode[s], gate_flag, z) for s, f in g.items()}, index


# Storm mode: the expectation and score


def m_hat(x, lcpre, c_ro, beta):
    """log(C_hat / C_pre) for x = log(D / D_pre), two-component mixing."""
    r = np.clip(c_ro / np.exp(lcpre), 0.0, None)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.log(r + (1.0 - r) * np.exp(-beta * x))


def residual(method, st, pos, fit):
    y = st["logc"].to_numpy()[pos] - st["lcpre"].to_numpy()[pos]
    x = st["logd"].to_numpy()[pos] - st["ldpre"].to_numpy()[pos]
    if method == "mixing":
        if fit["c_ro"] is None:
            return np.full(len(pos), np.nan)
        return y - m_hat(x, st["lcpre"].to_numpy()[pos], fit["c_ro"], fit["beta"])
    limb = st["limb"].to_numpy()[pos]
    r = np.full(len(pos), np.nan)
    for k, line in fit["lines"].items():
        if line is None:
            continue
        m = limb == float(k)
        r[m] = y[m] - (line[0] + line[1] * x[m])
    return r


def storm_scores(method, states, site, times, fold_of, fits):
    """One site's storm score at these times, NaN where the current model judges instead."""
    st = states[site]
    pos = st.index.get_indexer(times)
    good = pos >= 0
    run = np.zeros(len(times), dtype=bool)
    run[good] = (
        st["mode"].to_numpy()[pos[good]]
        & st["ok"].to_numpy()[pos[good]]
        & np.isfinite(st["logc"].to_numpy()[pos[good]])
    )
    score = np.full(len(times), np.nan)
    for f in np.unique(fold_of[run]):
        m = run & (fold_of == f)
        fit = fits[site][str(int(f))] if str(int(f)) in fits[site] else fits[site]["all"]
        score[m] = np.abs(residual(method, st, pos[m], fit)) / fit["cutoff"]
    return score


def transform(df, method, states, fold_of, fits):
    """score_eff, rain_line and mode for the current model, with storm-mode steps taken over."""
    df = df.copy()
    df["score_eff"], df["rain_line"], df["mode"] = (
        df["score"].to_numpy(),
        df["line"].to_numpy() * df["mult"].to_numpy(),
        df["flag"].to_numpy(dtype=bool),
    )
    sc = np.full(len(df), np.nan)
    if method != "current":
        times = pd.DatetimeIndex(df["time"].to_numpy())
        for site, idx in df.groupby("site").indices.items():
            if site in CORE and site in states:
                sc[idx] = storm_scores(method, states, site, times[idx], fold_of[idx], fits)
    on = np.isfinite(sc)
    df["score_eff"] = np.where(on, sc, df["score_eff"].to_numpy())
    df["rain_line"] = np.where(on, 1.0, df["rain_line"].to_numpy())
    df["mode"] = on | df["mode"].to_numpy(dtype=bool)
    df["storm"] = on
    return df


def over(df, s_current, s_storm):
    storm_ = df["storm"].fillna(False).to_numpy(dtype=bool)
    mode = df["mode"].fillna(False).to_numpy(dtype=bool)
    current = df["score_eff"].to_numpy() > np.where(
        mode, df["rain_line"].to_numpy() * s_current, df["line"].to_numpy()
    )
    with np.errstate(invalid="ignore"):
        return np.where(storm_, df["score_eff"].to_numpy() > s_storm, current)


def faulted_states(states, grids, mode, gate_flag, marks, faulted, z):
    """The core sites' frames with this fault in, rebuilt on the shared grid where it touched."""
    out = dict(states)
    for site, on in marks.items():
        if site not in CORE:
            continue
        lo, hi = pd.Timestamp(on[0]) - HISTORY, pd.Timestamp(on[-1]) + pd.Timedelta(hours=2)
        part = {site: faulted[site][(faulted[site].index >= lo) & (faulted[site].index < hi)]}
        g_part = grid(part, (site,))[site]
        g = grids[site].copy()
        inside = g.index.intersection(g_part.index)
        g.loc[inside] = g_part.reindex(inside).to_numpy()
        out[site] = frame(
            g,
            states[site]["ok"].copy(),
            mode[site].reindex(g.index).fillna(False).astype(bool),
            gate_flag,
            z,
        )
    return out


# Scoring one method on one model


def evaluate(method, clean, fruns, tables, nbm, groups, events, ws, ok, z, fits, scales):
    """
    Storm and dry detection, rain and dry false alarms, rain-only groups alerted
    and real events caught. scales holds s_current and s_storm, each as "all" plus
    the held-out values by old group, and group_map from this record's groups
    to the old ones, for the old storms only.
    """
    df = gate_cols(clean, nbm, z)
    df["where"] = where(df["time"], groups, events)
    _, gate_flag = gate(nbm.index, nbm, z)
    gate_flag = pd.Series(gate_flag, nbm.index)
    mode = trigger(tables, z)
    states, index = clean_states(tables, ok, mode, gate_flag, z)
    grids = {s: f.reindex(index) for s, f in grid(tables).items()}
    # the held-out fold and scales for each group that was one of the 22, by start time
    gmap = scales.get("group_map", {})
    lookup = np.array([gmap.get(i, -1) for i in range(len(groups))] + [-1])
    fold = lambda w: lookup[np.where(w >= 0, w, len(groups))]  # noqa: E731
    df = transform(df, method, states, fold(df["where"].to_numpy()), fits)
    sites = sorted(df["site"].unique())
    rates = ws_rates(ws, sites, groups, events)
    usable = np.array([not g["known_event"] for g in groups], dtype=bool)
    weeks_g = np.array([(df["where"] == i).sum() / WEEK for i in range(len(groups))])
    dry_weeks = (df["where"] == -1).sum() / WEEK

    def pair(k):
        old = str(gmap.get(k, "all"))
        return scales["s_current"].get(old, scales["s_current"]["all"]), scales["s_storm"].get(
            old, scales["s_storm"]["all"]
        )

    held, dry = np.zeros(len(groups)), 0
    pairs = {k: pair(k) for k in range(len(groups))}
    for _, sub in df.groupby("site"):
        for s_pair in set(pairs.values()) | {pair("all")}:
            ov = over(sub, *s_pair)
            for _, fo in alerts(sub["time"], ov):
                x = where([fo], groups, events)[0]
                if x >= 0 and usable[x] and pairs[x] == s_pair:
                    held[x] += 1
                elif x == -1 and s_pair == pair("all"):
                    dry += 1
    caught = []
    over_all = over(df, *pair("all"))
    graded = [e for e in events if e["label"] == GRADED_LABEL and e["site"] in sites]
    for e in graded:
        sel = (
            (df["site"] == e["site"]) & (df["time"] >= e["start"]) & (df["time"] <= e["end"])
        ).to_numpy()
        if len(set(df["time"][sel & over_all])) >= 3:
            caught.append(f"{e['group']}/{e['site']}")
    keyed = df.set_index(["site", "time"])
    rows = []
    for row in fruns["placements"].itertuples():
        res = fruns["runs"].get(row.id)
        if not res:
            continue
        s1, s2 = pair(row.group) if row.kind == "storm" else pair("all")
        faulted, marks = inject(
            tables,
            row.node,
            row.start,
            row.shape,
            row.size,
            row.sign,
            seed=zlib.crc32(row.id.encode()),
        )
        if faulted is None:
            continue
        fstates = (
            faulted_states(states, grids, mode, gate_flag, marks, faulted, z)
            if method != "current"
            else states
        )
        fr = pd.concat([f.assign(site=site) for site, f in res["sites"].items()], ignore_index=True)
        fr = gate_cols(fr, nbm, z)
        fr = transform(fr, method, fstates, fold(where(fr["time"], groups, events)), fits)
        marked = pd.concat(
            [
                pd.DataFrame({"site": site, "time": pd.DatetimeIndex(ts)})
                for site, ts in res["marks"].items()
            ],
            ignore_index=True,
        )
        take = (
            fr[["site", "time"]]
            .assign(_i=np.arange(len(fr)))
            .merge(marked, on=["site", "time"])["_i"]
            .to_numpy()
        )
        fr = fr.iloc[take].reset_index(drop=True)
        c = keyed.reindex(pd.MultiIndex.from_arrays([fr["site"], fr["time"]])).reset_index()
        new = over(fr, s1, s2) & ~over(c, s1, s2) & np.isfinite(c["score"].to_numpy())
        best = int(pd.Series(new).groupby(fr["site"].to_numpy()).sum().max()) if len(fr) else 0
        rows.append({"id": row.id, "kind": row.kind, "group": row.group, "detected": best >= 3})
    fr = pd.DataFrame(rows, columns=["id", "kind", "group", "detected"])
    storm_f, dry_f = fr[fr["kind"] == "storm"], fr[fr["kind"] == "dry"]
    return {
        "storm_detect": float(storm_f["detected"].mean()) if len(storm_f) else np.nan,
        "dry_detect": float(dry_f["detected"].mean()) if len(dry_f) else np.nan,
        "group_sd": float(storm_f.groupby("group")["detected"].mean().std())
        if len(storm_f)
        else np.nan,
        "rain_fa": float(held[usable].sum() / weeks_g[usable].sum())
        if weeks_g[usable].sum()
        else np.nan,
        "ws_rain_fa": rates["storm_rate"],
        "dry_fa": dry / dry_weeks if dry_weeks else np.nan,
        "rain_only_alerted": int(((held > 0) & usable).sum()),
        "rain_only_groups": int(usable.sum()),
        "events_caught": caught,
        "graded": [f"{e['group']}/{e['site']}" for e in graded],
        "storm_placements": len(storm_f),
        "dry_placements": len(dry_f),
    }


# The decision


def summary(runs, current):
    """One method's means over seeds, against the current model's on the same seeds, and which conditions it meets."""
    a = lambda k: np.array([runs[s][k] for s in SEEDS], dtype=float)  # noqa: E731
    ratio = a("storm_detect") / a("dry_detect")
    r1 = np.array([current[s]["storm_detect"] / current[s]["dry_detect"] for s in SEEDS])
    graded = runs[0]["graded"]
    caught = {e: sum(e in runs[s]["events_caught"] for s in SEEDS) for e in graded}
    ro, ro1 = (
        a("rain_only_alerted"),
        np.array([current[s]["rain_only_alerted"] for s in SEEDS], dtype=float),
    )
    ws = runs[0]["ws_rain_fa"]
    met = {
        1: bool(a("rain_fa").mean() <= ws),
        2: bool(ro.mean() <= ro1.mean()),
        3: all(v > len(SEEDS) / 2 for v in caught.values()),
        4: True,
        5: bool(ratio.mean() >= 0.75),
    }
    return {
        "storm_detect": a("storm_detect").mean(),
        "dry_detect": a("dry_detect").mean(),
        "ratio": ratio.mean(),
        "ratio_sd": ratio.std(ddof=1),
        "ratio_current": r1.mean(),
        "ratio_current_sd": r1.std(ddof=1),
        "rain_fa": a("rain_fa").mean(),
        "rain_fa_seeds": list(a("rain_fa")),
        "ws_rain_fa": ws,
        "dry_fa": a("dry_fa").mean(),
        "rain_only": ro.mean(),
        "rain_only_current": ro1.mean(),
        "caught": caught,
        "met": met,
    }


def decide(sums, n_groups, min_groups=10, order=("depth_fit", "mixing")):
    """The verdict on both runs (with oxford and without): a method, "current model" or "inconclusive", and why."""
    if n_groups < min_groups:
        return (
            "inconclusive",
            f"{n_groups} usable new storm groups, under the {min_groups} the rule needs",
        )
    wins, why = {}, {}
    for m in order:
        both = []
        for ox in ("with", "without"):
            s = sums[(m, ox)]
            spread = max(s["ratio_sd"], s["ratio_current_sd"])
            if all(s["met"].values()):
                both.append(True)
                why[(m, ox)] = "meets all five conditions"
            elif not all(s["met"][k] for k in (1, 2, 3, 4)):
                both.append(False)
                why[(m, ox)] = "fails one of conditions 1 to 4"
            elif s["ratio"] - s["ratio_current"] > spread:
                both.append(True)
                why[(m, ox)] = (
                    f"meets 1 to 4, ratio {s['ratio']:.2f} beats the current model's {s['ratio_current']:.2f} by more than the seed spread {spread:.2f}"
                )
            else:
                both.append(False)
                why[(m, ox)] = (
                    f"meets 1 to 4, but its ratio gain is not more than the seed spread {spread:.2f}"
                )
        wins[m] = all(both)
    winners = [m for m in order if wins[m]]
    if not winners:
        return "current model", why
    if len(winners) == 1:
        return winners[0], why
    r = {m: (sums[(m, "with")]["ratio"] + sums[(m, "without")]["ratio"]) for m in winners}
    top = max(winners, key=r.get)
    for m in winners:
        close = all(
            abs(sums[(m, ox)]["ratio"] - sums[(top, ox)]["ratio"])
            <= max(sums[(m, ox)]["ratio_sd"], sums[(top, ox)]["ratio_sd"])
            for ox in ("with", "without")
        )
        if close:
            return m, why
    return top, why


def save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    # Our own stages, written by save() into the work folder
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec B301
