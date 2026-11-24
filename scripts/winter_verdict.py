"""
The winter verdict on Dusk Crayfish's storm mode, by the decision rule in USER_MANUAL.md, Appendix A.

    python -m scripts.winter_verdict --start 2026-11-17 --end 2027-03-31

Every stage is saved in --work, so a second run skips what's done.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import winter_scoring as W

OLD_START, OLD_END = "2025-06-01", "2026-11-17"


def inputs(work, start, end, history_days, raw_folder, rain_folder, weather_file):
    """Raw readings, rain and weather for the record, pulled once and kept in work."""
    lo, hi = start - pd.Timedelta(days=history_days), end
    path = work / "inputs.pkl"
    if path.exists():
        return W.load(path)
    if raw_folder:
        from strawberrywatch.watch import sources

        raw = sources.CsvSource(raw_folder).raw_tables(list(W.SITES), lo, hi)
    else:
        raw = W.pull_readings(lo, hi)
    rain = {}
    for name, model in W.RAIN_MODELS.items():
        if rain_folder and (Path(rain_folder) / f"{model}.csv").exists():
            df = pd.read_csv(Path(rain_folder) / f"{model}.csv")
            s = pd.Series(
                pd.to_numeric(df["rain_mm"], errors="coerce").to_numpy(),
                pd.to_datetime(df["time"], utc=True),
            )
            s = s[~s.index.duplicated()]
        else:
            s = W.pull_rain(model, lo, hi)
        rain[name] = W.on_rain_grid(s, lo, hi)
        print(f"{model}: {int(rain[name].notna().sum())} quarter hours, {rain[name].sum():.1f} mm")
    # A weather frame we pickled ourselves, given on the command line
    weather = pd.read_pickle(weather_file) if weather_file else W.pull_weather(lo, hi)  # nosec B301
    out = {"lo": lo, "hi": hi, "raw": raw, "rain": rain, "weather": weather}
    W.save(out, path)
    return out


def record(work, inp, start, end, z):
    """Screened tables, real events, storm groups and fault placements."""
    path = work / "record.pkl"
    if path.exists():
        return W.load(path)
    dusk, ws, depth_flags = W.screen(inp["raw"], inp["lo"], inp["hi"])
    events = W.real_events(start, end)
    groups = W.storm_groups(dusk, events, z, start, end)
    wet = pd.concat(list(inp["rain"].values()), axis=1).max(axis=1) >= W.WET
    wet = wet[wet.index >= start]
    place = W.placements(dusk, groups, events, wet)
    out = {
        "dusk": dusk,
        "ws": ws,
        "depth_flags": depth_flags,
        "events": events,
        "groups": groups,
        "placements": place,
    }
    W.save(out, path)
    print(
        f"{len(groups)} storm groups, {sum(1 for g in groups if not g['known_event'])} rain-only; {len(place)} fault placements"
    )
    return out


def tables_for(rec, ox):
    return {s: f for s, f in rec["dusk"].items() if ox == "with" or s != "oxford"}


def score_models(work, inp, rec, start, end, workers):
    """Each NBM seed's scores on the clean record and on every faulted copy, with oxford and without, and Water Strider's."""
    weather = W.weather_with(inp["weather"], inp["rain"]["nbm"])
    for seed in W.SEEDS:
        folder = W.WINTER / f"nbm_s{seed}"
        for ox in ("with", "without"):
            path = work / f"scores_s{seed}_{ox}.pkl"
            if path.exists():
                continue
            tables = tables_for(rec, ox)
            clean = W.dusk_scores(W.load_dusk(folder), tables, weather, start, end)
            runs = W.fault_scores(folder, tables, weather, rec["placements"], workers)
            W.save({"clean": clean, "runs": runs}, path)
            print(
                f"seed {seed}, {ox} oxford: {len(clean)} scores, {sum(v is not None for v in runs.values())} faulted runs"
            )
    path = work / "water_strider.pkl"
    if not path.exists():
        ws_weather = W.weather_with(inp["weather"], inp["rain"]["hrrr"])
        W.save(
            W.water_strider_scores(W.WATER_STRIDER, rec["ws"], ws_weather, start, end),
            path,
        )


def scales_for(frozen, method, seed, ox, held_out, groups):
    current = frozen["current_model"][f"s{seed}"][ox]
    out = {"s_current": {"all": current["all"]}, "s_storm": {"all": frozen["s_storm"]}}
    if held_out:
        starts = {pd.Timestamp(g["start"]): j for j, g in enumerate(frozen["old_groups"])}
        out["group_map"] = {
            i: starts[g["start"]] for i, g in enumerate(groups) if g["start"] in starts
        }
        out["s_current"].update(current["groups"])
        if method != "current":
            h = frozen["held_out_s_storm"][method][f"s{seed}"][ox]
            out["s_storm"] = {"all": h["all"], **h["groups"]}
    return out


_JOB = {}


def _job_init(work, held_out):
    _JOB.update(
        inp=W.load(work / "inputs.pkl"),
        rec=W.load(work / "record.pkl"),
        ws=W.load(work / "water_strider.pkl"),
        frozen=W.load_frozen(),
        work=work,
        held_out=held_out,
    )


def _job(key):
    method, seed, ox = key
    j = _JOB
    rec, frozen = j["rec"], j["frozen"]
    sc = W.load(j["work"] / f"scores_s{seed}_{ox}.pkl")
    ok = {s: W.depth_passes(f, frozen["switch"]) for s, f in rec["depth_flags"].items()}
    fits = frozen["fits"][method] if method != "current" else {}
    res = W.evaluate(
        method,
        sc["clean"],
        {"placements": rec["placements"], "runs": sc["runs"]},
        tables_for(rec, ox),
        j["inp"]["rain"]["nbm"],
        rec["groups"],
        rec["events"],
        j["ws"],
        ok,
        frozen["switch"],
        fits,
        scales_for(frozen, method, seed, ox, j["held_out"], rec["groups"]),
    )
    return key, res


def print_verdict(results, rec, start, end, held_out):
    sums = {}
    for method in ("depth_fit", "mixing"):
        for ox in ("with", "without"):
            runs = {s: results[(method, s, ox)] for s in W.SEEDS}
            current = {s: results[("current", s, ox)] for s in W.SEEDS}
            sums[(method, ox)] = W.summary(runs, current)
            sums[("current", ox)] = W.summary(current, current)
    n_groups = len(rec["groups"])
    verdict, why = W.decide(sums, n_groups)
    lines = [
        "Winter verdict",
        "",
        f"Record {start:%Y-%m-%d} to {end:%Y-%m-%d}. {n_groups} storm groups, "
        f"{sum(1 for g in rec['groups'] if not g['known_event'])} rain-only. {len(rec['placements'])} fake faults placed. "
        f"Graded events: {', '.join(results[('current', 0, 'with')]['graded']) or 'none'}.",
        "",
        f"Fits: {'the held-out fits, per old storm group' if held_out else 'frozen, fitted on all 22 old storm groups'}.",
        "",
        f"Verdict: {W.NAMES.get(verdict, verdict)}.",
        "",
    ]
    if isinstance(why, str):
        lines += [why, ""]
    else:
        lines += [f"- {W.NAMES[m]}, {ox} oxford: {w}" for (m, ox), w in why.items()] + [""]
    lines += [
        "| oxford | method | storm % | dry % | storm/dry (seed SD) | rain FA a week (per seed) | WS rain FA | rain-only groups | current model's | events caught (seeds) | conditions met |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ox in ("with", "without"):
        for m in W.METHODS:
            s = sums[(m, ox)]
            name = W.NAMES[m]
            met = ", ".join(str(k) for k, v in s["met"].items() if v) or "none"
            caught = " ".join(f"{v}/5" for v in s["caught"].values()) or "no graded events"
            lines.append(
                f"| {ox} | {name} | {100 * s['storm_detect']:.1f} | {100 * s['dry_detect']:.1f} | {s['ratio']:.2f} ({s['ratio_sd']:.2f}) | "
                f"{s['rain_fa']:.3f} ({'/'.join(f'{v:.3f}' for v in s['rain_fa_seeds'])}) | {s['ws_rain_fa']:.3f} | "
                f"{s['rain_only']:.1f} | {s['rain_only_current']:.1f} | {caught} | {met} |"
            )
    lines += ["", "What each verdict means is in USER_MANUAL.md, Appendix A."]
    print("\n".join(lines))
    return verdict, sums


def main(
    start,
    end,
    work,
    history_days,
    old,
    held_out,
    workers,
    raw_folder,
    rain_folder,
    weather_file,
):
    if old:
        start, end, history_days = OLD_START, OLD_END, 0
    start = pd.Timestamp(start, tz="UTC")
    # the end date is the last day judged, except for the old record, which was scored to midnight
    end = pd.Timestamp(end, tz="UTC") + (pd.Timedelta(0) if old else pd.Timedelta(days=1))
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    frozen = W.load_frozen()
    z = frozen["switch"]
    inp = inputs(work, start, end, history_days, raw_folder, rain_folder, weather_file)
    rec = record(work, inp, start, end, z)
    score_models(work, inp, rec, start, end, workers)
    tag = "held_out" if held_out else "frozen"
    path = work / f"results_{tag}.pkl"
    if path.exists():
        results = W.load(path)
    else:
        keys = [(m, s, ox) for m in W.METHODS for s in W.SEEDS for ox in ("with", "without")]
        with Pool(workers, initializer=_job_init, initargs=(work, held_out)) as pool:
            results = dict(pool.map(_job, keys, chunksize=1))
        W.save(results, path)
    verdict, sums = print_verdict(results, rec, start, end, held_out)
    summary = {
        f"{m}_{ox}": {k: v for k, v in s.items() if k != "caught"} for (m, ox), s in sums.items()
    }
    as_json = json.dumps(
        summary, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)
    )
    (work / f"summary_{tag}.json").write_text(as_json)
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-11-17", help="the day storm mode was deployed")
    parser.add_argument("--end", default="2027-03-31")
    parser.add_argument("--work", default=str(W.WORK))
    parser.add_argument("--history-days", type=int, default=10)
    parser.add_argument(
        "--old",
        action="store_true",
        help="the 22 old storms, 1 June 2025 to 17 November 2026, to check the scoring",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="the held-out fits and scales per old storm group, in place of the frozen ones",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--raw-folder", default=None, help="one CSV of readings per site, in place of the API"
    )
    parser.add_argument(
        "--rain-folder", default=None, help="one CSV per Open-Meteo model, in place of the API"
    )
    parser.add_argument(
        "--weather-file", default=None, help="a pickled weather frame, in place of the archive"
    )
    _args = parser.parse_args()
    main(**vars(_args))
