# These are connectivity smoke checks, not assertions about correctness. Every
# one of them hits a live endpoint, so a pass means the network and credentials
# worked today and a failure may be an outage rather than a regression. Four of
# them (test_api, test_nws, test_sql, test_integration) return a bool instead of
# asserting, which pytest reports as PytestReturnNotNoneWarning and, more to the
# point, means they pass whatever the endpoint returns. Do not read a green run
# here as the clients being verified.
import argparse
import sys
import traceback
from datetime import UTC, datetime, timedelta


def banner(title):
    print()
    print("-" * 72)
    print(f"  {title}")
    print("-" * 72)


def show_df(df, label):
    """Compact summary of a DataFrame."""
    print(f"  {label}: shape={df.shape}")
    if df.empty:
        print(f"  {label}: (empty)")
        return
    print(f"  columns: {list(df.columns)}")
    # Find whatever time column we have
    time_col = next(
        (c for c in ("timestamp", "datetime") if c in df.columns),
        None,
    )
    if time_col:
        print(f"  range: {df[time_col].min()} to {df[time_col].max()}")
    elif df.index.name in ("timestamp", "datetime"):
        print(f"  range: {df.index.min()} to {df.index.max()}")
    print(f"  head:\n{df.head(2).to_string()}")


def test_imports():
    banner("Layer 0: imports")
    from strawberrywatch.config import Config  # noqa: F401

    print("  strawberrywatch.config        :)")
    from strawberrywatch.ingest import api_client  # noqa: F401

    print("  strawberrywatch.ingest.api_client     :)")
    from strawberrywatch.ingest import sql_client  # noqa: F401

    print("  strawberrywatch.ingest.sql_client     :)")
    from strawberrywatch.ingest import weather_client  # noqa: F401

    print("  strawberrywatch.ingest.weather_client  :)")


def test_config():
    banner("Layer 1: Config loads")
    from strawberrywatch.config import Config

    print(f"  API_BASE_URL    = {Config.API_BASE_URL}")
    print(f"  API_TOKEN set?  = {bool(Config.API_TOKEN)}")

    # MySQL: show keys but mask the password
    print(f"  MYSQL_HOST      = {Config.MYSQL_HOST or '(unset)'}")
    print(f"  MYSQL_USER      = {Config.MYSQL_USER or '(unset)'}")
    print(f"  MYSQL_PASSWORD  = {'(set)' if Config.MYSQL_PASSWORD else '(unset)'}")
    print(f"  MYSQL_DATABASE  = {Config.MYSQL_DATABASE or '(unset)'}")
    print(f"  MYSQL_PORT      = {Config.MYSQL_PORT}")

    print(f"  NWS_STATION_ID  = {Config.NWS_STATION_ID}")
    print(f"  NWS_USER_AGENT  = {Config.NWS_USER_AGENT}")
    print(f"  USE_NWS_WEATHER = {getattr(Config, 'USE_NWS_WEATHER', '(missing)')}")

    print(f"  LOCATIONS       = {Config.LOCATIONS}")
    print(f"  SEQUENCE_LENGTH = {Config.SEQUENCE_LENGTH}")


def test_api():
    banner("Layer 2a: API client")
    from strawberrywatch.config import Config
    from strawberrywatch.ingest.api_client import fetch_creek_data, fetch_network_snapshot

    end = datetime.now(UTC)
    start = end - timedelta(days=2)

    print("\n  fetch_creek_data('oxford', last 2 days)")
    df = fetch_creek_data("oxford", start, end)
    show_df(df, "single-site")

    print("\n  fetch_network_snapshot(last 2 days)")
    df_net = fetch_network_snapshot(start, end)
    show_df(df_net, "network")

    if df_net.empty:
        print("\n  network snapshot was empty, API may be rejecting requests")
        return False

    sites_seen = sorted(df_net["station_id"].unique()) if "station_id" in df_net.columns else []
    print(f"  sites returned: {sites_seen}")
    expected = set(Config.LOCATIONS)
    missing = expected - set(sites_seen)
    if missing:
        print(f"  missing sites: {sorted(missing)}")
    return True


def test_nws():
    banner("Layer 2b: NWS weather client")
    from strawberrywatch.ingest.weather_client import fetch_nws_weather

    end = datetime.now(UTC)
    start = end - timedelta(days=2)

    df = fetch_nws_weather(start, end)
    show_df(df, "nws")

    if df.empty:
        print("\n  NWS returned empty, station may be offline")
        return False

    # Sanity-check: temperature should exist and look like Celsius
    if "air_temp_c" in df.columns:
        tmin, tmax = df["air_temp_c"].min(), df["air_temp_c"].max()
        print(f"  air_temp_c range: {tmin:.1f} to {tmax:.1f} °C")
        if tmin < -20 or tmax > 50:
            print("  temperature range looks suspicious for Berkeley")
    return True


def test_sql():
    banner("Layer 2c: SQL client")
    from strawberrywatch.config import Config
    from strawberrywatch.ingest.sql_client import fetch_creek_data_sql, fetch_network_snapshot_sql

    if not all(
        [Config.MYSQL_HOST, Config.MYSQL_USER, Config.MYSQL_PASSWORD, Config.MYSQL_DATABASE]
    ):
        print("  MYSQL_* env vars not all set, skipping SQL test")
        return False

    end = datetime.now(UTC)
    start = end - timedelta(days=2)

    print("\n  fetch_creek_data_sql('oxford', last 2 days)")
    df = fetch_creek_data_sql("oxford", start, end)
    show_df(df, "single-site")

    print("\n  fetch_network_snapshot_sql(last 2 days)")
    df_net = fetch_network_snapshot_sql(start, end)
    show_df(df_net, "network")

    if df_net.empty:
        print("\n  SQL returned empty for all sites")
        return False

    sites_seen = sorted(df_net["station_id"].unique()) if "station_id" in df_net.columns else []
    print(f"  sites returned: {sites_seen}")
    return True


def test_integration():
    banner("Layer 3: timestamp alignment between sources")
    from strawberrywatch.ingest.api_client import fetch_network_snapshot
    from strawberrywatch.ingest.weather_client import fetch_nws_weather

    end = datetime.now(UTC)
    start = end - timedelta(days=2)

    print("  pulling creek (API) + weather (NWS) for the same 2-day window...")
    df_creek = fetch_network_snapshot(start, end)
    df_nws = fetch_nws_weather(start, end)

    if df_creek.empty:
        print("  creek data was empty, can't test merge")
        return False
    if df_nws.empty:
        print("  NWS data was empty, can't test merge")
        return False

    creek_times = df_creek["timestamp"].dt.floor("h").unique()
    nws_times = df_nws.index.floor("h").unique()
    overlap = set(creek_times) & set(nws_times)

    print(f"  creek hourly buckets: {len(creek_times)}")
    print(f"  nws   hourly buckets: {len(nws_times)}")
    print(f"  overlap: {len(overlap)}")

    if len(overlap) == 0:
        print("  no overlapping timestamps, merge would produce nothing")
        return False
    print("  :) timestamps overlap; merge would have rows to join")
    return True


TESTS = {
    "imports": test_imports,
    "config": test_config,
    "api": test_api,
    "nws": test_nws,
    "sql": test_sql,
    "integration": test_integration,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(TESTS.keys()), help="run a single test")
    ap.add_argument("--skip-sql", action="store_true", help="skip the SQL test")
    args = ap.parse_args()

    if args.only:
        to_run = [args.only]
    else:
        to_run = list(TESTS.keys())
        if args.skip_sql:
            to_run = [t for t in to_run if t != "sql"]

    failures = []
    for name in to_run:
        try:
            result = TESTS[name]()
            if result is False:
                failures.append(name)
        except Exception as e:
            print(f"\n  :( {name} FAILED: {e}")
            traceback.print_exc()
            failures.append(name)

    banner("summary")
    if failures:
        print(f"  {len(failures)} test(s) reported issues: {failures}")
        sys.exit(1)
    print(f"  all {len(to_run)} test(s) completed cleanly")


if __name__ == "__main__":
    main()
