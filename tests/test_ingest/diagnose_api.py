import requests
from strawberrywatch.config import Config
from datetime import datetime, timedelta, timezone

end = datetime.now(timezone.utc)
start = end - timedelta(days=1)
base_params = [
    ("site", "oxford"),
    ("start", start.strftime("%Y-%m-%dT%H:%M:%S")),
    ("end", end.strftime("%Y-%m-%dT%H:%M:%S")),
]

cases = [
    ("just timestamp + cond",       ["timestamp", "Meter_Hydros21_Cond"]),
    ("just timestamp + depth",      ["timestamp", "Meter_Hydros21_Depth"]),
    ("just timestamp + temp",       ["timestamp", "Meter_Hydros21_Temp"]),
    ("just timestamp + batt",       ["timestamp", "EnviroDIY_Mayfly_Batt"]),
    ("ts + cond + depth (3 vars)",   ["timestamp", "Meter_Hydros21_Cond", "Meter_Hydros21_Depth"]),
    ("ts + cond + temp (3 vars)",    ["timestamp", "Meter_Hydros21_Cond", "Meter_Hydros21_Temp"]),
    ("ts + cond + depth + temp",     ["timestamp", "Meter_Hydros21_Cond", "Meter_Hydros21_Depth", "Meter_Hydros21_Temp"]),
    ("ts + cond + depth + temp + batt", ["timestamp", "Meter_Hydros21_Cond", "Meter_Hydros21_Depth", "Meter_Hydros21_Temp", "EnviroDIY_Mayfly_Batt"]),
    # Order reversal test:
    ("REVERSED ts + batt + temp + depth + cond", ["timestamp", "EnviroDIY_Mayfly_Batt", "Meter_Hydros21_Temp", "Meter_Hydros21_Depth", "Meter_Hydros21_Cond"]),
    # No timestamp:
    ("just cond + depth (no ts)",  ["Meter_Hydros21_Cond", "Meter_Hydros21_Depth"]),
]

for label, cols in cases:
    params = base_params + [("vars", c) for c in cols]
    r = requests.get(Config.API_BASE_URL, params=params, timeout=30)
    if r.status_code != 200:
        print(f"[{label}]  HTTP {r.status_code}  {r.text[:120]}")
        continue
    data = r.json()
    if not data:
        print(f"[{label}]  empty response")
        continue
    keys_in_first = list(data[0].keys())
    keys_in_last = list(data[-1].keys())
    print(f"[{label}]")
    print(f"   requested:   {cols}")
    print(f"   returned:    {keys_in_first}  ({len(data)} rows)")
    if keys_in_first != keys_in_last:
        print(f"   last row had different keys: {keys_in_last}")