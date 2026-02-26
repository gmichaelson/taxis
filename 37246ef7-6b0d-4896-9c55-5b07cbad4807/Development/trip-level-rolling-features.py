
import pandas as pd
import numpy as np
import glob
import os

# ── 1. Load all 11 parquet files (pickup_datetime + PULocationID only) ────────
_base = "./"
_pq_files = sorted(glob.glob(os.path.join(_base, "fhvhv_tripdata_2025-*.parquet")))
print(f"Found {len(_pq_files)} parquet files:")
for _fp in _pq_files:
    print(f"  {os.path.basename(_fp)}")

_cols = ["pickup_datetime", "PULocationID"]
_dfs = []
for _fp in _pq_files:
    _chunk = pd.read_parquet(_fp, columns=_cols)
    _dfs.append(_chunk)
    print(f"  Loaded {os.path.basename(_fp)}: {len(_chunk):,} rows")

# ── 2. Concatenate, parse datetime, sort by zone then time ───────────────────
_trips = pd.concat(_dfs, ignore_index=True)
print(f"\nTotal trips loaded: {len(_trips):,}")

_trips["pickup_datetime"] = pd.to_datetime(_trips["pickup_datetime"])
_trips = _trips.dropna(subset=["pickup_datetime", "PULocationID"]).copy()
_trips["PULocationID"] = _trips["PULocationID"].astype("int32")

print("Sorting by PULocationID then pickup_datetime...")
_trips = _trips.sort_values(["PULocationID", "pickup_datetime"]).reset_index(drop=True)

# ── 3. Trip-level rolling counts via numpy searchsorted ───────────────────────
# Use int64 nanoseconds for fast vectorised window comparisons
print("Computing rolling trip counts (1h / 6h / 24h) using searchsorted...")

_ts   = _trips["pickup_datetime"].values.astype("int64")   # ns since epoch
_zone = _trips["PULocationID"].values

_ns_1h  = np.int64(3_600_000_000_000)   # 1 hour  in nanoseconds
_ns_6h  = np.int64(21_600_000_000_000)  # 6 hours in nanoseconds
_ns_24h = np.int64(86_400_000_000_000)  # 24 hours in nanoseconds

# Identify contiguous zone boundaries (array is sorted by zone already)
_, _zone_starts = np.unique(_zone, return_index=True)
_zone_ends = np.append(_zone_starts[1:], len(_zone))

_r1h  = np.empty(len(_trips), dtype="int32")
_r6h  = np.empty(len(_trips), dtype="int32")
_r24h = np.empty(len(_trips), dtype="int32")

for _zi in range(len(_zone_starts)):
    _s, _e = int(_zone_starts[_zi]), int(_zone_ends[_zi])
    _t = _ts[_s:_e]
    _local_idx = np.arange(_e - _s, dtype="int32")
    # Count trips in (t - window, t] — current trip is included (right-open on the left)
    _r1h[_s:_e]  = _local_idx - np.searchsorted(_t, _t - _ns_1h,  side="left")
    _r6h[_s:_e]  = _local_idx - np.searchsorted(_t, _t - _ns_6h,  side="left")
    _r24h[_s:_e] = _local_idx - np.searchsorted(_t, _t - _ns_24h, side="left")

_trips["rolling_1h"]  = _r1h
_trips["rolling_6h"]  = _r6h
_trips["rolling_24h"] = _r24h

print(f"Trip-level rolling features computed for {len(_trips):,} trips.")
print(f"  rolling_1h  range: {_r1h.min()} – {_r1h.max()}")
print(f"  rolling_6h  range: {_r6h.min()} – {_r6h.max()}")
print(f"  rolling_24h range: {_r24h.min()} – {_r24h.max()}")

# ── 4. Aggregate to daily (date + PULocationID) ───────────────────────────────
print("\nAggregating trip-level features to daily zone summaries...")

_trips["date"] = _trips["pickup_datetime"].dt.normalize()

daily_zone_features = (
    _trips
    .groupby(["date", "PULocationID"], observed=True, sort=True)
    .agg(
        daily_trips      = ("pickup_datetime", "count"),
        rolling_1h_mean  = ("rolling_1h",  "mean"),
        rolling_6h_mean  = ("rolling_6h",  "mean"),
        rolling_24h_mean = ("rolling_24h", "mean"),
    )
    .reset_index()
    .rename(columns={"PULocationID": "zone"})
)

daily_zone_features["rolling_1h_mean"]  = daily_zone_features["rolling_1h_mean"].round(2)
daily_zone_features["rolling_6h_mean"]  = daily_zone_features["rolling_6h_mean"].round(2)
daily_zone_features["rolling_24h_mean"] = daily_zone_features["rolling_24h_mean"].round(2)

# ── 5. Print shape, dtypes, head(10), memory usage ───────────────────────────
print(f"\n{'='*60}")
print(f"Shape:   {daily_zone_features.shape}")
print(f"\nDtypes:")
print(daily_zone_features.dtypes.to_string())
print(f"\nHead (10 rows):")
print(daily_zone_features.head(10).to_string(index=False))
print(f"\nMemory usage (deep):")
_mem = daily_zone_features.memory_usage(deep=True)
print(_mem.to_string())
print(f"  Total: {_mem.sum() / 1e6:.2f} MB")
print(f"\nDate range:    {daily_zone_features['date'].min()} → {daily_zone_features['date'].max()}")
print(f"Unique zones:  {daily_zone_features['zone'].nunique()}")
print(f"Total trips:   {daily_zone_features['daily_trips'].sum():,}")
