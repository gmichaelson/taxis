
import pandas as pd
import numpy as np
import glob
import os

print("Loading parquet files (pickup_datetime + PULocationID only)...")

_BASE_DIR = "./"
_parquet_files = sorted(glob.glob(os.path.join(_BASE_DIR, "fhvhv_tripdata_2025-*.parquet")))
print(f"Found {len(_parquet_files)} parquet files")

_cols = ["pickup_datetime", "PULocationID"]
_dfs = []
for _fp in _parquet_files:
    _chunk = pd.read_parquet(_fp, columns=_cols)
    _dfs.append(_chunk)
    print(f"  Loaded {os.path.basename(_fp)}: {len(_chunk):,} rows")

_trips = pd.concat(_dfs, ignore_index=True)
print(f"\nTotal trips: {len(_trips):,}")

# Clean & parse
_trips["pickup_datetime"] = pd.to_datetime(_trips["pickup_datetime"])
_trips = _trips.dropna(subset=["pickup_datetime", "PULocationID"]).copy()
_trips["PULocationID"] = _trips["PULocationID"].astype("int32")

# ── Sort by zone, then time ───────────────────────────────────────────────────
print("Sorting by PULocationID and pickup_datetime...")
_trips = _trips.sort_values(["PULocationID", "pickup_datetime"]).reset_index(drop=True)

# ── Trip-level rolling counts via numpy searchsorted ─────────────────────────
# Convert datetimes to int64 nanoseconds for arithmetic
print("Computing rolling pickup counts per zone using searchsorted (1h, 6h, 24h)...")

_ts   = _trips["pickup_datetime"].values.astype("int64")   # nanoseconds since epoch
_zone = _trips["PULocationID"].values

_ns_1h  = np.int64(3_600_000_000_000)
_ns_6h  = np.int64(21_600_000_000_000)
_ns_24h = np.int64(86_400_000_000_000)

# Find contiguous start/end index of each zone group (array is sorted by zone)
_, _zone_starts = np.unique(_zone, return_index=True)
_zone_ends = np.append(_zone_starts[1:], len(_zone))

_r1h  = np.empty(len(_trips), dtype="int32")
_r6h  = np.empty(len(_trips), dtype="int32")
_r24h = np.empty(len(_trips), dtype="int32")

for _zi in range(len(_zone_starts)):
    _s, _e = _zone_starts[_zi], _zone_ends[_zi]
    _t = _ts[_s:_e]
    # Count how many prior trips in [t-window, t) using searchsorted on sorted timestamps
    _local_idx = np.arange(_e - _s)   # 0-based index within zone
    _r1h[_s:_e]  = _local_idx - np.searchsorted(_t, _t - _ns_1h,  side="left")
    _r6h[_s:_e]  = _local_idx - np.searchsorted(_t, _t - _ns_6h,  side="left")
    _r24h[_s:_e] = _local_idx - np.searchsorted(_t, _t - _ns_24h, side="left")

_trips["rolling_1h"]  = _r1h
_trips["rolling_6h"]  = _r6h
_trips["rolling_24h"] = _r24h

# Trip-level dataframe: pickup_datetime, PULocationID, rolling_1h, rolling_6h, rolling_24h
_trip_features = _trips[["pickup_datetime", "PULocationID", "rolling_1h", "rolling_6h", "rolling_24h"]].copy()
print(f"\nTrip-level features computed.")
print(f"  Shape: {_trip_features.shape}")
print(f"  Columns: {list(_trip_features.columns)}")
print(_trip_features.head(5).to_string(index=False))

# ── Aggregate to daily (date + zone) ─────────────────────────────────────────
print("\nAggregating to daily zone features (mean rolling + trip count)...")

_trips["date"] = _trips["pickup_datetime"].dt.normalize()

daily_zone_features = (
    _trips
    .groupby(["date", "PULocationID"], observed=True, sort=True)
    .agg(
        daily_trips      = ("pickup_datetime",  "count"),
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

# ── Summary output ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"daily_zone_features shape:   {daily_zone_features.shape}")
print(f"Columns:                     {list(daily_zone_features.columns)}")
print(f"Date range:                  {daily_zone_features['date'].min()} → {daily_zone_features['date'].max()}")
print(f"Unique zones:                {daily_zone_features['zone'].nunique()}")
print(f"Total daily_trips sum:       {daily_zone_features['daily_trips'].sum():,}")
print(f"\nMemory usage:")
print(daily_zone_features.memory_usage(deep=True).to_string())
print(f"  Total: {daily_zone_features.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"\nHead (10 rows):")
print(daily_zone_features.head(10).to_string(index=False))
