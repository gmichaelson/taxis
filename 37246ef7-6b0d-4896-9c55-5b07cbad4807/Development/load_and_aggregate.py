
import pandas as pd
import glob
import os

# Files are in current working directory /tmp/tmpXXXX/files
BASE_DIR = "./" # /tmp/tmpXXXX/files

# ── 1. Load all 11 monthly parquet files ─────────────────────────────────────
parquet_files = sorted(glob.glob(os.path.join(BASE_DIR, "fhvhv_tripdata_2025-*.parquet")))
print(f"Found {len(parquet_files)} parquet files:")
for f in parquet_files:
    print(f"  {os.path.basename(f)}")

# Read only the columns we need: pickup datetime and zone
cols = ["pickup_datetime", "PULocationID"]

dfs = []
for fp in parquet_files:
    _df = pd.read_parquet(fp, columns=cols)
    dfs.append(_df)
    print(f"  Loaded {os.path.basename(fp)}: {len(_df):,} rows")

trips_raw = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows across all months: {len(trips_raw):,}")

# ── 2. Parse date and aggregate daily ride counts per taxi zone ───────────────
trips_raw["date"] = pd.to_datetime(trips_raw["pickup_datetime"]).dt.normalize()

daily_zone_counts = (
    trips_raw
    .groupby(["date", "PULocationID"], observed=True)
    .size()
    .reset_index(name="ride_count")
)

daily_zone_counts.rename(columns={"PULocationID": "LocationID"}, inplace=True)
daily_zone_counts["LocationID"] = daily_zone_counts["LocationID"].astype(int)

print(f"\nDaily zone aggregate shape: {daily_zone_counts.shape}")
print(f"Date range: {daily_zone_counts['date'].min()} → {daily_zone_counts['date'].max()}")
print(f"Unique zones: {daily_zone_counts['LocationID'].nunique()}")
print(daily_zone_counts.head(10))
