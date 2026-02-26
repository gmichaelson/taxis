
import pandas as pd
import numpy as np
import subprocess, sys

# ── Install lightgbm + scikit-learn at runtime ────────────────────────────────
print("Installing lightgbm and scikit-learn …")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "lightgbm", "scikit-learn"],
    stdout=subprocess.DEVNULL,
)
import lightgbm as lgb
print(f"lightgbm {lgb.__version__} ready")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

# ── Zerve design system colours ───────────────────────────────────────────────
BG_COL   = "#1D1D20"
TEXT_COL = "#fbfbff"
GRID_COL = "#2e2e34"
ACTUAL_C = "#A1C9F4"
PRED_C   = "#FFB482"

# ── Feature engineering ───────────────────────────────────────────────────────
def make_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["dow"]    = df["date"].dt.dayofweek
    df["dom"]    = df["date"].dt.day
    df["month"]  = df["date"].dt.month
    df["woy"]    = df["date"].dt.isocalendar().week.astype(int)
    df["lag_1"]  = df["daily_trips"].shift(1)
    df["lag_7"]  = df["daily_trips"].shift(7)
    df["lag_14"] = df["daily_trips"].shift(14)
    return df

FEATURE_COLS = [
    "dow", "dom", "month", "woy",
    "lag_1", "lag_7", "lag_14",
    "rolling_1h_mean", "rolling_6h_mean", "rolling_24h_mean",
]
TARGET_COL = "daily_trips"
TEST_DAYS  = 14

print("\nStarting per-zone LightGBM walk-forward forecasting …")
print(f"  Dataset: {daily_zone_counts.shape[0]:,} rows | "
      f"{daily_zone_counts['zone'].nunique()} unique zones")
print(f"  Walk-forward split: train = all but last {TEST_DAYS} days | "
      f"test = last {TEST_DAYS} days")

# ── Top-10 busiest zones ──────────────────────────────────────────────────────
_zone_totals = (
    daily_zone_counts
    .groupby("zone")["daily_trips"]
    .sum()
    .sort_values(ascending=False)
)
top10_zones = _zone_totals.head(10).index.tolist()
print(f"\nTop-10 busiest zones (by total trip volume): {top10_zones}")

# ── LightGBM native API (no sklearn wrapper needed) ──────────────────────────
_lgb_params = dict(
    objective        = "regression_l1",
    num_boost_round  = 300,
    learning_rate    = 0.05,
    num_leaves       = 31,
    min_child_samples= 5,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 1,
    seed             = 42,
    verbose          = -1,
)

all_zone_maes = []
zone_results  = {}

_all_zones = daily_zone_counts["zone"].unique()

for _z in _all_zones:
    _zdf = daily_zone_counts[daily_zone_counts["zone"] == _z].copy()
    _zdf = make_zone_features(_zdf).dropna(subset=FEATURE_COLS)

    if len(_zdf) < TEST_DAYS + 15:
        continue

    _cutoff = _zdf["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    _train  = _zdf[_zdf["date"] <  _cutoff]
    _test   = _zdf[_zdf["date"] >= _cutoff]

    if len(_train) < 10 or len(_test) < 1:
        continue

    _dtrain = lgb.Dataset(_train[FEATURE_COLS], label=_train[TARGET_COL])
    _model  = lgb.train(_lgb_params, _dtrain, num_boost_round=_lgb_params["num_boost_round"])

    _y_pred = np.maximum(0, _model.predict(_test[FEATURE_COLS].values))
    _y_test = _test[TARGET_COL].values

    _mae_z = _mae(_y_test, _y_pred)
    all_zone_maes.append(_mae_z)

    if _z in top10_zones:
        zone_results[_z] = {
            "dates" : _test["date"].values,
            "actual": _y_test,
            "pred"  : _y_pred,
        }

overall_mae = float(np.mean(all_zone_maes))
print(f"\nOverall MAE across all {len(all_zone_maes)} zones: {overall_mae:,.1f} trips/day")

# ── 2×5 multi-panel figure ────────────────────────────────────────────────────
forecast_fig, axes = plt.subplots(2, 5, figsize=(22, 9), facecolor=BG_COL)
forecast_fig.patch.set_facecolor(BG_COL)
forecast_fig.suptitle(
    "LightGBM Forecast vs Actual — Top 10 Busiest NYC Taxi Zones\n"
    f"Walk-Forward Test Window: Last {TEST_DAYS} Days",
    color=TEXT_COL, fontsize=14, fontweight="bold", y=1.02,
)

_axes_flat = axes.flatten()

for _idx, _z in enumerate(top10_zones):
    _ax = _axes_flat[_idx]
    _ax.set_facecolor(BG_COL)

    if _z not in zone_results:
        _ax.text(0.5, 0.5, "Insufficient data", color=TEXT_COL,
                 ha="center", va="center", transform=_ax.transAxes, fontsize=9)
        _ax.set_title(f"Zone {_z}", color=TEXT_COL, fontsize=10, fontweight="bold")
        for _sp in _ax.spines.values():
            _sp.set_edgecolor(GRID_COL)
        continue

    _res   = zone_results[_z]
    _dates = pd.to_datetime(_res["dates"])
    _rank  = _idx + 1
    _total = int(_zone_totals[_z])
    _z_mae = _mae(_res["actual"], _res["pred"])

    _ax.plot(_dates, _res["actual"], color=ACTUAL_C, linewidth=2.0,
             marker="o", markersize=4, label="Actual")
    _ax.plot(_dates, _res["pred"],   color=PRED_C,   linewidth=2.0,
             marker="s", markersize=4, linestyle="--", label="Predicted")

    _ax.set_title(
        f"Zone {_z}  (Rank #{_rank})\nTotal: {_total:,}  |  MAE: {_z_mae:,.0f}",
        color=TEXT_COL, fontsize=8.5, fontweight="bold", pad=5,
    )
    _ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    _ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    _ax.tick_params(axis="x", colors=TEXT_COL, labelsize=6.5, rotation=35)
    _ax.tick_params(axis="y", colors=TEXT_COL, labelsize=7)
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(GRID_COL)
    _ax.grid(axis="y", color=GRID_COL, linewidth=0.5, linestyle=":")

    if _idx == 0:
        _ax.legend(loc="upper left", fontsize=7,
                   facecolor=BG_COL, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL)

forecast_fig.text(0.5, -0.02, "Date (MM/DD)", ha="center",
                  color=TEXT_COL, fontsize=11)
forecast_fig.text(0.0, 0.5, "Daily Trips", va="center", rotation="vertical",
                  color=TEXT_COL, fontsize=11)

plt.tight_layout(rect=[0.02, 0.02, 1, 1])

_out_path = "./forecast_top10_zones.png"
forecast_fig.savefig(
    _out_path, dpi=120, bbox_inches="tight",
    facecolor=BG_COL, edgecolor="none",
)
plt.close(forecast_fig)

print(f"\nFigure saved → {_out_path}")
print("Done.")
