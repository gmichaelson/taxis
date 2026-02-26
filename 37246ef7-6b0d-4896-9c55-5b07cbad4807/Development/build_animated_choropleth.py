
import os
import io
import numpy as np
import pandas as pd
import shapefile                      # pyshp – the ONLY spatial library used
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 1. Read shapefile with pyshp  (no geopandas, no pyproj)
# ─────────────────────────────────────────────────────────────────────────────
SHP_PATH = "./taxi_zones/taxi_zones.shp"
sf = shapefile.Reader(SHP_PATH)
print(f"Shapefile opened: {len(sf.shapes())} zones")
print(f"Fields: {[f[0] for f in sf.fields[1:]]}")

# Build a dict: LocationID → list of numpy arrays, each array = one polygon ring
# Shape type 5 = Polygon; split by parts
zone_polygons = {}   # {loc_id: [np.array(N,2), ...]}  (exterior rings only)

for sr in sf.shapeRecords():
    rec    = sr.record.as_dict()
    loc_id = int(rec.get("LocationID", 0))
    shape  = sr.shape
    pts    = np.array(shape.points, dtype=np.float64)
    parts  = list(shape.parts) + [len(pts)]
    rings  = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    if rings:
        zone_polygons[loc_id] = rings

print(f"Zones parsed: {len(zone_polygons)}")

# Compute canvas bounds for consistent axes
all_pts = np.vstack([r for rings in zone_polygons.values() for r in rings])
x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
print(f"Coordinate range  X: [{x_min:.0f}, {x_max:.0f}]  Y: [{y_min:.0f}, {y_max:.0f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prepare daily counts from upstream variable
# ─────────────────────────────────────────────────────────────────────────────
_counts = daily_zone_counts.copy()
_counts["date"] = pd.to_datetime(_counts["date"])
_counts["date_str"] = _counts["date"].dt.strftime("%Y-%m-%d")
_counts["LocationID"] = _counts["LocationID"].astype(int)

all_dates = sorted(_counts["date_str"].unique())
print(f"Date range : {all_dates[0]} → {all_dates[-1]}  ({len(all_dates)} days)")

# Fixed global colour scale: use 97th percentile as vmax so very busy hubs
# don't wash out the rest of the map
vmin = 0
vmax = int(_counts["ride_count"].quantile(0.97))
total_rides = int(_counts["ride_count"].sum())
print(f"Colour scale : vmin={vmin}  vmax={vmax:,}  (97th pct)")
print(f"Total rides  : {total_rides:,}")

CMAP_NAME   = "YlOrRd"
cmap        = cm.get_cmap(CMAP_NAME)
norm        = mcolors.Normalize(vmin=vmin, vmax=vmax)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build choropleth frames
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#1D1D20"
FIG_W, FIG_H = 12, 10   # inches — keeps GIF manageable
DPI          = 90        # → frame size ≈ 1080 × 900 px

def make_frame(date_str):
    """Render one choropleth frame; return PNG bytes."""
    _day = _counts[_counts["date_str"] == date_str].set_index("LocationID")["ride_count"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    patches, colours = [], []
    for loc_id, rings in zone_polygons.items():
        count = _day.get(loc_id, 0)
        colour = cmap(norm(count))
        for ring in rings:
            poly = MplPolygon(ring, closed=True)
            patches.append(poly)
            colours.append(colour)

    pc = PatchCollection(patches, facecolors=colours,
                         edgecolors="#1D1D20", linewidths=0.3, zorder=2)
    ax.add_collection(pc)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=30)
    cbar.set_label("Daily Ride Pickups", color="#fbfbff", fontsize=10, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="#909094")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#909094", fontsize=8)
    cbar.outline.set_edgecolor("#444444")

    ax.set_title(f"NYC Taxi Pickup Density  ·  {date_str}",
                 color="#fbfbff", fontsize=14, fontweight="bold", pad=10)

    fig.text(0.5, 0.01,
             f"NYC TLC FHVHV 2025  ·  {total_rides:,} total rides  ·  97th-pct scale",
             ha="center", va="bottom", color="#909094", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Render frames & assemble GIF
# ─────────────────────────────────────────────────────────────────────────────
OUT_PATH = "./heatmap_2025.gif"
N_DAYS   = len(all_dates)
pil_frames = []

for _i, _date_str in enumerate(all_dates):
    if _i % 30 == 0:
        print(f"  Rendering frame {_i + 1}/{N_DAYS}  ({_date_str})")
    _png = make_frame(_date_str)
    _img = Image.open(io.BytesIO(_png)).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    pil_frames.append(_img)

print(f"All {N_DAYS} frames rendered. Assembling GIF…")

# Save animated GIF with Pillow
pil_frames[0].save(
    OUT_PATH,
    format="GIF",
    save_all=True,
    append_images=pil_frames[1:],
    loop=0,
    duration=120,      # ms per frame  → ~8 fps
    optimize=False,
)

gif_size_mb = os.path.getsize(OUT_PATH) / 1_048_576
print(f"\n✅  Animated GIF saved → {OUT_PATH}")
print(f"   Frames : {N_DAYS}")
print(f"   Size   : {gif_size_mb:.1f} MB")
print(f"   FPS    : ~{1000/120:.0f}")
