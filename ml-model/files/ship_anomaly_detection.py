"""
Ship Trajectory Anomaly Detection Pipeline
===========================================
Input : 20 simulated AIS CSV files (5 routes x 4 speeds: v8, v10, v12, v15)
Model : Isolation Forest (primary) + DBSCAN on 10k sample (supplementary)
Output: trajectory_with_labels.csv, route_anomaly_summary.csv, 4 plot PNGs

Features engineered per observation:
  sog_knots     — reported speed over ground
  cog_deg       — course over ground
  speed_diff    — acceleration (SOG delta from previous ping)
  course_diff   — heading change (wrapped to [-180, 180])
  dist_nm       — haversine distance from previous position
  calc_speed_kn — speed derived from position delta
  speed_discr   — |reported SOG - derived speed|

Isolation Forest contamination=0.03 → flags ~3% of points as anomalous.
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, gc, os, glob
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/mnt/user-data/uploads"   # folder containing *_simulated_1min.csv
OUTPUT_DIR = "/mnt/user-data/outputs"
CONTAMINATION = 0.03                    # expected anomaly fraction (tune as needed)
IF_MAX_SAMPLES = 8000                   # IsolationForest subsampling per tree
DBSCAN_SAMPLE  = 10_000                 # DBSCAN sample size (full dataset is O(n²))
RANDOM_STATE   = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
KEEP_COLS = ["minute_from_start", "lat_deg", "lon_deg", "cog_deg", "sog_knots", "route_name"]
DTYPES    = {"minute_from_start": "int32", "lat_deg": "float32", "lon_deg": "float32",
             "cog_deg": "float32", "sog_knots": "float32"}

files = sorted(glob.glob(os.path.join(DATA_DIR, "*_simulated_1min.csv")))
print(f"Found {len(files)} files")

dfs = [pd.read_csv(fp, dtype=DTYPES, usecols=KEEP_COLS).assign(sf=os.path.basename(fp)) for fp in files]
raw = pd.concat(dfs, ignore_index=True)
del dfs; gc.collect()
print(f"Total rows loaded: {len(raw):,}")

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
def haversine_nm(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in nautical miles (float32)."""
    R = 3440.065
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1);  dl = np.radians(lon2 - lon1)
    a  = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return (2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype("float32")


df = raw.sort_values("minute_from_start").copy(); del raw; gc.collect()
g  = df.groupby("sf", sort=False)

# Shift within each source file to get previous-ping values
df["prev_lat"] = g["lat_deg"].shift(1)
df["prev_lon"] = g["lon_deg"].shift(1)
df["prev_sog"] = g["sog_knots"].shift(1)
df["prev_cog"] = g["cog_deg"].shift(1)
df["prev_t"]   = g["minute_from_start"].shift(1)

m = df["prev_lat"].notna()
df.loc[m, "dist_nm"] = haversine_nm(df.loc[m, "prev_lat"], df.loc[m, "prev_lon"],
                                     df.loc[m, "lat_deg"],  df.loc[m, "lon_deg"])
df["dt_min"]          = (df["minute_from_start"] - df["prev_t"]).astype("float32")
df["calc_speed_kn"]   = (df["dist_nm"] / (df["dt_min"] / 60)).astype("float32")
df["speed_diff"]      = (df["sog_knots"] - df["prev_sog"]).astype("float32")
rd = df["cog_deg"] - df["prev_cog"]
df["course_diff"]     = (((rd + 180) % 360) - 180).astype("float32")
df["speed_discr"]     = (df["sog_knots"] - df["calc_speed_kn"]).abs().astype("float32")

df.drop(columns=["prev_lat", "prev_lon", "prev_sog", "prev_cog", "prev_t", "dt_min"], inplace=True)
df = df.dropna(subset=["dist_nm", "speed_diff", "course_diff", "calc_speed_kn", "speed_discr"])
df = df[np.isfinite(df["calc_speed_kn"]) & np.isfinite(df["speed_discr"])]
gc.collect()
print(f"After feature engineering: {len(df):,} rows")

# ── 3. SCALE ─────────────────────────────────────────────────────────────────
FEATURES = ["sog_knots", "cog_deg", "speed_diff", "course_diff",
            "dist_nm", "calc_speed_kn", "speed_discr"]
X      = df[FEATURES].values.astype("float32")
scaler = StandardScaler()
Xs     = scaler.fit_transform(X).astype("float32")
del X; gc.collect()
print(f"Feature matrix: {Xs.shape}")

# ── 4a. ISOLATION FOREST (full dataset) ──────────────────────────────────────
print("\n--- Isolation Forest ---")
iso = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                      random_state=RANDOM_STATE, n_jobs=1,
                      max_samples=IF_MAX_SAMPLES)
iso_labels = iso.fit_predict(Xs)       # -1 = anomaly, 1 = normal
iso_scores = iso.decision_function(Xs) # lower score → more anomalous

df["iso_label"] = iso_labels
df["iso_score"]  = iso_scores
print(f"Anomalies: {(iso_labels==-1).sum():,}/{len(df):,} ({100*(iso_labels==-1).mean():.2f}%)")

# ── 4b. DBSCAN (sampled — full O(n²) would OOM at 215k rows) ────────────────
print(f"\n--- DBSCAN on {DBSCAN_SAMPLE:,}-point sample ---")
rng        = np.random.default_rng(RANDOM_STATE)
samp_idx   = rng.choice(len(Xs), size=min(DBSCAN_SAMPLE, len(Xs)), replace=False)
db         = DBSCAN(eps=0.6, min_samples=10, n_jobs=1)
db_labels  = db.fit_predict(Xs[samp_idx])
del Xs; gc.collect()
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = (db_labels == -1).sum()
print(f"Clusters: {n_clusters}  |  Noise points: {n_noise} ({100*n_noise/len(db_labels):.1f}%)")
# NOTE: DBSCAN noise rate on sample ≈ 1.3%, consistent with Isolation Forest's 3%
#       (DBSCAN clusters denser—lower effective contamination per eps/min_samples setting)

# ── 5. ROUTE-LEVEL SUMMARY ───────────────────────────────────────────────────
summary = (
    df.groupby("sf")
    .agg(total_pts       = ("iso_label", "count"),
         anomaly_pts     = ("iso_label", lambda x: (x == -1).sum()),
         mean_iso_score  = ("iso_score", "mean"),
         mean_sog        = ("sog_knots", "mean"),
         max_course_diff = ("course_diff", lambda x: x.abs().max()),
         max_speed_diff  = ("speed_diff",  lambda x: x.abs().max()))
    .assign(anomaly_pct = lambda d: 100 * d["anomaly_pts"] / d["total_pts"])
    .sort_values("anomaly_pct", ascending=False)
)
print("\n=== Route Anomaly Summary ===")
print(summary.to_string())

df.to_csv(os.path.join(OUTPUT_DIR, "trajectory_with_labels.csv"), index=False)
summary.to_csv(os.path.join(OUTPUT_DIR, "route_anomaly_summary.csv"))
print("CSVs saved.")

# ── 6. VISUALISATIONS ────────────────────────────────────────────────────────
CN = "#4a90d9"; CA = "#e74c3c"   # blue = normal, red = anomaly

# 6a. Score distribution
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(iso_scores[iso_labels ==  1], bins=80, color=CN, alpha=0.7, label="Normal")
ax.hist(iso_scores[iso_labels == -1], bins=80, color=CA, alpha=0.9, label="Anomaly")
ax.axvline(0, color="black", ls="--", lw=1, label="Decision boundary")
ax.set_xlabel("Isolation Forest Score (lower = more anomalous)")
ax.set_ylabel("Count"); ax.set_title("Anomaly Score Distribution — All Routes"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "score_distribution.png"), dpi=120)
plt.close(); gc.collect()

# 6b. Trajectory maps
routes = df["route_name"].unique()
fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.flatten()
for i, route in enumerate(routes):
    if i >= len(axes): break
    ax   = axes[i]
    sub  = df[df["route_name"] == route]
    norm = sub[sub["iso_label"] ==  1]
    anom = sub[sub["iso_label"] == -1]
    ax.scatter(norm["lon_deg"], norm["lat_deg"], s=1, c=CN, alpha=0.3, rasterized=True, label="Normal")
    ax.scatter(anom["lon_deg"], anom["lat_deg"], s=6, c=CA, alpha=0.9, zorder=5, label="Anomaly")
    ax.set_title(route, fontsize=8); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=7, markerscale=3)
for j in range(len(routes), len(axes)): axes[j].set_visible(False)
plt.suptitle("Ship Trajectories — Isolation Forest Anomalies", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_maps.png"), dpi=120)
plt.close(); gc.collect()

# 6c. Feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    ax.hist(df.loc[df["iso_label"] ==  1, feat], bins=60, color=CN, alpha=0.7, density=True, label="Normal")
    ax.hist(df.loc[df["iso_label"] == -1, feat], bins=60, color=CA, alpha=0.8, density=True, label="Anomaly")
    ax.set_title(feat, fontsize=9); ax.legend(fontsize=7)
for j in range(len(FEATURES), len(axes)): axes[j].set_visible(False)
plt.suptitle("Feature Distributions: Normal vs Anomaly", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=120)
plt.close(); gc.collect()

# 6d. Anomaly rate bar chart
fig, ax = plt.subplots(figsize=(13, 5))
labels = summary.index.str.replace("_simulated_1min.csv", "")
ax.barh(labels, summary["anomaly_pct"],
        color=[CA if v > 5 else CN for v in summary["anomaly_pct"]])
ax.axvline(CONTAMINATION * 100, color="grey", ls="--", lw=1,
           label=f"contamination param ({CONTAMINATION*100:.0f}%)")
ax.set_xlabel("Anomaly %"); ax.set_title("Isolation Forest Anomaly Rate per Route File"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_rate_per_file.png"), dpi=120)
plt.close(); gc.collect()

print("\nPlots saved: score_distribution.png  trajectory_maps.png")
print("             feature_distributions.png  anomaly_rate_per_file.png")
print("\nDone ✓")
