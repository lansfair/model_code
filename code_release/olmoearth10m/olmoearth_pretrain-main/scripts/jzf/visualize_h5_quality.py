#!/usr/bin/env python3
"""H5 Dataset Quality Check & Visualization for OlmoEarth Pretraining.

Scans H5 files in parallel, generates a self-contained HTML quality report
with diagnostic visualizations, and writes a CSV of training-ready samples.

Usage:
    python visualize_h5_quality.py /path/to/h5_dir/
    python visualize_h5_quality.py /path/to/h5_dir/ \\
        --output-html report.html --output-csv clean.csv --workers 8
    python visualize_h5_quality.py file_list.txt   # one .h5 path per line
    python visualize_h5_quality.py /path/to/h5_dir/ --max-files 100 --recursive
"""

import argparse
import base64
import csv
import io
import multiprocessing
import os
import random
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - required to decode compressed HDF5
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        desc = kw.get("desc", "")
        if desc:
            print(f"{desc}...", flush=True)
        return it


# ── Dataset-level configuration ───────────────────────────────────────────────
# Change MAX_SEQUENCE_LENGTH here to update the timestep check.
MISSING_VALUE       = -99999
MAX_SEQUENCE_LENGTH = 15   # max allowed timesteps per sample


# ── Modality registry ─────────────────────────────────────────────────────────
# To add a new modality: append one row here — no other edits required.
# To modify an existing modality: change its values here.
#
# Fields:
#   bands      – expected number of bands (last axis size)
#   spacetime  – True if the array has a time axis (shape = H,W,T,C or T,C)
#   zero_ok    – True if high zero ratios are normal (suppresses zero warnings)
#   hist       – (vmin, vmax, nbins) for histogram plots, or None to skip
#   nodata     – integer NODATA sentinel flagged as hard error if >5%, or None
#   preview    – (R_band_idx, G_band_idx, B_band_idx) for RGB thumbnails, or None
MODALITY_PROPERTIES = {
    # ── spacetime-varying modalities ──────────────────────────────────────────
    "sentinel2_l2a":         dict(bands=12, spacetime=True,  zero_ok=False, hist=(0,     10000, 60), nodata=None,   preview=(2, 1, 0)),
    "sentinel1":             dict(bands=2,  spacetime=True,  zero_ok=False, hist=(-50,   0,     60), nodata=-32768, preview=None),
    "landsat":               dict(bands=11, spacetime=True,  zero_ok=False, hist=(0,     10000, 60), nodata=None,   preview=None),
    "planet_rgbnir":         dict(bands=3,  spacetime=True,  zero_ok=False, hist=(0,     10000, 60), nodata=None,   preview=(0, 1, 2)),
    "rgb":                   dict(bands=4,  spacetime=True,  zero_ok=False, hist=(0,     255,   60), nodata=None,   preview=(0, 1, 2)),
    "sar":                   dict(bands=1,  spacetime=True,  zero_ok=False, hist=(0,     255,   60), nodata=None,   preview=None),
    "lt1":                   dict(bands=1,  spacetime=True,  zero_ok=False, hist=(0,     255,   60), nodata=None,   preview=None),
    # ── static modalities ─────────────────────────────────────────────────────
    "srtm":                  dict(bands=1,  spacetime=False, zero_ok=False, hist=(-500,  9000,  60), nodata=None,   preview=None),
    "era5_10":               dict(bands=6,  spacetime=False, zero_ok=False, hist=(-60,   340,   60), nodata=None,   preview=None),
    "worldpop":              dict(bands=1,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "wri_canopy_height_map": dict(bands=1,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "openstreetmap_raster":  dict(bands=30, spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "naip_10":               dict(bands=4,  spacetime=False, zero_ok=False, hist=(0,     10000, 60), nodata=None,   preview=(0, 1, 2)),
    "gse":                   dict(bands=64, spacetime=False, zero_ok=False, hist=None,               nodata=None,   preview=None),
    # ── classification / land-cover maps (zero = no-feature, not an anomaly) ──
    "worldcover":            dict(bands=1,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "worldcereal":           dict(bands=8,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "cdl":                   dict(bands=1,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "eurocrops":             dict(bands=1,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "landcover_1m":          dict(bands=1,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
    "landcover_30m":         dict(bands=1,  spacetime=False, zero_ok=True,  hist=None,               nodata=None,   preview=None),
}

# Derived look-up structures — do NOT edit; edit MODALITY_PROPERTIES above.
SPACETIME_MODALITIES = frozenset(k for k, v in MODALITY_PROPERTIES.items() if v["spacetime"])
ZERO_OK_MODALITIES   = frozenset(k for k, v in MODALITY_PROPERTIES.items() if v["zero_ok"])
_HIST_CFG            = {k: v["hist"]    for k, v in MODALITY_PROPERTIES.items() if v["hist"]}
_NODATA_CHECKS       = {k: v["nodata"]  for k, v in MODALITY_PROPERTIES.items() if v["nodata"] is not None}
_PREVIEW_RGB         = {k: v["preview"] for k, v in MODALITY_PROPERTIES.items() if v["preview"]}

# Compute histograms in only the first N files to bound memory
N_HIST_FILES = 1000
PREVIEW_DISPLAY_PX = 320  # CSS display width (px) for preview thumbnails in HTML


# ─────────────────────────────────────────────────────────────────────────────
# Preview renderer  (defined before worker so it's pickleable on all OSes)
# ─────────────────────────────────────────────────────────────────────────────

def _render_modality_preview(data, rgb_indices=(0, 1, 2)):
    """Return a full-resolution RGB PNG (base64) from a 4-D (H,W,T,C) array.

    rgb_indices: (R_band, G_band, B_band) indices into the last axis.
    No spatial downsampling is applied — display scaling is left to the caller
    (matplotlib imshow / browser CSS), which produces sharper results than
    pre-downsampling a small thumbnail.
    Returns base64-encoded PNG string or None on failure.
    """
    try:
        H, W, T, C = data.shape
        r_i, g_i, b_i = rgb_indices

        # First timestep with > 10% valid pixels in the red band
        best_t = None
        for t in range(T):
            if (data[:, :, t, r_i] != MISSING_VALUE).mean() > 0.1:
                best_t = t
                break
        if best_t is None:
            return None

        rgb = data[:, :, best_t][:, :, [r_i, g_i, b_i]].astype(np.float32)

        if data.dtype == np.uint8:
            out = np.clip(rgb / 255.0, 0.0, 1.0)
        else:
            missing = np.any(rgb == MISSING_VALUE, axis=-1)
            rgb[missing] = np.nan
            out = np.zeros((H, W, 3), dtype=np.float32)
            for i in range(3):
                ch = rgb[:, :, i]
                valid = ch[~np.isnan(ch)]
                if valid.size == 0:
                    continue
                lo, hi = np.percentile(valid, [2, 98])
                hi = max(hi, lo + 1.0)
                out[:, :, i] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
            out = np.nan_to_num(out, nan=0.0)

        buf = io.BytesIO()
        plt.imsave(buf, out, format="png")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker: check one H5 file
# ─────────────────────────────────────────────────────────────────────────────

def check_single_file(args):
    """Check one H5 file.  Runs inside multiprocessing.Pool.

    args = (filepath: str, compute_histograms: bool)
    Returns a JSON-serialisable dict.
    """
    filepath, compute_histograms = args
    result = {
        "filepath":           filepath,
        "filename":           os.path.basename(filepath),
        "file_size_mb":       0.0,
        "latlon":             None,
        "num_timesteps":      None,
        "modalities_present": [],
        "per_modality":       {},
        "hist_data":          {},   # mod → {band_name: {counts, edges}}
        "previews":           {},   # mod → base64 PNG thumbnail
        "errors":             [],
        "warnings":           [],
        "is_training_ready":  True,
    }

    try:
        if not os.path.exists(filepath):
            result["errors"].append("File not found")
            result["is_training_ready"] = False
            return result

        result["file_size_mb"] = os.path.getsize(filepath) / (1024 * 1024)

        with h5py.File(filepath, "r") as f:
            top_keys = set(f.keys())

            # ── latlon ───────────────────────────────────────────────────────
            if "latlon" not in top_keys:
                result["errors"].append("Missing latlon dataset")
            else:
                ll = f["latlon"][()]
                if ll.shape == (2,):
                    result["latlon"] = [float(ll[0]), float(ll[1])]
                else:
                    result["errors"].append(f"latlon wrong shape {ll.shape}")

            # ── timestamps ───────────────────────────────────────────────────
            if "timestamps" not in top_keys:
                result["errors"].append("Missing timestamps dataset")
            else:
                ts = f["timestamps"][()]
                if ts.ndim != 2 or ts.shape[1] != 3:
                    result["errors"].append(f"timestamps wrong shape {ts.shape}, expected (T, 3)")
                else:
                    result["num_timesteps"] = int(ts.shape[0])
                    if ts.shape[0] > MAX_SEQUENCE_LENGTH:
                        result["errors"].append(
                            f"Too many timesteps: {ts.shape[0]} > {MAX_SEQUENCE_LENGTH}"
                        )
                    for i, (day, month, year) in enumerate(ts):
                        if not (0 <= int(month) <= 11):
                            result["warnings"].append(f"timestep {i}: unexpected month={month} (expected 0–11)")
                        if not (1 <= int(day) <= 31):
                            result["warnings"].append(f"timestep {i}: unexpected day={day} (expected 1–31)")

            # ── modality datasets ─────────────────────────────────────────────
            mod_keys = [
                k for k in top_keys
                if k not in ("latlon", "timestamps", "missing_timesteps_masks")
                and isinstance(f[k], h5py.Dataset)
            ]

            preview_data = {}   # key → 4-D array, for modalities that support thumbnails

            for key in mod_keys:
                prop = MODALITY_PROPERTIES.get(key)
                if prop is None:
                    result["warnings"].append(f"Unknown modality key: {key}")
                    continue

                data = f[key][()]
                expected_bands = prop["bands"]
                actual_bands   = int(data.shape[-1]) if data.ndim >= 1 else 1
                num_elements   = int(data.size)

                if actual_bands != expected_bands:
                    result["errors"].append(
                        f"{key}: expected {expected_bands} bands, got {actual_bands}"
                    )

                # Cast to float64 for statistics; use int64 for MISSING check to
                # avoid float-precision issues with -99999 in float16 data.
                as_int = data.astype(np.int64)
                missing_mask  = (as_int == MISSING_VALUE).ravel()
                missing_ratio = float(missing_mask.mean()) if num_elements > 0 else 1.0

                as_f = data.astype(np.float64)
                nan_count = int(np.isnan(as_f).sum())
                inf_count = int(np.isinf(as_f).sum())

                mod_info = {
                    "shape":         list(data.shape),
                    "dtype":         str(data.dtype),
                    "missing_ratio": missing_ratio,
                    "nan_count":     nan_count,
                    "inf_count":     inf_count,
                    "nodata_count":  0,
                }

                # ── Generic NODATA check (derived from registry) ──────────────
                nodata_val = _NODATA_CHECKS.get(key)
                if nodata_val is not None and num_elements > 0:
                    nd_count = int((as_int == nodata_val).sum())
                    nd_ratio = nd_count / num_elements
                    mod_info["nodata_count"] = nd_count
                    if nd_ratio > 0.05:
                        result["errors"].append(
                            f"{key}: {nd_ratio:.1%} NODATA pixels (>{5}% threshold)"
                        )

                valid_mask = ~missing_mask.reshape(data.shape) & np.isfinite(as_f)
                if valid_mask.any():
                    valid_vals = as_f[valid_mask]
                    mod_info["value_min"]  = float(valid_vals.min())
                    mod_info["value_max"]  = float(valid_vals.max())
                    mod_info["value_mean"] = float(valid_vals.mean())
                    mod_info["value_std"]  = float(valid_vals.std())

                    # Per-band means (last axis = bands dimension)
                    if data.ndim >= 2:
                        band_means = []
                        for b in range(actual_bands):
                            bd = as_f[..., b]
                            bd_int = as_int[..., b]
                            valid_b = bd[(bd_int != MISSING_VALUE) & np.isfinite(bd)]
                            band_means.append(float(valid_b.mean()) if valid_b.size else float("nan"))
                        mod_info["band_means"] = band_means

                result["per_modality"][key] = mod_info
                result["modalities_present"].append(key)

                # ── Training-readiness checks ─────────────────────────────────
                if nan_count > 0 or inf_count > 0:
                    result["errors"].append(
                        f"{key}: {nan_count} NaN + {inf_count} Inf values"
                    )

                if missing_ratio > 0.90:
                    result["warnings"].append(f"{key}: {missing_ratio:.0%} pixels are MISSING_VALUE")

                if key not in ZERO_OK_MODALITIES and num_elements > 0:
                    zero_ratio = float((data == 0).mean())
                    if zero_ratio > 0.30:
                        result["warnings"].append(f"{key}: {zero_ratio:.0%} pixels are zero")

                # ── Histogram (subsample every 8th pixel to save time) ────────
                if compute_histograms and key in _HIST_CFG:
                    vmin, vmax, nbins = _HIST_CFG[key]
                    edges = np.linspace(vmin, vmax, nbins + 1)
                    # Use generic band names from registry band count
                    hist_entry = {}
                    for b in range(actual_bands):
                        bname = f"band_{b}"
                        # Extract the same spatial subsample for both float and int views
                        if data.ndim == 4:       # (H, W, T, C)
                            bd = as_f[::8, ::8, :, b].ravel()
                            bi = as_int[::8, ::8, :, b].ravel()
                        elif data.ndim == 3:     # (H, W, C)
                            idx = b if data.shape[-1] > 1 else 0
                            bd = as_f[::8, ::8, idx].ravel()
                            bi = as_int[::8, ::8, idx].ravel()
                        elif data.ndim == 2:     # (T, C) — e.g. era5_10
                            bd = as_f[:, b].ravel()
                            bi = as_int[:, b].ravel()
                        else:
                            bd = as_f.ravel()
                            bi = as_int.ravel()
                        bd = bd[(bi != MISSING_VALUE) & np.isfinite(bd)]
                        counts, _ = np.histogram(bd, bins=edges)
                        hist_entry[bname] = {
                            "counts": counts.tolist(),
                            "edges":  edges.tolist(),
                        }
                    result["hist_data"][key] = hist_entry

                # ── Collect 4-D arrays that can produce RGB previews ──────────
                if key in _PREVIEW_RGB and data.ndim == 4:
                    preview_data[key] = data

            # ── At least one spacetime-varying modality must be non-empty ─────
            present_st = [m for m in result["modalities_present"] if m in SPACETIME_MODALITIES]
            if not present_st:
                result["errors"].append(
                    f"No spacetime-varying modality present "
                    f"(need one of: {', '.join(sorted(SPACETIME_MODALITIES))})"
                )
            else:
                all_fully_missing = all(
                    result["per_modality"][m]["missing_ratio"] >= 1.0
                    for m in present_st
                )
                if all_fully_missing:
                    result["errors"].append(
                        "All spacetime-varying modalities are 100% MISSING_VALUE"
                    )

            # ── Render a thumbnail for every modality that supports preview ──
            for pkey, pidx in _PREVIEW_RGB.items():
                if pkey in preview_data:
                    b64 = _render_modality_preview(preview_data[pkey], pidx)
                    if b64 is not None:
                        result["previews"][pkey] = b64

    except Exception as exc:
        result["errors"].append(f"Fatal error: {exc}\n{traceback.format_exc()}")

    if result["errors"]:
        result["is_training_ready"] = False

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_stats(results):
    total  = len(results)
    n_pass = sum(1 for r in results if r["is_training_ready"])

    error_counts = defaultdict(int)
    for r in results:
        for e in r["errors"]:
            prefix = e.split(":")[0].strip()
            error_counts[prefix] += 1

    modality_presence = defaultdict(int)
    for r in results:
        for m in r["modalities_present"]:
            modality_presence[m] += 1

    return {
        "total":             total,
        "n_pass":            n_pass,
        "n_fail":            total - n_pass,
        "pass_rate":         n_pass / max(total, 1),
        "error_counts":      dict(error_counts),
        "modality_presence": dict(modality_presence),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_geo_plot(results):
    pass_lats, pass_lons = [], []
    fail_lats, fail_lons = [], []
    for r in results:
        if r["latlon"] is None:
            continue
        lat, lon = r["latlon"]
        if r["is_training_ready"]:
            pass_lats.append(lat); pass_lons.append(lon)
        else:
            fail_lats.append(lat); fail_lons.append(lon)

    fig, ax = plt.subplots(figsize=(12, 5))
    if pass_lons:
        ax.scatter(pass_lons, pass_lats, s=5, c="#2ecc71", alpha=0.6,
                   label=f"Pass ({len(pass_lats)})", rasterized=True)
    if fail_lons:
        ax.scatter(fail_lons, fail_lats, s=8, c="#e74c3c", alpha=0.85,
                   label=f"Fail ({len(fail_lats)})", rasterized=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of H5 Samples")
    ax.legend(markerscale=3, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def make_missing_rate_plot(results, agg):
    mod_missing = defaultdict(list)
    for r in results:
        for mod, info in r["per_modality"].items():
            mod_missing[mod].append(info["missing_ratio"])

    mods = sorted(
        [m for m, vals in mod_missing.items() if len(vals) >= 10],
        key=lambda m: -float(np.median(mod_missing[m])),
    )
    if not mods:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Not enough data to plot", transform=ax.transAxes, ha="center")
        return _fig_to_b64(fig)

    data_to_plot = [mod_missing[m] for m in mods]
    fig, ax = plt.subplots(figsize=(max(8, len(mods) * 1.0), 5))
    bp = ax.boxplot(data_to_plot, tick_labels=mods, patch_artist=True, vert=True,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.55)
    ax.axhline(0.90, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="90% MISSING threshold")
    ax.set_ylabel("Missing Value Ratio")
    ax.set_title("Distribution of MISSING_VALUE Ratio per Modality")
    ax.set_ylim(-0.05, 1.07)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def make_band_histograms(results):
    """Aggregate histogram counts from results and return {mod_name: base64_png}."""
    agg_hists = {}
    for r in results:
        for mod_name, band_dict in r.get("hist_data", {}).items():
            if mod_name not in agg_hists:
                agg_hists[mod_name] = {}
            for band_name, h in band_dict.items():
                if band_name not in agg_hists[mod_name]:
                    agg_hists[mod_name][band_name] = {
                        "counts": np.zeros(len(h["counts"]), dtype=np.int64),
                        "edges":  np.array(h["edges"]),
                    }
                agg_hists[mod_name][band_name]["counts"] += np.array(h["counts"], dtype=np.int64)

    out = {}
    for mod_name, band_data in agg_hists.items():
        bands = sorted(band_data.keys())
        n = len(bands)
        if n == 0:
            continue
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.8 * ncols, 3.0 * nrows), squeeze=False)
        vmin, vmax, _ = _HIST_CFG.get(mod_name, (None, None, None))

        for idx, bname in enumerate(bands):
            ax      = axes[idx // ncols][idx % ncols]
            h       = band_data[bname]
            edges   = h["edges"]
            counts  = h["counts"]
            centers = (edges[:-1] + edges[1:]) / 2
            width   = edges[1] - edges[0]
            ax.bar(centers, counts, width=width, color="#3498db", alpha=0.7, edgecolor="none")
            if vmin is not None:
                ax.axvline(vmin, color="#2ecc71", linestyle="--", linewidth=1, label="expected min")
                ax.axvline(vmax, color="#e74c3c", linestyle="--", linewidth=1, label="expected max")
                if idx == 0:
                    ax.legend(fontsize=6)
            ax.set_title(bname, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_ylabel("count", fontsize=7)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"{mod_name} – Band Value Distributions", fontsize=10, y=1.01)
        fig.tight_layout()
        out[mod_name] = _fig_to_b64(fig)

    return out


def make_preview_grid_html(results, modality, preview_count=16):
    """Return an HTML string with full-resolution preview images scaled via CSS.

    Images are embedded at their original pixel resolution; the browser scales
    them to PREVIEW_DISPLAY_PX wide.  No pixel-level resampling is applied.
    """
    candidates = [r for r in results if r["is_training_ready"] and modality in r.get("previews", {})]
    if not candidates:
        return None

    random.shuffle(candidates)
    selected = candidates[:preview_count]

    items = []
    for r in selected:
        b64  = r["previews"][modality]
        miss = r["per_modality"].get(modality, {}).get("missing_ratio", float("nan"))
        label = r["filename"][:24] + f"<br>miss={miss:.0%}"
        items.append(
            f"<div style='display:inline-block;margin:5px;text-align:center;vertical-align:top'>"
            f"<img src='data:image/png;base64,{b64}' "
            f"style='width:{PREVIEW_DISPLAY_PX}px;display:block;image-rendering:auto'>"
            f"<div style='font-size:9px;color:#555;max-width:{PREVIEW_DISPLAY_PX}px;"
            f"word-break:break-all;margin-top:2px'>{label}</div>"
            f"</div>"
        )

    return (
        f"<div style='display:flex;flex-wrap:wrap;gap:4px'>"
        + "".join(items)
        + "</div>"
    )


def make_all_preview_grids(results, preview_count=16):
    """Return {modality: html_string} for every modality that has preview support."""
    out = {}
    for mod in _PREVIEW_RGB:
        grid = make_preview_grid_html(results, mod, preview_count=preview_count)
        if grid is not None:
            out[mod] = grid
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(results, agg, geo_png, missing_png,
                         band_hist_pngs, preview_grids, args_str=""):
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total    = agg["total"]
    n_pass   = agg["n_pass"]
    n_fail   = agg["n_fail"]
    pass_pct = agg["pass_rate"] * 100

    err_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in sorted(agg["error_counts"].items(), key=lambda x: -x[1])
    ) or "<tr><td colspan='2' style='color:#27ae60'>No errors found</td></tr>"

    mod_rows = "\n".join(
        f"<tr><td>{m}</td><td>{c}</td><td>{c/max(total,1)*100:.1f}%</td></tr>"
        for m, c in sorted(agg["modality_presence"].items(), key=lambda x: -x[1])
    )

    sorted_results = sorted(results, key=lambda r: r["is_training_ready"])
    file_rows = []
    for r in sorted_results:
        ok  = r["is_training_ready"]
        cls = "pass" if ok else "fail"
        txt = "✓ PASS" if ok else "✗ FAIL"
        ll  = r["latlon"]
        ll_str    = f"{ll[0]:.3f}, {ll[1]:.3f}" if ll else "–"
        errs_str  = "<br>".join(r["errors"])
        warns_str = "<br>".join(r["warnings"])
        mods_str  = ", ".join(r["modalities_present"])
        file_rows.append(
            f"<tr class='{cls}'>"
            f"<td title='{r['filepath']}'>{r['filename']}</td>"
            f"<td>{r['file_size_mb']:.1f}</td>"
            f"<td style='font-size:0.8em'>{mods_str}</td>"
            f"<td>{r.get('num_timesteps', '')}</td>"
            f"<td>{ll_str}</td>"
            f"<td><b>{txt}</b></td>"
            f"<td style='color:#c0392b;font-size:0.82em'>{errs_str}</td>"
            f"<td style='color:#d35400;font-size:0.82em'>{warns_str}</td>"
            f"</tr>"
        )
    file_table = "\n".join(file_rows)

    hist_sections = ""
    for mod_name, b64 in band_hist_pngs.items():
        hist_sections += (
            f"<h3 style='color:#555;margin-top:20px'>{mod_name}</h3>"
            f"<img src='data:image/png;base64,{b64}' style='max-width:100%'>"
        )

    preview_section = ""
    if preview_grids:
        preview_section = "<h2>RGB Modality Previews</h2>"
        for mod_name, grid_html in preview_grids.items():
            preview_section += (
                f"<h3 style='color:#555;margin-top:20px'>{mod_name}</h3>"
                + grid_html
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OlmoEarth H5 Quality Report</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       margin:0;padding:24px 32px;background:#f4f6f9;color:#2c3e50;}}
  h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:8px;margin-bottom:4px}}
  h2{{color:#34495e;margin-top:36px;border-left:4px solid #3498db;padding-left:10px}}
  .meta{{color:#95a5a6;font-size:.88em;margin-bottom:20px}}
  .cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:28px}}
  .card{{background:#fff;border-radius:10px;padding:18px 28px;
         box-shadow:0 2px 10px rgba(0,0,0,.07);min-width:130px;text-align:center}}
  .card .num{{font-size:2.2em;font-weight:700}}
  .card .lbl{{font-size:.75em;color:#95a5a6;text-transform:uppercase;letter-spacing:.05em}}
  .green{{color:#27ae60}} .red{{color:#e74c3c}} .blue{{color:#2980b9}}
  table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;
         box-shadow:0 2px 10px rgba(0,0,0,.07);margin-bottom:20px;font-size:.84em;
         overflow:hidden}}
  th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #ecf0f1}}
  th{{background:#3498db;color:#fff;font-weight:600;position:sticky;top:0;z-index:1}}
  tr.fail{{background:#fdecea}}
  tr.pass{{background:#eafaf1}}
  tr:hover{{filter:brightness(.97)}}
  details summary{{cursor:pointer;padding:8px 0;font-weight:600;color:#3498db;
                   font-size:1.02em;user-select:none}}
  img{{border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1);display:block;
       margin:10px 0}}
  code{{background:#f0f0f0;padding:2px 6px;border-radius:4px;font-size:.85em}}
  .scroll{{overflow-x:auto}}
</style>
</head>
<body>
<h1>OlmoEarth H5 Quality Report</h1>
<p class="meta">Generated: {ts} &nbsp;|&nbsp; Command: <code>{args_str}</code></p>

<div class="cards">
  <div class="card"><div class="num blue">{total}</div><div class="lbl">Total files</div></div>
  <div class="card"><div class="num green">{n_pass}</div><div class="lbl">Pass&nbsp;({pass_pct:.1f}%)</div></div>
  <div class="card"><div class="num red">{n_fail}</div><div class="lbl">Fail</div></div>
</div>

<h2>Error Summary</h2>
<table style="max-width:600px">
  <tr><th>Error type</th><th>Files affected</th></tr>
  {err_rows}
</table>

<h2>Modality Presence</h2>
<table style="max-width:500px">
  <tr><th>Modality</th><th>File count</th><th>Presence rate</th></tr>
  {mod_rows}
</table>

<h2>Geographic Distribution</h2>
<img src="data:image/png;base64,{geo_png}" style="max-width:100%">

<h2>Missing Data Rate per Modality</h2>
<img src="data:image/png;base64,{missing_png}" style="max-width:100%">

<h2>Per-Band Value Distributions</h2>
{hist_sections or "<p style='color:#7f8c8d'>No histogram data collected (try removing --max-files limit).</p>"}

{preview_section}

<h2>Per-File Details</h2>
<details open>
<summary>Show / hide file table &nbsp;({total} rows, failures first)</summary>
<div class="scroll">
<table>
  <tr>
    <th>Filename</th><th>MB</th><th>Modalities</th>
    <th>T</th><th>Lat, Lon</th><th>Status</th><th>Errors</th><th>Warnings</th>
  </tr>
  {file_table}
</table>
</div>
</details>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def write_clean_csv(results, output_path):
    """Write training-ready files to CSV. Returns row count."""
    passing = [r for r in results if r["is_training_ready"]]
    if not passing:
        print(f"[WARN] No passing samples — CSV not written.", file=sys.stderr)
        return 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "file_size_mb", "modalities_present",
                    "num_timesteps", "lat", "lon"])
        for r in passing:
            ll = r["latlon"] or [None, None]
            w.writerow([
                r["filepath"],
                f"{r['file_size_mb']:.2f}",
                "|".join(r["modalities_present"]),
                r.get("num_timesteps", ""),
                f"{ll[0]:.6f}" if ll[0] is not None else "",
                f"{ll[1]:.6f}" if ll[1] is not None else "",
            ])
    return len(passing)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_h5_files(input_path, recursive=False):
    p = Path(input_path)
    if p.is_file() and p.suffix == ".h5":
        return [str(p)]
    if p.is_file():
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        return [l for l in lines if Path(l).exists()]
    if p.is_dir():
        pattern = "**/*.h5" if recursive else "*.h5"
        return sorted(str(x) for x in p.glob(pattern))
    raise ValueError(f"Input not found or not a file/directory: {input_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check and visualise H5 files for OlmoEarth pretraining quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",
                        help="Directory of .h5 files, a single .h5 file, "
                             "or a text file listing .h5 paths (one per line)")
    parser.add_argument("--output-html",    default="h5_quality_report.html",
                        help="Output HTML report path")
    parser.add_argument("--output-csv",     default="clean_samples.csv",
                        help="Output CSV of passing files")
    parser.add_argument("--workers",        type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="Parallel worker processes")
    parser.add_argument("--max-files",      type=int, default=0,
                        help="Limit files checked (0 = all)")
    parser.add_argument("--preview-count",  type=int, default=16,
                        help="Number of RGB thumbnails in the preview grid")
    parser.add_argument("--recursive",      action="store_true",
                        help="Recurse into subdirectories when scanning a directory")
    args = parser.parse_args()

    # ── Discover files ─────────────────────────────────────────────────────────
    print(f"[1/5] Discovering H5 files in: {args.input}", flush=True)
    all_files = discover_h5_files(args.input, recursive=args.recursive)
    if not all_files:
        print("[ERROR] No H5 files found. Exiting.", file=sys.stderr)
        sys.exit(1)
    if args.max_files > 0:
        all_files = all_files[: args.max_files]
    print(f"       Found {len(all_files)} file(s).", flush=True)

    hist_set    = set(all_files[:N_HIST_FILES])
    worker_args = [(fp, fp in hist_set) for fp in all_files]

    # ── Parallel quality checks ────────────────────────────────────────────────
    print(f"[2/5] Checking files using {args.workers} worker(s)...", flush=True)
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(check_single_file, worker_args, chunksize=8),
                total=len(worker_args),
                desc="Checking H5 files",
            )
        )

    agg = aggregate_stats(results)
    print(
        f"       Done. {agg['n_pass']}/{agg['total']} pass "
        f"({agg['pass_rate']*100:.1f}%).",
        flush=True,
    )

    # ── Visualisations ─────────────────────────────────────────────────────────
    print("[3/5] Generating visualisations...", flush=True)
    geo_png     = make_geo_plot(results)
    missing_png = make_missing_rate_plot(results, agg)
    band_hists  = make_band_histograms(results)
    preview_grids = make_all_preview_grids(results, preview_count=args.preview_count)
    n_figs = 2 + len(band_hists) + len(preview_grids)
    print(f"       {n_figs} figure(s) generated.", flush=True)

    # ── HTML report ────────────────────────────────────────────────────────────
    print(f"[4/5] Writing HTML report  → {args.output_html}", flush=True)
    html = generate_html_report(
        results, agg,
        geo_png, missing_png, band_hists, preview_grids,
        args_str=" ".join(sys.argv),
    )
    Path(args.output_html).write_text(html, encoding="utf-8")

    # ── Clean CSV ──────────────────────────────────────────────────────────────
    print(f"[5/5] Writing clean CSV    → {args.output_csv}", flush=True)
    n_written = write_clean_csv(results, args.output_csv)
    print(f"       {n_written} training-ready sample(s) written.", flush=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
