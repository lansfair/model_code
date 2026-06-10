# visualize_h5_quality.py

A parallel H5 dataset quality checker and visualizer for OlmoEarth pretraining.
Scans any number of `.h5` sample files, flags degenerate or corrupt samples, and
produces a self-contained HTML report with diagnostic plots plus a filtered CSV of
training-ready files.

The script is **self-contained** — it does not import from `check_h5_sample.py`.
All modality metadata, `MISSING_VALUE`, and `MAX_SEQUENCE_LENGTH` are defined
directly at the top of the file.

---

## Motivation

Malformed H5 samples (all-MISSING modalities, NaN/Inf values, missing spacetime-varying
data) cause silent training failures such as `num_encoded_tokens == 0` in
`nn/pooling.py`. This script lets you audit an entire dataset before launching a
training run and produces a clean file list you can feed directly into the training
config.

---

## Quick start

```bash
# Check all files in a directory (uses all CPUs - 1 by default)
conda run -n olmoearth python3 scripts/jzf/visualize_h5_quality.py /path/to/h5_dir/

# Limit to 200 files for a quick sanity check
conda run -n olmoearth python3 scripts/jzf/visualize_h5_quality.py /path/to/h5_dir/ \
    --max-files 200 --workers 8

# Recurse into subdirectories, custom output paths
conda run -n olmoearth python3 scripts/jzf/visualize_h5_quality.py /path/to/h5_dir/ \
    --recursive \
    --output-html report.html \
    --output-csv clean_samples.csv \
    --workers 16

# Pass a text file with one .h5 path per line
conda run -n olmoearth python3 scripts/jzf/visualize_h5_quality.py file_list.txt
```

Open `report.html` in any browser — no server required, all images are embedded.

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `input` | *(required)* | Directory of `.h5` files, a single `.h5` file, or a text file listing paths (one per line) |
| `--output-html` | `h5_quality_report.html` | Path for the self-contained HTML report |
| `--output-csv` | `clean_samples.csv` | Path for the filtered CSV of passing files |
| `--workers` | `cpu_count - 1` | Number of parallel worker processes |
| `--max-files` | `0` (all) | Cap on total files checked; useful for quick spot-checks |
| `--preview-count` | `16` | Number of RGB thumbnails in the preview grid |
| `--recursive` | off | Recurse into subdirectories when scanning a directory |

---

## What is checked

### Hard errors → sample is marked **FAIL** (excluded from CSV)

| Check | Details |
|---|---|
| **No spacetime-varying modality** | None of `sentinel2_l2a`, `sentinel1`, `landsat`, `planet_rgbnir`, `rgb`, `sar`, `lt1` present |
| **All spacetime modalities 100% MISSING** | Every pixel in every spacetime modality equals `MISSING_VALUE = -99999` |
| **NaN or Inf values** | Any NaN or Inf pixel in any modality |
| **NODATA threshold exceeded** | More than 5% of pixels equal the modality's NODATA sentinel (currently `sentinel1 = -32768`) |
| **Wrong band count** | Actual last-axis size doesn't match the expected band count in `MODALITY_PROPERTIES` |
| **Missing required keys** | `latlon` or `timestamps` dataset absent |
| **Too many timesteps** | `T > MAX_SEQUENCE_LENGTH (15)` |

### Warnings → sample still **PASS**, flagged in report

| Warning | Threshold |
|---|---|
| High MISSING ratio | Any modality has > 90% `MISSING_VALUE` pixels |
| Unexpected timestamp | `month` outside `[0, 11]` or `day` outside `[1, 31]` |
| High zero ratio | > 30% zero pixels in non-classification modalities |

> **Zero-value warnings are suppressed** for modalities where zero is semantically
> valid: `openstreetmap_raster`, `wri_canopy_height_map`, `worldpop`, `worldcover`,
> `worldcereal`, `cdl`, `eurocrops`, `landcover_1m`, `landcover_30m`.

---

## Adding or modifying modalities

All modality configuration lives in the `MODALITY_PROPERTIES` dict near the top of
the script. To add a new modality, append **one row**:

```python
MODALITY_PROPERTIES = {
    ...
    "my_new_modality": dict(
        bands=4,           # expected number of bands (last axis)
        spacetime=True,    # True if array has a time axis
        zero_ok=False,     # True to suppress high-zero warnings
        hist=(0, 255, 60), # (vmin, vmax, nbins) for histogram, or None
        nodata=None,       # integer NODATA sentinel for >5% check, or None
        preview=None,      # (R, G, B) band indices for thumbnail, or None
    ),
}
```

Everything downstream (`SPACETIME_MODALITIES`, `ZERO_OK_MODALITIES`, histogram
config, NODATA checks, preview rendering) is derived automatically — no other
edits required.

### Changing the max sequence length

Edit the constant at the top of the file:

```python
MAX_SEQUENCE_LENGTH = 15   # change here
```

---

## HTML report contents

The single-file HTML report (`report.html`) contains:

1. **Summary cards** — total files, pass count, fail count, pass rate
2. **Error summary table** — grouped error types and how many files each affects
3. **Modality presence table** — which modalities appear and how often
4. **Geographic distribution** — lat/lon scatter plot, green = pass, red = fail
5. **Missing data rate per modality** — box plot of `missing_ratio` distributions,
   with a 90% threshold line
6. **Per-band value histograms** — aggregated from up to 1000 files for modalities
   that have a `hist` range defined (`sentinel2_l2a`, `sentinel1`, `srtm`, `era5_10`,
   `landsat`, `planet_rgbnir`, `rgb`, `sar`, `lt1`, etc.); dashed lines show expected
   value range
7. **RGB modality preview grid** — up to `--preview-count` thumbnails rendered from
   the first modality with `preview` indices set (preference order:
   `sentinel2_l2a` → `rgb` → others); percentile-stretched [2, 98]
8. **Per-file details table** — sortable columns: filename, size, modalities,
   timesteps, lat/lon, pass/fail status, errors, warnings

---

## CSV output format

`clean_samples.csv` contains **only files that passed all hard checks**:

```
filepath,file_size_mb,modalities_present,num_timesteps,lat,lon
/data/sample_0.h5,4.41,sentinel2_l2a|sentinel1|worldcover|srtm,12,50.680042,22.528675
/data/sample_1.h5,4.37,sentinel2_l2a|sentinel1|worldcover|srtm,12,50.682100,22.531000
```

`modalities_present` is pipe-separated. You can use this CSV as a file list for
the `OlmoEarthDatasetConfig` or to filter the dataset index.

---

## Performance

On a 24-core machine, ~1000 files per minute with `--workers 16`. Histogram
computation is capped at the first 1000 files to bound memory; quality checks
(pass/fail) still run on every file.

---

## Dependencies

All available in the `olmoearth` conda environment — no new packages needed:

| Package | Use |
|---|---|
| `h5py` + `hdf5plugin` | Reading compressed HDF5 files |
| `numpy` | Statistics and histogram computation |
| `matplotlib` | All plots and RGB thumbnail rendering |
| `tqdm` | Progress bar (optional; falls back gracefully) |

---

## Related scripts

| Script | Purpose |
|---|---|
| `check_h5_sample.py` | Detailed text-only inspection of one or a few H5 files |
| `diagnose_timesteps.py` | Diagnose timestamp alignment issues |
| `fix_timesteps.py` | Fix timestamp padding in H5 files |
| `generate_sample_metadata.py` | Regenerate `sample_metadata.csv` from H5 directory |
