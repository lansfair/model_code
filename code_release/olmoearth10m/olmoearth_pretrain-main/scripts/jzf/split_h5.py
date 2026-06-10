"""
Split H5 geospatial files into smaller spatial crops.
MULTIPROCESSING VERSION (truly parallel via multiprocessing.Pool).

Crop size is specified in logical (image_tile_size_factor=1) pixel space.
Modalities with a larger image_tile_size_factor are automatically cropped
to a proportionally larger pixel region. For example, with --crop_size 128:
  - factor=1 modality  → 128×128 crop
  - factor=4 modality  → 512×512 crop

Example usage:
    # Split 256×256 into 4 files of 128×128
    python scripts/jzf/split_h5.py --src_dir data/dataset2 --crop_size 128 \\
        --output_dir data/dataset2_split

    # Split with overlapping crops (50% overlap)
    python scripts/jzf/split_h5.py --src_dir data/dataset2 --crop_size 128 --stride 64 \\
        --output_dir data/dataset2_split

    # Drop crops where >80% pixels are zero or MISSING_VALUE
    python scripts/jzf/split_h5.py --src_dir data/dataset2 --crop_size 128 \\
        --output_dir data/dataset2_split --missing_threshold 0.8
"""

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

MISSING_VALUE = -99999
_IMAGE_TILE_SIZE = 256  # matches constants.IMAGE_TILE_SIZE

# image_tile_size_factor per modality (mirrors constants.py Modality definitions).
# Keys absent from this dict are treated as unknown (factor inferred from dims).
_MODALITY_FACTORS: dict[str, int] = {
    "naip": 1,
    "naip_10": 4,
    "sentinel1": 1,
    "sentinel2": 1,
    "sentinel2_l2a": 1,
    "landsat": 1,
    "worldcover": 1,
    "worldcereal": 1,
    "srtm": 1,
    "openstreetmap": 1,
    "openstreetmap_raster": 1,
    "era5": 1,
    "era5_10": -256,
    "gse": 1,
    "cdl": 1,
    "worldpop": 1,
    "wri_canopy_height_map": 1,
    "ndvi": 1,
    "eurocrops": 1,
    "planet_rgbnir": 4,
    "rgb": 4,
    "sar": 4,
    "lt1": 4,
    "landcover_10m": 1,
    "landcover_30m": 1,
}

NON_SPATIAL_KEYS = {"timestamps", "latlon"}
SKIP_GROUPS = {"missing_timesteps_masks"}


def _get_factor(key: str) -> int | None:
    """Return image_tile_size_factor for a known modality, or None if unknown."""
    return _MODALITY_FACTORS.get(key)


def _expected_tile_size(factor: int) -> int:
    """Compute expected tile pixel size from image_tile_size_factor."""
    if factor < 0:
        return _IMAGE_TILE_SIZE // abs(factor)
    return _IMAGE_TILE_SIZE * factor


def _spatial_keys(h5path: Path) -> list[str]:
    """Return keys in the H5 file that represent spatial data."""
    with h5py.File(h5path, "r") as f:
        keys = []
        for k in f:
            if k in NON_SPATIAL_KEYS or k in SKIP_GROUPS:
                continue
            if f[k].ndim < 2:
                continue
            # Exclude modalities whose expected tile size is ≤ 1 (time-only, e.g. era5_10)
            factor = _MODALITY_FACTORS.get(str(k))
            if factor is not None and _expected_tile_size(factor) <= 1:
                continue
            keys.append(k)
        return keys


def _crop_grid(orig_h: int, orig_w: int, crop_h: int, crop_w: int, stride_h: int, stride_w: int):
    """
    Yield (row, col, h0, h1, w0, w1) for each crop.
    Edge crops are snapped to the border to ensure full coverage with no gaps.
    """
    r, h0 = 0, 0
    while True:
        h1 = h0 + crop_h
        if h1 > orig_h:
            h0 = orig_h - crop_h
            h1 = orig_h

        c, w0 = 0, 0
        while True:
            w1 = w0 + crop_w
            if w1 > orig_w:
                w0 = orig_w - crop_w
                w1 = orig_w

            yield r, c, h0, h1, w0, w1

            if w1 == orig_w:
                break
            c += 1
            w0 += stride_w

        if h1 == orig_h:
            break
        r += 1
        h0 += stride_h


def _make_dataset(dst_group, name: str, data: np.ndarray) -> None:
    """Create dataset with zstd level-3 compression."""
    dst_group.create_dataset(name, data=data, **hdf5plugin.Zstd(clevel=3))  # type: ignore[attr-defined]


def _bad_ratio(arr: np.ndarray) -> float:
    """Fraction of elements equal to 0 or MISSING_VALUE."""
    flat = arr.ravel().astype(np.float32)
    return float(np.sum((flat == 0) | (flat == MISSING_VALUE)) / flat.size)


def _slice_for_key(
    key: str,
    factor: int | None,
    h0: int, h1: int,
    w0: int, w1: int,
    logical_h: int,
    logical_w: int,
    src: h5py.File,
) -> tuple[int, int, int, int]:
    """
    Convert logical crop coordinates to actual pixel coordinates for a modality.
    Known modalities use image_tile_size_factor; unknown ones use a dimension ratio.
    """
    if factor is not None:
        if factor >= 1:
            return h0 * factor, h1 * factor, w0 * factor, w1 * factor
        else:
            n = abs(factor)
            return h0 // n, h1 // n, w0 // n, w1 // n
    else:
        # Unknown modality: infer scale from actual vs logical dims
        actual_h = src[key].shape[0]
        actual_w = src[key].shape[1]
        sf_h = actual_h / logical_h
        sf_w = actual_w / logical_w
        return int(h0 * sf_h), int(h1 * sf_h), int(w0 * sf_w), int(w1 * sf_w)


def split_one(
    h5f: Path,
    output_dir: Path,
    crop_h: int,
    crop_w: int,
    stride_h: int,
    stride_w: int,
    missing_threshold: float,
) -> tuple[int, int]:
    """Process a single H5 file. Returns (written, skipped) crop counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial = _spatial_keys(h5f)
    if not spatial:
        print(f"[skip] no spatial data: {h5f.name}")
        return 0, 0

    try:
        with h5py.File(h5f, "r") as src:
            mod_factors = {k: _get_factor(k) for k in spatial}

            # Determine logical (factor=1) dimensions from the first spatial key
            ref_key = spatial[0]
            f0 = mod_factors[ref_key]
            actual_h = src[ref_key].shape[0]
            actual_w = src[ref_key].shape[1]
            if f0 is None:
                logical_h, logical_w = actual_h, actual_w
            elif f0 >= 1:
                logical_h = actual_h // f0
                logical_w = actual_w // f0
            else:
                logical_h = actual_h * abs(f0)
                logical_w = actual_w * abs(f0)

            crops = list(_crop_grid(logical_h, logical_w, crop_h, crop_w, stride_h, stride_w))
            written = skipped = 0

            for r, c, h0, h1, w0, w1 in crops:
                out_path = output_dir / f"{h5f.stem}_r{r:02d}_c{c:02d}.h5"

                # Pre-read all spatial crops and check quality threshold
                cropped_arrays: dict[str, np.ndarray] = {}
                bad = False
                for key in spatial:
                    ah0, ah1, aw0, aw1 = _slice_for_key(
                        key, mod_factors[key], h0, h1, w0, w1, logical_h, logical_w, src
                    )
                    arr = src[key][ah0:ah1, aw0:aw1]
                    if _bad_ratio(arr) > missing_threshold:
                        bad = True
                        break
                    cropped_arrays[key] = arr

                if bad:
                    skipped += 1
                    continue

                with h5py.File(out_path, "w") as dst:
                    for key in src:
                        obj = src[key]

                        if key in SKIP_GROUPS:
                            grp = dst.create_group(key)
                            for mk in obj:
                                grp.create_dataset(mk, data=obj[mk][()])
                            continue

                        if key not in spatial:
                            dst.create_dataset(key, data=obj[()])
                            continue

                        _make_dataset(dst, key, cropped_arrays[key])

                written += 1

        print(f"[done] {h5f.name}: {written} written, {skipped} skipped")
        return written, skipped

    except Exception as e:
        print(f"[fail] {h5f.name}: {e}")
        return 0, 0

# ---------------- 放到全局位置（split_one 函数下方、main 上方）----------------
def _star(args: tuple) -> tuple[int, int]:
    return split_one(*args)

def main():
    parser = argparse.ArgumentParser(
        description="Split H5 files into smaller spatial crops (multiprocessing)")
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Source directory containing H5 files")
    parser.add_argument("--crop_size", type=int, nargs="+", required=True,
                        help="Crop size in logical (factor=1) pixels: single int for square, or 'H W'")
    parser.add_argument("--stride", type=int, nargs="+", default=None,
                        help="Stride between crops in logical pixels (default: same as crop_size)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for split H5 files")
    parser.add_argument("--pattern", type=str, default="sample_*.h5",
                        help="Glob pattern for input H5 files (default: sample_*.h5)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--missing_threshold", type=float, default=0.9,
                        help="Discard a crop if (zeros + MISSING_VALUE) fraction exceeds this (default: 0.9)")

    args = parser.parse_args()

    crop_h, crop_w = args.crop_size[0], args.crop_size[-1]
    if args.stride:
        stride_h, stride_w = args.stride[0], args.stride[-1]
    else:
        stride_h, stride_w = crop_h, crop_w

    if stride_h > crop_h or stride_w > crop_w:
        print(f"[warn] stride ({stride_h}x{stride_w}) > crop ({crop_h}x{crop_w}): gaps may appear between crops")

    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)
    h5_files = sorted(src_dir.glob(args.pattern))

    if not h5_files:
        print(f"No files matching '{args.pattern}' in {src_dir}")
        return

    num_workers = args.num_workers or cpu_count()

    print(f"Found {len(h5_files)} H5 files")
    print(f"Crop (logical): {crop_h}x{crop_w} | Stride: {stride_h}x{stride_w}")
    print(f"Missing threshold: {args.missing_threshold}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_dir}\n")

    task_args = [
        (h5f, output_dir, crop_h, crop_w, stride_h, stride_w, args.missing_threshold)
        for h5f in h5_files
    ]


    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_star, task_args),
            total=len(h5_files),
            desc="Splitting",
            unit="file",
        ))

    total_written = sum(w for w, _ in results)
    total_skipped = sum(s for _, s in results)

    print(f"\nDone! Input: {len(h5_files)} files | Written: {total_written} | Skipped (threshold): {total_skipped}")
    print(f"Output path: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
