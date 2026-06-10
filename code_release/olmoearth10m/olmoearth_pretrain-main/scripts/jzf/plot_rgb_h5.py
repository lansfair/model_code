#!/usr/bin/env python3
"""Plot rgb modality from H5 samples as 1024×1024 PNG images.

Usage:
    python plot_rgb_h5.py sample_0000.h5                  # one file
    python plot_rgb_h5.py /path/to/h5_dir/                # whole directory
    python plot_rgb_h5.py /path/to/h5_dir/ --max 20       # first 20 files
    python plot_rgb_h5.py /path/to/h5_dir/ --out my_dir/  # custom output dir
"""

import argparse
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — required to decode compressed HDF5
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MISSING_VALUE = -99999
# rgb band order: R=0, G=1, B=2, NIR=3
RGB_BANDS = (0, 1, 2)
# raw value range from the registry hist entry
RAW_MIN, RAW_MAX = 0, 255


def pick_best_timestep(data: np.ndarray) -> int:
    """Return the timestep index with the most valid (non-missing) pixels."""
    H, W, T, _ = data.shape
    valid_counts = [
        (data[:, :, t, RGB_BANDS[0]] != MISSING_VALUE).sum()
        for t in range(T)
    ]
    return int(np.argmax(valid_counts))


def to_uint8(channel: np.ndarray) -> np.ndarray:
    """Clip to [RAW_MIN, RAW_MAX] and scale to uint8."""
    clipped = np.clip(channel.astype(np.float32), RAW_MIN, RAW_MAX)
    return ((clipped - RAW_MIN) / (RAW_MAX - RAW_MIN) * 255).astype(np.uint8)


def plot_rgb_from_h5(h5_path: Path, out_dir: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        if "rgb" not in f:
            print(f"  [SKIP] no 'rgb' key in {h5_path.name}")
            return
        data = f["rgb"][()]          # shape: (H, W, T, C)

    if data.ndim != 4 or data.shape[-1] < 3:
        print(f"  [SKIP] unexpected rgb shape {data.shape} in {h5_path.name}")
        return

    t = pick_best_timestep(data)
    frame = data[:, :, t, :]        # (H, W, C)

    # Build RGB uint8 image; missing pixels → black
    rgb = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    valid = frame[:, :, RGB_BANDS[0]] != MISSING_VALUE
    for out_c, in_c in enumerate(RGB_BANDS):
        ch = frame[:, :, in_c].copy()
        ch[~valid] = 0
        rgb[:, :, out_c] = to_uint8(ch)

    out_path = out_dir / (h5_path.stem + f"_rgb_t{t}.png")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=128)   # 1024 px output
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(f"{h5_path.name}  |  timestep {t}", fontsize=9, pad=4)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [OK]   {out_path.name}  ({rgb.shape[0]}×{rgb.shape[1]}px, t={t})")


def main():
    parser = argparse.ArgumentParser(description="Plot rgb modality from H5 samples.")
    parser.add_argument("input", help="H5 file or directory of H5 files")
    parser.add_argument("--out", default="rgb_plots",
                        help="Output directory for PNG files (default: ./rgb_plots)")
    parser.add_argument("--max", type=int, default=0,
                        help="Max files to process (0 = all)")
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.h5"))
    else:
        print(f"[ERROR] Not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.max > 0:
        files = files[: args.max]

    if not files:
        print("[ERROR] No H5 files found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(files)} file(s) → {out_dir}/")

    for h5_path in files:
        try:
            plot_rgb_from_h5(h5_path, out_dir)
        except Exception as exc:
            print(f"  [ERR]  {h5_path.name}: {exc}")


if __name__ == "__main__":
    main()
