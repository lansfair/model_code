#!/usr/bin/env python3
"""Print checkpoint keys and tensor shapes for Copernicus-FM debugging.

Example:
    python print_checkpoint_keys.py --checkpoint output_dir/checkpoint-999.pth
    python print_checkpoint_keys.py --checkpoint weights/foo.pth --limit 200
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect checkpoint keys.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Maximum number of tensor keys to print.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Only print tensor keys starting with this prefix.",
    )
    parser.add_argument(
        "--top-prefix-depth",
        type=int,
        default=2,
        help="Prefix depth used for summary counts, split by '.'.",
    )
    return parser.parse_args()


def shape_of(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    return type(value).__name__


def find_weight_dict(checkpoint: Any) -> tuple[str, Any]:
    if not isinstance(checkpoint, dict):
        return "<raw>", checkpoint
    for key in ("model", "state_dict", "model_state_dict", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return key, value
    return "<checkpoint>", checkpoint


def prefix_of(key: str, depth: int) -> str:
    parts = key.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else key


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu")

    print(f"checkpoint: {path}")
    print(f"checkpoint_type: {type(checkpoint).__name__}")

    if isinstance(checkpoint, dict):
        print(f"top_level_key_count: {len(checkpoint)}")
        print("top_level_keys:")
        for key in checkpoint.keys():
            print(f"  {key}: {shape_of(checkpoint[key])}")

    weight_key, state = find_weight_dict(checkpoint)
    print(f"\nselected_weight_dict: {weight_key}")
    print(f"selected_type: {type(state).__name__}")
    if not isinstance(state, dict):
        print("selected object is not a dict; stop.")
        return

    keys = list(state.keys())
    tensor_keys = [key for key in keys if isinstance(state[key], torch.Tensor)]
    non_tensor_keys = [key for key in keys if not isinstance(state[key], torch.Tensor)]

    print(f"selected_key_count: {len(keys)}")
    print(f"tensor_key_count: {len(tensor_keys)}")
    print(f"non_tensor_key_count: {len(non_tensor_keys)}")
    if non_tensor_keys:
        print("first_non_tensor_keys:")
        for key in non_tensor_keys[:20]:
            print(f"  {key}: {shape_of(state[key])}")

    prefix_counts = Counter(prefix_of(key, args.top_prefix_depth) for key in tensor_keys)
    print(f"\nprefix_counts_depth_{args.top_prefix_depth}:")
    for prefix, count in prefix_counts.most_common(80):
        print(f"  {prefix}: {count}")

    filtered = tensor_keys
    if args.prefix is not None:
        filtered = [key for key in tensor_keys if key.startswith(args.prefix)]

    print("\ntensor_keys:")
    for key in filtered[: args.limit]:
        value = state[key]
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    if len(filtered) > args.limit:
        print(f"  ... ({len(filtered) - args.limit} more)")

    interesting_prefixes = (
        "patch_embed",
        "patch_embed_spectral",
        "patch_embed_variable",
        "blocks",
        "pos_embed",
        "coord_fc",
        "scale_fc",
        "time_fc",
        "decoder",
        "decoder_",
        "teacher",
        "student_proj",
        "module.",
        "model.",
        "backbone.",
        "backbone.cfm.",
        "cfm.",
        "encoder.",
    )
    print("\ninteresting_prefix_presence:")
    for prefix in interesting_prefixes:
        matched = [key for key in tensor_keys if key.startswith(prefix)]
        if matched:
            print(f"  {prefix}: {len(matched)}; first={matched[:5]}")


if __name__ == "__main__":
    main()
