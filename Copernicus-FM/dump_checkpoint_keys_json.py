#!/usr/bin/env python3
"""Dump all checkpoint keys to JSON for offline Copernicus-FM debugging.

Example:
    python dump_checkpoint_keys_json.py \
        --checkpoint output_dir/checkpoint-999.pth \
        --output checkpoint_keys.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump checkpoint keys to JSON.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <checkpoint>.keys.json.",
    )
    return parser.parse_args()


def value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
    if isinstance(value, dict):
        return {"type": "dict", "len": len(value)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value)}
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def find_weight_dict(checkpoint: Any) -> tuple[str, Any]:
    if not isinstance(checkpoint, dict):
        return "<raw>", checkpoint
    for key in ("model", "state_dict", "model_state_dict", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return key, value
    return "<checkpoint>", checkpoint


def default_output_path(checkpoint_path: Path) -> Path:
    suffix = "".join(checkpoint_path.suffixes)
    if suffix:
        stem = checkpoint_path.name[: -len(suffix)]
        return checkpoint_path.with_name(f"{stem}.keys.json")
    return checkpoint_path.with_name(f"{checkpoint_path.name}.keys.json")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else default_output_path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weight_key, state = find_weight_dict(checkpoint)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": type(checkpoint).__name__,
        "selected_weight_dict": weight_key,
        "selected_type": type(state).__name__,
    }

    if isinstance(checkpoint, dict):
        result["top_level_key_count"] = len(checkpoint)
        result["top_level_keys"] = list(checkpoint.keys())
        result["top_level_key_info"] = {
            key: value_summary(value) for key, value in checkpoint.items()
        }

    if not isinstance(state, dict):
        result["error"] = "Selected weight object is not a dict."
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {output_path}")
        return

    keys = list(state.keys())
    tensor_keys = [key for key in keys if isinstance(state[key], torch.Tensor)]
    non_tensor_keys = [key for key in keys if not isinstance(state[key], torch.Tensor)]

    result.update(
        {
            "selected_key_count": len(keys),
            "selected_keys": keys,
            "tensor_key_count": len(tensor_keys),
            "tensor_keys": tensor_keys,
            "non_tensor_key_count": len(non_tensor_keys),
            "non_tensor_keys": non_tensor_keys,
            "tensor_key_info": {
                key: {
                    "shape": list(state[key].shape),
                    "dtype": str(state[key].dtype),
                    "numel": state[key].numel(),
                }
                for key in tensor_keys
            },
            "non_tensor_key_info": {
                key: value_summary(state[key]) for key in non_tensor_keys
            },
        }
    )

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
