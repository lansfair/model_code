#!/usr/bin/env python3
"""Export a Copernicus-FM backbone checkpoint from a pretraining checkpoint.

The pretraining script saves a full MAE training checkpoint. That checkpoint
contains encoder weights, decoder weights, optimizer state, scaler state, and
distillation-only modules. Downstream Copernicus-FM code expects a compact
backbone checkpoint whose ``model`` entry can be loaded into ``vit_*_patch16``.

Example:
    python export_copernicusfm_backbone.py \
        --input output_dir/checkpoint-999.pth \
        --output weights/CopernicusFM_ViT_base_custom.pth \
        --arch base
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Copernicus-FM encoder/backbone weights."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the pretraining checkpoint, e.g. checkpoint-999.pth.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the exported backbone checkpoint.",
    )
    parser.add_argument(
        "--arch",
        default="base",
        choices=("small", "base", "large", "huge"),
        help="Target Copernicus-FM backbone size. Must match pretraining.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Target model image size used only for shape validation.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help=(
            "Target patch size used only for shape validation. Defaults to 16 "
            "for small/base/large and 14 for huge."
        ),
    )
    parser.add_argument(
        "--global-pool",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build target model with global pooling for validation. Official "
            "README examples use --no-global-pool."
        ),
    )
    parser.add_argument(
        "--no-validate-shape",
        action="store_true",
        help="Skip target model construction and export by key filtering only.",
    )
    parser.add_argument(
        "--state-key",
        default="model",
        help=(
            "Top-level key used in the exported checkpoint. Use 'none' to save "
            "a raw state_dict instead of {'model': state_dict}."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Allow missing target encoder keys after shape validation. Useful "
            "when exporting for a segmentation variant without cls norm/head."
        ),
    )
    return parser.parse_args()


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)}")
    return checkpoint


def strip_known_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "model.encoder.",
        "model.backbone.",
        "model.backbone.cfm.",
        "backbone.cfm.",
        "backbone.encoder.",
        "backbone.",
        "cfm.",
        "encoder.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def is_training_only_key(key: str) -> bool:
    training_only_prefixes = (
        "decoder_",
        "decoder.",
        "decoder_blocks.",
        "decoder_norm.",
        "decoder_pred_",
        "coord_fc_decoder.",
        "scale_fc_decoder.",
        "time_fc_decoder.",
        "teacher.",
        "student_proj.",
        "teacher_avgpool.",
        "cos.",
    )
    training_only_keys = {
        "mask_token",
        "decoder_pos_embed",
        "coord_token_dec",
        "scale_token_dec",
        "time_token_dec",
    }
    return key in training_only_keys or key.startswith(training_only_prefixes)


def normalize_pretrain_state(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    exported: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for raw_key, value in state.items():
        key = strip_known_prefixes(raw_key)
        if is_training_only_key(key):
            dropped.append(raw_key)
            continue
        if not isinstance(value, torch.Tensor):
            dropped.append(raw_key)
            continue
        exported[key] = value
    return exported, dropped


def load_model_vit_module():
    """Load model_vit.py as a synthetic package so relative imports work."""
    package_name = "_copernicusfm_export_pkg"
    package_spec = importlib.util.spec_from_loader(package_name, loader=None)
    package = importlib.util.module_from_spec(package_spec)
    package.__path__ = [str(SCRIPT_DIR)]
    sys.modules[package_name] = package

    module_name = f"{package_name}.model_vit"
    module_path = SCRIPT_DIR / "model_vit.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_target_state(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    module = load_model_vit_module()
    factory_name = f"vit_{args.arch}_patch16"
    if args.arch == "huge":
        factory_name = "vit_huge_patch14"
    factory = getattr(module, factory_name)
    patch_size = args.patch_size
    if patch_size is None:
        patch_size = 14 if args.arch == "huge" else 16
    model = factory(
        img_size=args.img_size,
        patch_size=patch_size,
        global_pool=args.global_pool,
    )
    return model.state_dict()


def filter_by_target_shape(
    exported: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    allow_missing: bool,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[tuple[str, Any, Any]]]:
    filtered: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatch: list[tuple[str, Any, Any]] = []

    for key, value in exported.items():
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        filtered[key] = value

    missing = sorted(set(target) - set(filtered))
    ignorable_missing = {
        "head.weight",
        "head.bias",
        "fc_norm.weight",
        "fc_norm.bias",
        "norm.weight",
        "norm.bias",
    }
    hard_missing = [key for key in missing if key not in ignorable_missing]
    if hard_missing and not allow_missing:
        preview = ", ".join(hard_missing[:20])
        raise RuntimeError(
            "Missing target backbone keys after export. This usually means "
            f"--arch/--img-size/--patch-size is wrong. First missing: {preview}"
        )
    return filtered, missing, unexpected, shape_mismatch


def save_export(output: Path, state: dict[str, torch.Tensor], state_key: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if state_key.lower() == "none":
        payload: Any = state
    else:
        payload = {state_key: state}
    torch.save(payload, output)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    raw_state = load_checkpoint_state(input_path)
    exported, dropped_training = normalize_pretrain_state(raw_state)

    missing: list[str] = []
    unexpected: list[str] = []
    shape_mismatch: list[tuple[str, Any, Any]] = []
    if not args.no_validate_shape:
        target = build_target_state(args)
        exported, missing, unexpected, shape_mismatch = filter_by_target_shape(
            exported,
            target,
            allow_missing=args.allow_missing,
        )
        if shape_mismatch:
            preview = "; ".join(
                f"{key}: {src} -> {dst}"
                for key, src, dst in shape_mismatch[:10]
            )
            raise RuntimeError(
                "Shape mismatch while exporting Copernicus-FM backbone. "
                "Check --arch, --img-size and --patch-size. First mismatches: "
                f"{preview}"
            )

    save_export(output_path, exported, args.state_key)

    print(f"Loaded checkpoint: {input_path}")
    print(f"Exported checkpoint: {output_path}")
    print(f"Exported tensors: {len(exported)}")
    print(f"Dropped training-only/non-tensor keys: {len(dropped_training)}")
    if unexpected:
        print(f"Ignored keys not in target backbone: {len(unexpected)}")
        print("  first:", unexpected[:10])
    if missing:
        print(f"Missing target keys after export: {len(missing)}")
        print("  first:", missing[:10])


if __name__ == "__main__":
    main()
