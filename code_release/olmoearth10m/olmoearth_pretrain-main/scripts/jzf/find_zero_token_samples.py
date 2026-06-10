"""找出 num_encoded_tokens == 0 的 sample。

这个脚本模拟训练时的 masking 流程，逐 sample 检查哪些 sample 在 masking 后
没有任何 ONLINE_ENCODER token，导致 pool_instance_wise 报错。

用法:
    # 检查整个 h5 目录
    python find_zero_encoded_samples.py \
        --h5py_dir /mnt/ht2-nas2/00-model/00-jiangzf/coderepo/H5_DIR/.../3996 \
        --training_modalities sentinel2_l2a sentinel1 landsat worldcover srtm openstreetmap_raster wri_canopy_height_map cdl worldcereal planet_rgbnir \
        --masking_type modality_cross_random \
        --encode_ratio 0.5 --decode_ratio 0.5 \
        --only_decode_modalities worldcover srtm openstreetmap_raster wri_canopy_height_map cdl worldcereal \
        --patch_size 1 --sampled_hw_p 6 --token_budget 2250

    # 快速检查前 100 个 sample
    python find_zero_encoded_samples.py \
        --h5py_dir ... \
        --training_modalities ... \
        --masking_type modality_cross_random \
        --max_samples 100

    # 只检查 h5 数据质量（不做 masking），找出全 MISSING 的 sample
    python find_zero_encoded_samples.py \
        --h5py_dir ... \
        --training_modalities ... \
        --check_data_only

    python find_zero_token_samples.py --h5py_dir /mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/2205 --training_modalities landcover_1m landcover_30m landsat lt1 rgb sar sentinel1 sentinel2_l2a srtm worldcereal worldcover --check_data_only --max_samples 100

    python find_zero_token_samples.py --h5py_dir /mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/2205 --training_modalities landcover_1m landcover_30m landsat lt1 rgb sar sentinel1 sentinel2_l2a srtm worldcereal worldcover --masking_type modality_cross_random  --encode_ratio 0.5 --decode_ratio 0.5  --only_decode_modalities worldcover worldcereal srtm landcover_1m landcover_30m --allow_encoding_decoding_same_bandset --patch_size 1 --sampled_hw_p 6 --token_budget 2250 --num_trials 5 --max_samples 100

"""

import argparse
import logging
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent / "../.." 
print(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthDatasetConfig,
)
from olmoearth_pretrain.datatypes import MaskValue, MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskingConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_data_quality(
    h5py_dir: str,
    training_modalities: list[str],
    max_samples: int = 0,
) -> list[dict]:
    """只检查 h5 数据质量，不做 masking。

    找出以下问题:
    - 所有 spacetime-varying modality 全为 MISSING_VALUE 的 sample
    - 某个 modality 全为 MISSING_VALUE 的 sample
    - missing_timesteps_masks 中所有 timestep 都缺失的 sample
    """
    problems = []
    h5_dir = Path(h5py_dir)

    # 获取 sample 数量
    num_samples = int(h5_dir.name)

    # 获取 metadata
    import pandas as pd
    metadata_path = h5_dir / "sample_metadata.csv"
    if metadata_path.exists():
        metadata_df = pd.read_csv(str(metadata_path))
    else:
        metadata_df = None

    # spacetime_varying modalities
    spacetime_modalities = [
        m for m in training_modalities
        if Modality.get(m).is_spacetime_varying and not Modality.get(m).ignore_when_parsing
    ]

    indices_to_check = range(num_samples)
    if max_samples > 0:
        indices_to_check = list(indices_to_check)[:max_samples]

    logger.info(f"检查 {len(list(indices_to_check))} 个 sample 的数据质量...")

    for idx in range(num_samples):
        if max_samples > 0 and idx >= max_samples:
            break

        h5_path = h5_dir / f"sample_{idx}.h5"
        if not h5_path.exists():
            problems.append({
                "index": idx,
                "problem": "h5 file not found",
                "details": str(h5_path),
            })
            continue

        try:
            with h5py.File(str(h5_path), "r") as f:
                sample_dict = {k: v[()] for k, v in f.items() if k in training_modalities or k == "timestamps"}

                # 检查 missing_timesteps_masks
                mask_group_name = "missing_timesteps_masks"
                if mask_group_name in f:
                    missing_masks = {k: v[()] for k, v in f[mask_group_name].items() if k in training_modalities}
                else:
                    missing_masks = {}

                # 检查每个 modality
                for modality in training_modalities:
                    if modality not in sample_dict:
                        # modality 完全不存在
                        continue

                    data = sample_dict[modality]
                    if data is None:
                        continue

                    # 检查是否全为 MISSING_VALUE
                    all_missing = np.all(data == MISSING_VALUE)
                    if all_missing:
                        problems.append({
                            "index": idx,
                            "problem": f"{modality} all MISSING_VALUE",
                            "details": f"shape={data.shape}",
                        })

                    # 检查 missing_timesteps_masks
                    if modality in missing_masks:
                        mask = missing_masks[modality]
                        if mask.sum() == 0:
                            problems.append({
                                "index": idx,
                                "problem": f"{modality} all timesteps missing in mask",
                                "details": f"mask={mask.tolist()}",
                            })

                # 检查是否有任何 spacetime-varying modality 存在
                has_any_spacetime = False
                for modality in spacetime_modalities:
                    if modality in sample_dict:
                        data = sample_dict[modality]
                        if data is not None and not np.all(data == MISSING_VALUE):
                            has_any_spacetime = True
                            break

                if not has_any_spacetime:
                    problems.append({
                        "index": idx,
                        "problem": "no spacetime-varying modality has valid data",
                        "details": f"checked: {spacetime_modalities}",
                    })

        except Exception as e:
            problems.append({
                "index": idx,
                "problem": f"read error: {e}",
                "details": "",
            })

        if (idx + 1) % 500 == 0:
            logger.info(f"  已检查 {idx + 1}/{num_samples} 个 sample...")

    return problems


def check_masking(
    h5py_dir: str,
    training_modalities: list[str],
    masking_config: MaskingConfig,
    patch_size: int = 1,
    sampled_hw_p: int = 6,
    token_budget: int = 2250,
    max_samples: int = 0,
    num_trials: int = 5,
) -> list[dict]:
    """模拟 masking 流程，找出 masking 后 num_encoded_tokens == 0 的 sample。

    对每个 sample 进行多次 masking 试验（因为 masking 是随机的），
    如果任何一次试验出现 0 encoded tokens，就报告该 sample。
    """
    problems = []

    # 构建 dataset
    dataset_config = OlmoEarthDatasetConfig(
        h5py_dir=h5py_dir,
        training_modalities=training_modalities,
        dtype="float32",
        normalize=True,
    )
    dataset = dataset_config.build()
    dataset.prepare()

    # 构建 masking strategy
    masking_strategy = masking_config.build()

    num_samples = len(dataset)
    if max_samples > 0:
        num_samples = min(num_samples, max_samples)

    logger.info(f"检查 {num_samples} 个 sample 的 masking 结果 (每个 sample {num_trials} 次试验)...")
    logger.info(f"Masking strategy: {masking_config.strategy_config}")

    for i in range(num_samples):
        # 获取 sample
        args = GetItemArgs(
            idx=i,
            patch_size=patch_size,
            sampled_hw_p=sampled_hw_p,
            token_budget=token_budget,
        )

        try:
            _, sample = dataset[args]
        except Exception as e:
            problems.append({
                "index": i,
                "problem": f"dataset __getitem__ error: {e}",
                "details": "",
            })
            continue

        # 对这个 sample 进行多次 masking 试验
        for trial in range(num_trials):
            try:
                # 和 collate_olmoearth_pretrain 一样：
                # 用 torch.from_numpy + stack 添加 batch 维度
                batch_dict = {}
                for field in sample.modalities_with_timestamps:
                    val = getattr(sample, field)
                    if val is None:
                        continue
                    if isinstance(val, np.ndarray):
                        batch_dict[field] = torch.from_numpy(val).unsqueeze(0)
                    elif isinstance(val, torch.Tensor):
                        batch_dict[field] = val.unsqueeze(0)
                    else:
                        batch_dict[field] = val

                batch_sample = OlmoEarthSample(**batch_dict)

                # 应用 masking
                masked_sample = masking_strategy.apply_mask(batch_sample, patch_size)

                # 检查 num_encoded_tokens
                # 遍历所有 modality 的 mask，统计 ONLINE_ENCODER token 数
                total_encoded = 0
                modality_details = {}
                for mod_name in masked_sample.modalities:
                    mask_attr_name = MaskedOlmoEarthSample.get_masked_modality_name(mod_name)
                    mask = getattr(masked_sample, mask_attr_name)
                    if mask is not None:
                        encoded_count = (mask == MaskValue.ONLINE_ENCODER.value).sum().item()
                        missing_count = (mask == MaskValue.MISSING.value).sum().item()
                        total_count = mask.numel()
                        total_encoded += encoded_count
                        modality_details[mod_name] = {
                            "encoded": encoded_count,
                            "missing": missing_count,
                            "total": total_count,
                        }

                if total_encoded == 0:
                    problems.append({
                        "index": i,
                        "problem": "num_encoded_tokens == 0 after masking",
                        "trial": trial,
                        "details": {
                            "modality_breakdown": modality_details,
                            "sample_modalities": sample.modalities,
                        },
                    })

            except Exception as e:
                problems.append({
                    "index": i,
                    "problem": f"masking error: {e}",
                    "trial": trial,
                    "details": "",
                })

        if (i + 1) % 100 == 0:
            logger.info(f"  已检查 {i + 1}/{num_samples} 个 sample...")

    return problems


def main():
    parser = argparse.ArgumentParser(description="找出 num_encoded_tokens == 0 的 sample")
    parser.add_argument("--h5py_dir", type=str, required=True, help="h5 文件目录")
    parser.add_argument("--training_modalities", nargs="+", required=True, help="训练模态列表")
    parser.add_argument("--max_samples", type=int, default=0, help="最多检查的 sample 数 (0=全部)")
    parser.add_argument("--check_data_only", action="store_true", help="只检查数据质量，不做 masking")

    # Masking 参数
    parser.add_argument("--masking_type", type=str, default="modality_cross_random", help="masking 策略类型")
    parser.add_argument("--encode_ratio", type=float, default=0.5, help="encode ratio")
    parser.add_argument("--decode_ratio", type=float, default=0.5, help="decode ratio")
    parser.add_argument("--only_decode_modalities", nargs="+", default=[], help="只 decode 的模态")
    parser.add_argument("--allow_encoding_decoding_same_bandset", action="store_true", help="允许同一 bandset 同时 encode 和 decode")
    parser.add_argument("--patch_size", type=int, default=1, help="patch size")
    parser.add_argument("--sampled_hw_p", type=int, default=6, help="sampled hw in patches")
    parser.add_argument("--token_budget", type=int, default=2250, help="token budget")
    parser.add_argument("--num_trials", type=int, default=5, help="每个 sample 的 masking 试验次数")

    args = parser.parse_args()

    print("=" * 80)
    print("查找 num_encoded_tokens == 0 的 sample")
    print("=" * 80)
    print(f"h5py_dir: {args.h5py_dir}")
    print(f"training_modalities: {args.training_modalities}")
    print(f"max_samples: {args.max_samples or 'all'}")
    print()

    start_time = time.time()

    if args.check_data_only:
        problems = check_data_quality(
            h5py_dir=args.h5py_dir,
            training_modalities=args.training_modalities,
            max_samples=args.max_samples,
        )
    else:
        masking_config = MaskingConfig(
            strategy_config={
                "type": args.masking_type,
                "encode_ratio": args.encode_ratio,
                "decode_ratio": args.decode_ratio,
                "allow_encoding_decoding_same_bandset": args.allow_encoding_decoding_same_bandset,
                "only_decode_modalities": args.only_decode_modalities,
            },
        )
        problems = check_masking(
            h5py_dir=args.h5py_dir,
            training_modalities=args.training_modalities,
            masking_config=masking_config,
            patch_size=args.patch_size,
            sampled_hw_p=args.sampled_hw_p,
            token_budget=args.token_budget,
            max_samples=args.max_samples,
            num_trials=args.num_trials,
        )

    elapsed = time.time() - start_time

    # 输出结果
    print()
    print("=" * 80)
    print(f"检查完成，耗时 {elapsed:.1f} 秒")
    print(f"发现 {len(problems)} 个问题")
    print("=" * 80)

    if problems:
        # 按问题类型分组
        from collections import Counter
        problem_types = Counter(p["problem"] for p in problems)
        print("\n问题类型统计:")
        for ptype, count in problem_types.most_common():
            print(f"  {ptype}: {count} 次")

        # 输出有问题的 sample 索引
        problem_indices = sorted(set(p["index"] for p in problems))
        print(f"\n有问题的 sample 索引 ({len(problem_indices)} 个):")
        for idx in problem_indices:
            idx_problems = [p for p in problems if p["index"] == idx]
            print(f"\n  sample_{idx}:")
            for p in idx_problems:
                detail_str = ""
                if isinstance(p.get("details"), dict):
                    detail_str = str(p["details"])
                elif p.get("details"):
                    detail_str = str(p["details"])
                trial_str = f" (trial {p['trial']})" if "trial" in p else ""
                print(f"    - {p['problem']}{trial_str}: {detail_str}")
    else:
        print("\n未发现问题！所有 sample 的 num_encoded_tokens > 0。")


if __name__ == "__main__":
    main()
