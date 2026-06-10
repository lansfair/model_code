"""从H5文件读取10m RGB图像并进行超分辨率,生成planet_rgbnir模态。

这个脚本从现有的H5文件中读取10m分辨率的RGB图像(通常来自sentinel2_l2a),
使用超分辨率算法将其提升到3m分辨率,然后作为planet_rgbnir模态保存回H5文件。

Usage:
    python generate_planet_rgbnir.py --h5-file /path/to/sample.h5
    python generate_planet_rgbnir.py --h5-dir /path/to/h5py_dir/
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import h5py
import hdf5plugin  # noqa: F401 - 注册HDF5 filter插件
import numpy as np
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, MISSING_VALUE

logger = logging.getLogger(__name__)


def super_resolution_bicubic(
    rgb_10m: np.ndarray, 
    scale_factor: float = 10.0 / 3.0
) -> np.ndarray:
    """使用双三次插值对RGB图像进行超分辨率。
    
    Args:
        rgb_10m: 10m分辨率的RGB图像,形状为 (H, W, T, 3)
        scale_factor: 上采样因子,默认10/3 ≈ 3.33
        
    Returns:
        超分辨率后的RGB图像,形状为 (H*scale_factor, W*scale_factor, T, 3)
    """
    if rgb_10m.ndim != 4:
        raise ValueError(f"Expected 4D array (H, W, T, C), got shape {rgb_10m.shape}")
    
    h, w, t, c = rgb_10m.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {c} channels")
    
    # 计算目标尺寸
    target_h = int(h * scale_factor)
    target_w = int(w * scale_factor)
    
    logger.info(f"Upsampling from ({h}, {w}) to ({target_h}, {target_w}) with scale factor {scale_factor:.2f}")
    
    # 创建输出数组
    rgb_hr = np.empty((target_h, target_w, t, 3), dtype=rgb_10m.dtype)
    
    # 对每个时间步进行超分辨率
    for i in range(t):
        # 检查是否有缺失值
        if np.any(rgb_10m[:, :, i, :] == MISSING_VALUE):
            logger.warning(f"Timestep {i} contains missing values")
            rgb_hr[:, :, i, :] = MISSING_VALUE
            continue
        
        # 使用OpenCV的双三次插值进行上采样
        rgb_timestep = rgb_10m[:, :, i, :].astype(np.float32)
        rgb_upsampled = cv2.resize(
            rgb_timestep,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC
        )
        rgb_hr[:, :, i, :] = rgb_upsampled.astype(rgb_10m.dtype)
    
    return rgb_hr


def process_single_h5_file(
    h5_file_path: Path,
    source_modality: str = "sentinel2_l2a",
    target_modality: str = "planet_rgbnir",
    compression: str = "zstd",
    compression_opts: int = 3,
    dry_run: bool = False,
) -> bool:
    """处理单个H5文件,生成planet_rgbnir模态。
    
    Args:
        h5_file_path: H5文件路径
        source_modality: 源模态名称,默认为sentinel2_l2a
        target_modality: 目标模态名称,默认为planet_rgbnir
        compression: 压缩算法,默认为zstd
        compression_opts: 压缩级别,默认为3
        dry_run: 如果为True,只读取不写入
        
    Returns:
        是否成功处理
    """
    try:
        logger.info(f"Processing file: {h5_file_path}")
        
        # 打开H5文件
        with h5py.File(h5_file_path, "a") as f:
            # 检查源模态是否存在
            if source_modality not in f:
                logger.warning(f"Source modality '{source_modality}' not found in {h5_file_path}")
                return False
            
            # 读取源数据
            s2_data = f[source_modality][()]
            logger.info(f"Loaded {source_modality} with shape {s2_data.shape}")
            
            # 验证数据形状: (H, W, T, C)
            if s2_data.ndim != 4:
                logger.error(f"Expected 4D data (H, W, T, C), got shape {s2_data.shape}")
                return False
            
            h, w, t, c = s2_data.shape
            
            # 获取Sentinel-2的RGB波段索引
            # Sentinel-2 L2A band order: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
            # RGB: R=B04(index 2), G=B03(index 1), B=B02(index 0)
            s2_modality = Modality.get(source_modality)
            band_order = s2_modality.band_order
            
            try:
                r_idx = band_order.index("B04")
                g_idx = band_order.index("B03")
                b_idx = band_order.index("B02")
                rgb_indices = [r_idx, g_idx, b_idx]
                logger.info(f"Using RGB indices: R={r_idx}, G={g_idx}, B={b_idx}")
            except ValueError as e:
                logger.error(f"Could not find RGB bands in {source_modality}: {e}")
                return False
            
            # 提取RGB通道
            rgb_10m = s2_data[:, :, :, rgb_indices]  # (H, W, T, 3)
            logger.info(f"Extracted RGB with shape {rgb_10m.shape}")
            
            # 检查是否有缺失值
            missing_mask = rgb_10m == MISSING_VALUE
            if np.any(missing_mask):
                missing_ratio = np.sum(missing_mask) / rgb_10m.size
                logger.warning(f"Missing value ratio: {missing_ratio:.2%}")
            
            # 进行超分辨率
            # planet_rgbnir的tile_resolution_factor=16,与sentinel2_l2a相同
            # 但image_tile_size_factor不同,需要调整
            planet_modality = Modality.get(target_modality)
            source_tile_size = s2_modality.get_expected_tile_size()
            target_tile_size = planet_modality.get_expected_tile_size()
            
            # 计算缩放因子
            scale_factor = target_tile_size / source_tile_size
            logger.info(f"Scale factor: {scale_factor:.2f} (source tile size: {source_tile_size}, target tile size: {target_tile_size})")
            
            rgb_hr = super_resolution_bicubic(rgb_10m, scale_factor=scale_factor)
            logger.info(f"Super-resolution result shape: {rgb_hr.shape}")
            
            if dry_run:
                logger.info("Dry run mode, skipping write")
                return True
            
            # 删除已存在的数据集(如果有)
            if target_modality in f:
                logger.info(f"Deleting existing {target_modality} dataset")
                del f[target_modality]
            
            # 创建新的数据集
            create_kwargs = {
                "compression": compression,
            }
            
            if compression == "gzip":
                create_kwargs["compression_opts"] = compression_opts
                create_kwargs["shuffle"] = True
            elif compression == "zstd":
                create_kwargs["compression"] = hdf5plugin.Zstd(clevel=compression_opts)
            else:
                raise ValueError(f"Unsupported compression: {compression}")
            
            f.create_dataset(
                target_modality,
                data=rgb_hr,
                **create_kwargs
            )
            
            logger.info(f"Successfully wrote {target_modality} to {h5_file_path}")
            logger.info(f"Dataset shape: {rgb_hr.shape}, dtype: {rgb_hr.dtype}")
            
            # 验证写入
            written_data = f[target_modality][()]
            if written_data.shape != rgb_hr.shape:
                logger.error(f"Verification failed: expected shape {rgb_hr.shape}, got {written_data.shape}")
                return False
            
            logger.info("Verification passed")
            return True
            
    except Exception as e:
        logger.error(f"Error processing {h5_file_path}: {e}", exc_info=True)
        return False


def process_h5_directory(
    h5_dir: Path,
    source_modality: str = "sentinel2_l2a",
    target_modality: str = "planet_rgbnir",
    compression: str = "zstd",
    compression_opts: int = 3,
    max_files: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    """处理目录中的所有H5文件。
    
    Args:
        h5_dir: H5文件目录
        source_modality: 源模态名称
        target_modality: 目标模态名称
        compression: 压缩算法
        compression_opts: 压缩级别
        max_files: 最多处理的文件数量,None表示处理所有文件
        dry_run: 如果为True,只读取不写入
    """
    h5_dir = Path(h5_dir)
    if not h5_dir.exists():
        raise FileNotFoundError(f"Directory not found: {h5_dir}")
    
    # 查找所有H5文件
    h5_files = sorted(h5_dir.glob("*.h5"))
    logger.info(f"Found {len(h5_files)} H5 files in {h5_dir}")
    
    if max_files is not None:
        h5_files = h5_files[:max_files]
        logger.info(f"Processing first {max_files} files")
    
    success_count = 0
    fail_count = 0
    
    for i, h5_file in enumerate(h5_files):
        logger.info(f"[{i+1}/{len(h5_files)}] Processing {h5_file.name}")
        if process_single_h5_file(
            h5_file,
            source_modality=source_modality,
            target_modality=target_modality,
            compression=compression,
            compression_opts=compression_opts,
            dry_run=dry_run,
        ):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Processing complete: {success_count} succeeded, {fail_count} failed")


def main():
    """主函数,解析命令行参数并执行处理。"""
    parser = argparse.ArgumentParser(
        description="Generate planet_rgbnir modality from 10m RGB images using super-resolution"
    )
    
    parser.add_argument(
        "--h5-file",
        type=Path,
        help="Path to a single H5 file to process"
    )
    
    parser.add_argument(
        "--h5-dir",
        type=Path,
        help="Path to directory containing H5 files to process"
    )
    
    parser.add_argument(
        "--source-modality",
        type=str,
        default="sentinel2_l2a",
        help="Source modality name (default: sentinel2_l2a)"
    )
    
    parser.add_argument(
        "--target-modality",
        type=str,
        default="planet_rgbnir",
        help="Target modality name (default: planet_rgbnir)"
    )
    
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["gzip", "zstd"],
        help="Compression algorithm (default: zstd)"
    )
    
    parser.add_argument(
        "--compression-opts",
        type=int,
        default=3,
        help="Compression level (default: 3)"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: read only, don't write"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 验证参数
    if args.h5_file is None and args.h5_dir is None:
        parser.error("Either --h5-file or --h5-dir must be specified")
    
    if args.h5_file is not None and args.h5_dir is not None:
        parser.error("Cannot specify both --h5-file and --h5-dir")
    
    # 执行处理
    if args.h5_file is not None:
        if not args.h5_file.exists():
            raise FileNotFoundError(f"File not found: {args.h5_file}")
        
        success = process_single_h5_file(
            args.h5_file,
            source_modality=args.source_modality,
            target_modality=args.target_modality,
            compression=args.compression,
            compression_opts=args.compression_opts,
            dry_run=args.dry_run,
        )
        
        if success:
            logger.info("Successfully processed file")
        else:
            logger.error("Failed to process file")
            exit(1)
    
    else:  # args.h5_dir is not None
        process_h5_directory(
            args.h5_dir,
            source_modality=args.source_modality,
            target_modality=args.target_modality,
            compression=args.compression,
            compression_opts=args.compression_opts,
            max_files=args.max_files,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
