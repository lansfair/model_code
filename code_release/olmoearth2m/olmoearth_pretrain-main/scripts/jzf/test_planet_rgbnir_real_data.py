"""Planet RGBNIR 超分辨率测试 - 使用真实数据。

这个脚本用于在 /mnt/ht2-nas2/00-model/00-jiangzf/coderepo/3996 路径下
的测试数据上运行超分辨率功能。

使用方法 (在 Jupyter Notebook 中):
    %run test_planet_rgbnir_real_data.py
    
或者直接在命令行运行:
    python test_planet_rgbnir_real_data.py
"""

import logging
from pathlib import Path
import h5py
import numpy as np
import cv2
import hdf5plugin  # noqa: F401

# Force reload to ensure latest Modality definitions are loaded
import importlib
import olmoearth_pretrain.data.constants
importlib.reload(olmoearth_pretrain.data.constants)

from olmoearth_pretrain.data.constants import Modality, MISSING_VALUE

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 测试数据路径
TEST_DATA_DIR = Path("/mnt/ht2-nas2/00-model/00-jiangzf/coderepo/3996")


def inspect_h5_file(h5_path: Path):
    """检查H5文件的结构。"""
    print(f"\n{'='*80}")
    print(f"检查文件: {h5_path.name}")
    print(f"{'='*80}")
    
    with h5py.File(h5_path, "r") as f:
        print(f"\n数据集列表:")
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(f"  - {key}: shape={obj.shape}, dtype={obj.dtype}, compression={obj.compression}")
            elif isinstance(obj, h5py.Group):
                print(f"  - {key}: [Group] with {len(obj)} items")
        
        # 检查sentinel2_l2a
        if "sentinel2_l2a" in f:
            s2 = f["sentinel2_l2a"]
            print(f"\nSentinel-2 L2A 详细信息:")
            print(f"  Shape: {s2.shape}")
            print(f"  Dtype: {s2.dtype}")
            print(f"  Compression: {s2.compression}")
            
            # 读取一小部分数据查看
            sample = s2[0:10, 0:10, 0, :]
            print(f"  Sample data range: [{sample.min()}, {sample.max()}]")
            print(f"  Has missing values: {np.any(sample == MISSING_VALUE)}")


def super_resolution_test(h5_path: Path, output_dir: Path = None, dry_run: bool = True):
    """对单个H5文件进行超分辨率测试,结果写入新文件。
    
    Args:
        h5_path: 输入H5文件路径
        output_dir: 输出目录,默认为输入文件的同级目录下的"planet_rgbnir_output"文件夹
        dry_run: 如果为True,只执行超分辨率不写入文件
    """
    print(f"\n{'='*80}")
    print(f"超分辨率测试: {h5_path.name}")
    print(f"{'='*80}")
    
    mode = "DRY RUN" if dry_run else "WRITE"
    print(f"模式: {mode}")
    
    # 确定输出文件路径
    if output_dir is None:
        output_dir = h5_path.parent / "planet_rgbnir_output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / h5_path.name
    
    print(f"输出文件: {output_path}")
    
    with h5py.File(h5_path, "r") as f_in:
        # 检查源数据
        if "sentinel2_l2a" not in f_in:
            print("❌ sentinel2_l2a 不存在")
            return False
        
        s2_data = f_in["sentinel2_l2a"][()]
        print(f"\n✅ 加载 sentinel2_l2a: shape={s2_data.shape}, dtype={s2_data.dtype}")
        
        # 获取RGB波段索引
        s2_modality = Modality.get("sentinel2_l2a")
        band_order = s2_modality.band_order
        r_idx = band_order.index("B04")
        g_idx = band_order.index("B03")
        b_idx = band_order.index("B02")
        print(f"✅ RGB波段索引: R=B04({r_idx}), G=B03({g_idx}), B=B02({b_idx})")
        
        # 提取RGB
        rgb_10m = s2_data[:, :, :, [r_idx, g_idx, b_idx]]
        print(f"✅ 提取RGB: shape={rgb_10m.shape}")
        
        # 计算缩放因子
        planet_modality = Modality.get("planet_rgbnir")
        source_tile_size = s2_modality.get_expected_tile_size()
        target_tile_size = planet_modality.get_expected_tile_size()
        scale_factor = target_tile_size / source_tile_size
        print(f"✅ 缩放因子: {scale_factor:.2f} ({source_tile_size} -> {target_tile_size})")
        
        # 执行超分辨率
        h, w, t, c = rgb_10m.shape
        
        # 确保所有样本都有 12 个时间步
        TARGET_TIMESTEPS = 12
        if t < TARGET_TIMESTEPS:
            print(f"⚠️  时间步数不足 ({t} < {TARGET_TIMESTEPS})，将填充到 {TARGET_TIMESTEPS} 步")
            # 创建填充数组
            rgb_padded = np.full((h, w, TARGET_TIMESTEPS, c), MISSING_VALUE, dtype=rgb_10m.dtype)
            # 复制现有数据
            rgb_padded[:, :, :t, :] = rgb_10m
            rgb_10m = rgb_padded
            t = TARGET_TIMESTEPS
        
        target_h = int(h * scale_factor)
        target_w = int(w * scale_factor)
        print(f"\n🔄 执行超分辨率: ({h}, {w}, {t}) -> ({target_h}, {target_w}, {t})")
        
        rgb_hr = np.empty((target_h, target_w, t, c), dtype=rgb_10m.dtype)
        
        for i in range(t):
            if np.any(rgb_10m[:, :, i, :] == MISSING_VALUE):
                logger.warning(f"Timestep {i} contains missing values")
                rgb_hr[:, :, i, :] = MISSING_VALUE
                continue
            
            rgb_timestep = rgb_10m[:, :, i, :].astype(np.float32)
            rgb_upsampled = cv2.resize(
                rgb_timestep,
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC
            )
            rgb_hr[:, :, i, :] = rgb_upsampled.astype(rgb_10m.dtype)
        
        print(f"✅ 超分辨率完成: shape={rgb_hr.shape}, dtype={rgb_hr.dtype}")
        print(f"   数据范围: [{rgb_hr.min()}, {rgb_hr.max()}]")
        
        if dry_run:
            print("\n⏭️  Dry run模式,跳过写入")
            return True
        
        # 复制到新文件并添加planet_rgbnir
        print(f"\n💾 创建新文件: {output_path}")
        print("📋 复制原始数据...")
        
        with h5py.File(output_path, "w") as f_out:
            # 复制所有原始数据集
            for key in f_in.keys():
                if key == "missing_timesteps_masks":
                    # 特殊处理group
                    group = f_in[key]
                    out_group = f_out.create_group(key)
                    for subkey in group.keys():
                        out_group.create_dataset(subkey, data=group[subkey][()], 
                                               compression=hdf5plugin.Zstd(clevel=3))
                else:
                    # 复制dataset
                    data = f_in[key][()]
                    if isinstance(data, np.ndarray):
                        f_out.create_dataset(key, data=data, compression=hdf5plugin.Zstd(clevel=3))
                    else:
                        f_out.create_dataset(key, data=data)
            
            print("✅ 原始数据复制完成")
            
            # 添加planet_rgbnir
            print("💾 写入 planet_rgbnir...")
            f_out.create_dataset(
                "planet_rgbnir",
                data=rgb_hr,
                compression=hdf5plugin.Zstd(clevel=3)
            )
            
            # 验证
            written_data = f_out["planet_rgbnir"][()]
            if written_data.shape == rgb_hr.shape:
                print("✅ 验证成功!")
                print(f"📊 输出文件大小: {output_path.stat().st_size / (1024*1024):.2f} MB")
                return True
            else:
                print(f"❌ 验证失败: expected {rgb_hr.shape}, got {written_data.shape}")
                return False


def main():
    """主函数。"""
    print("=" * 80)
    print("Planet RGBNIR 超分辨率 - 真实数据测试")
    print("=" * 80)
    print(f"\n测试数据目录: {TEST_DATA_DIR}")
    
    if not TEST_DATA_DIR.exists():
        print(f"❌ 目录不存在: {TEST_DATA_DIR}")
        return
    
    # 查找H5文件
    h5_files = sorted(TEST_DATA_DIR.glob("*.h5"))
    print(f"找到 {len(h5_files)} 个H5文件\n")
    
    if len(h5_files) == 0:
        print("❌ 没有找到H5文件")
        return
    
    # 测试第一个文件
    test_file = h5_files[0]
    print(f"选择测试文件: {test_file.name}\n")
    
    # Step 1: 检查文件结构
    inspect_h5_file(test_file)
    
    # Step 2: 运行超分辨率测试 (dry run)
    success = super_resolution_test(test_file, dry_run=True)
    
    if success:
        print("\n" + "=" * 80)
        print("✅ 测试成功!")
        print("💡 要实际写入新文件,请使用:")
        print(f"   super_resolution_test(test_file, dry_run=False)")
        print(f"   输出将保存到: {test_file.parent / 'planet_rgbnir_output' / test_file.name}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ 测试失败")
        print("=" * 80)


if __name__ == "__main__":
    main()
