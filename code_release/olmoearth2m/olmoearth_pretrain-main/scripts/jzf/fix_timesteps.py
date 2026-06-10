"""批量修复脚本：确保所有样本都有统一的 12 个时间步。

这个脚本会：
1. 读取现有的 H5 文件
2. 将所有模态填充到 12 个时间步（缺失的用 MISSING_VALUE 填充）
3. 更新 missing_timesteps_masks
4. 保存到新文件
"""

import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 配置
INPUT_DIR = Path("/mnt/ht2-nas2/00-model/00-jiangzf/coderepo/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map_planet_rgbnir/3996")
OUTPUT_DIR = INPUT_DIR / "fixed_timesteps"
TARGET_TIMESTEPS = 12
MISSING_VALUE = -9999  # 根据实际数据调整

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("批量修复时间步数")
print("=" * 80)
print(f"输入目录: {INPUT_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"目标时间步数: {TARGET_TIMESTEPS}\n")

# 获取所有 H5 文件
h5_files = sorted(INPUT_DIR.glob("sample_*.h5"))
print(f"找到 {len(h5_files)} 个文件\n")

# 需要填充的模态列表（多时相模态）
multitemporal_modalities = [
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    "planet_rgbnir",
    # 添加其他多时相模态...
]

fixed_count = 0
skipped_count = 0

for h5_file in tqdm(h5_files, desc="Processing"):
    output_file = OUTPUT_DIR / h5_file.name
    
    try:
        with h5py.File(h5_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
            needs_fix = False
            
            # 检查是否需要修复
            for modality in multitemporal_modalities:
                if modality in f_in:
                    shape = f_in[modality].shape
                    if len(shape) >= 3 and shape[2] != TARGET_TIMESTEPS:
                        needs_fix = True
                        break
            
            if not needs_fix:
                # 不需要修复，直接复制
                skipped_count += 1
                continue
            
            # 需要修复，逐个处理
            for key in f_in.keys():
                if key == "missing_timesteps_masks":
                    # 特殊处理 masks group
                    mask_group_in = f_in[key]
                    mask_group_out = f_out.create_group(key)
                    
                    for subkey in mask_group_in.keys():
                        mask_data = mask_group_in[subkey][:]
                        if len(mask_data) < TARGET_TIMESTEPS:
                            # 扩展 mask
                            new_mask = np.ones(TARGET_TIMESTEPS, dtype=bool)
                            new_mask[:len(mask_data)] = mask_data
                            mask_group_out.create_dataset(
                                subkey, 
                                data=new_mask,
                                compression=hdf5plugin.Zstd(clevel=3)
                            )
                        else:
                            mask_group_out.create_dataset(
                                subkey,
                                data=mask_data,
                                compression=hdf5plugin.Zstd(clevel=3)
                            )
                elif key in multitemporal_modalities:
                    # 处理多时相模态
                    data = f_in[key][:]
                    if len(data.shape) >= 3 and data.shape[2] < TARGET_TIMESTEPS:
                        # 填充到目标时间步数
                        new_shape = list(data.shape)
                        new_shape[2] = TARGET_TIMESTEPS
                        new_data = np.full(new_shape, MISSING_VALUE, dtype=data.dtype)
                        new_data[:, :, :data.shape[2], :] = data
                        f_out.create_dataset(
                            key,
                            data=new_data,
                            compression=hdf5plugin.Zstd(clevel=3)
                        )
                    else:
                        f_out.create_dataset(
                            key,
                            data=data,
                            compression=hdf5plugin.Zstd(clevel=3)
                        )
                else:
                    # 其他数据直接复制
                    data = f_in[key][:]
                    if isinstance(data, np.ndarray):
                        f_out.create_dataset(
                            key,
                            data=data,
                            compression=hdf5plugin.Zstd(clevel=3)
                        )
                    else:
                        f_out.create_dataset(key, data=data)
            
            fixed_count += 1
    
    except Exception as e:
        print(f"\n❌ Error processing {h5_file.name}: {e}")
        continue

print("\n" + "=" * 80)
print("修复完成！")
print("=" * 80)
print(f"修复的文件数: {fixed_count}")
print(f"跳过的文件数（已符合要求的）: {skipped_count}")
print(f"输出目录: {OUTPUT_DIR}")
print("\n下一步:")
print(f"1. 备份原目录: mv {INPUT_DIR} {INPUT_DIR}_backup")
print(f"2. 移动修复后的文件: mv {OUTPUT_DIR}/* {INPUT_DIR}/")
print(f"3. 重新运行诊断脚本验证")
