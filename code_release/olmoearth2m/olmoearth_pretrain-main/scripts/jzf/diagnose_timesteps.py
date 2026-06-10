"""诊断脚本：检查数据集中所有样本的时间步数一致性。"""

import h5py
import hdf5plugin
from pathlib import Path
from collections import Counter

data_dir = Path("/mnt/ht2-nas2/00-model/00-jiangzf/coderepo/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map_planet_rgbnir/3996")

print("=" * 80)
print("数据集时间步数一致性诊断")
print("=" * 80)
print(f"\n数据目录: {data_dir}")
print(f"目录存在: {data_dir.exists()}\n")

if not data_dir.exists():
    print("❌ 目录不存在！")
    exit(1)

# 统计时间步数分布
s2_timesteps = Counter()
planet_timesteps = Counter()
inconsistent_samples = []

num_samples = 3996  # 根据目录名推断
check_count = min(100, num_samples)  # 检查前100个样本

print(f"检查前 {check_count} 个样本的时间步数...\n")

for i in range(check_count):
    h5_file = data_dir / f"sample_{i}.h5"
    if not h5_file.exists():
        continue
    
    try:
        with h5py.File(h5_file, 'r') as f:
            s2_shape = None
            planet_shape = None
            
            if 'sentinel2_l2a' in f:
                s2_shape = f['sentinel2_l2a'].shape
                s2_timesteps[s2_shape[2]] += 1
            
            if 'planet_rgbnir' in f:
                planet_shape = f['planet_rgbnir'].shape
                planet_timesteps[planet_shape[2]] += 1
            
            # 检查是否一致
            if s2_shape and planet_shape:
                if s2_shape[2] != planet_shape[2]:
                    inconsistent_samples.append((i, s2_shape[2], planet_shape[2]))
                
                # 打印前10个样本的详细信息
                if i < 10:
                    print(f"sample_{i:4d}: S2={s2_shape}, Planet={planet_shape}")
    
    except Exception as e:
        print(f"Error processing sample_{i}: {e}")

print("\n" + "=" * 80)
print("统计结果:")
print("=" * 80)
print(f"\nSentinel-2 L2A 时间步数分布: {dict(s2_timesteps)}")
print(f"Planet RGBNIR 时间步数分布: {dict(planet_timesteps)}")

if inconsistent_samples:
    print(f"\n⚠️  发现 {len(inconsistent_samples)} 个不一致的样本:")
    for idx, s2_t, p_t in inconsistent_samples[:10]:
        print(f"  sample_{idx}: S2={s2_t}, Planet={p_t}")
else:
    print("\n✅ 所有样本的时间步数一致！")

print("\n" + "=" * 80)
print("建议:")
print("=" * 80)
if len(s2_timesteps) > 1 or len(planet_timesteps) > 1:
    print("❌ 数据存在时间步数不一致问题！")
    print("解决方案:")
    print("1. 重新生成数据，确保所有样本都有相同的时间步数（12步）")
    print("2. 修改超分辨率脚本，填充缺失的时间步到12步")
    print("3. 在训练配置中设置 max_sequence_length 为最小时间步数")
else:
    print("✅ 时间步数一致，问题可能出在其他地方")
