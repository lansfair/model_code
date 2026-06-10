"""Generate sample_metadata.csv and latlon_distribution.npy for the H5 dataset."""

import h5py
import hdf5plugin  # noqa: F401 - needed for decompression
import pandas as pd
import numpy as np
import os
from pathlib import Path

# 配置路径
h5_dir = Path("/mnt/ht2-nas2/00-model/00-jiangzf/coderepo/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map_planet_rgbnir/3996")

# 从目录名推断支持的模态
modalities_str = h5_dir.parent.name
print(f"Modalities string: {modalities_str}")

# 解析模态名称（需要特殊处理组合模态）
modalities_raw = modalities_str.split("_")
supported_modalities = []

# 特殊处理组合模态
i = 0
while i < len(modalities_raw):
    if modalities_raw[i] == "sentinel2" and i + 1 < len(modalities_raw) and modalities_raw[i+1] == "l2a":
        supported_modalities.append("sentinel2_l2a")
        i += 2
    elif modalities_raw[i] == "openstreetmap" and i + 1 < len(modalities_raw) and modalities_raw[i+1] == "raster":
        supported_modalities.append("openstreetmap_raster")
        i += 2
    elif modalities_raw[i] == "wri" and i + 1 < len(modalities_raw) and modalities_raw[i+1] == "canopy" and i + 2 < len(modalities_raw) and modalities_raw[i+2] == "height" and i + 3 < len(modalities_raw) and modalities_raw[i+3] == "map":
        supported_modalities.append("wri_canopy_height_map")
        i += 4
    elif modalities_raw[i] == "planet" and i + 1 < len(modalities_raw) and modalities_raw[i+1] == "rgbnir":
        supported_modalities.append("planet_rgbnir")
        i += 2
    else:
        supported_modalities.append(modalities_raw[i])
        i += 1

print(f"Supported modalities: {supported_modalities}")

# ============================================================================
# 1. 生成 sample_metadata.csv
# ============================================================================
print("\n=== Generating sample_metadata.csv ===")

metadata_dict = {
    "sample_index": [],
}
for modality in supported_modalities:
    metadata_dict[modality] = []

num_samples = int(h5_dir.name)
print(f"Processing {num_samples} samples...")

for idx in range(num_samples):
    h5_file = h5_dir / f"sample_{idx}.h5"
    
    if not h5_file.exists():
        print(f"Warning: {h5_file} does not exist, skipping...")
        continue
    
    try:
        with h5py.File(str(h5_file), 'r') as f:
            available_keys = set(f.keys()) - {'latlon', 'timestamps'}
            
            metadata_dict["sample_index"].append(idx)
            
            for modality in supported_modalities:
                metadata_dict[modality].append(1 if modality in available_keys else 0)
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{num_samples} samples")
            
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        continue

df = pd.DataFrame(metadata_dict)
csv_path = h5_dir / "sample_metadata.csv"
df.to_csv(csv_path, index=False)

print(f"✓ Created {csv_path}")
print(f"  Total samples: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# 2. 生成 latlon_distribution.npy
# ============================================================================
print("\n=== Generating latlon_distribution.npy ===")

latlon_list = []
print(f"Extracting lat/lon from {num_samples} samples...")

for idx in range(num_samples):
    h5_file = h5_dir / f"sample_{idx}.h5"
    
    if not h5_file.exists():
        print(f"Warning: {h5_file} does not exist, skipping...")
        continue
    
    try:
        with h5py.File(str(h5_file), 'r') as f:
            if 'latlon' in f:
                latlon_data = f['latlon'][:]
                # latlon 可能是 [lat, lon] 格式
                if latlon_data.ndim == 1 and len(latlon_data) == 2:
                    latlon_list.append(latlon_data)
                elif latlon_data.ndim == 2 and latlon_data.shape[1] == 2:
                    # 如果已经是 [N, 2] 格式，取第一个
                    latlon_list.append(latlon_data[0])
                else:
                    print(f"Warning: Unexpected latlon shape {latlon_data.shape} for sample {idx}")
                    continue
            else:
                print(f"Warning: No 'latlon' key in {h5_file}, skipping...")
                continue
        
        if (idx + 1) % 500 == 0:
            print(f"Extracted {idx + 1}/{num_samples} samples")
            
    except Exception as e:
        print(f"Error extracting latlon from sample {idx}: {e}")
        continue

if latlon_list:
    latlon_array = np.array(latlon_list)
    latlon_path = h5_dir / "latlon_distribution.npy"
    with open(latlon_path, 'wb') as f:
        np.save(f, latlon_array)
    
    print(f"✓ Created {latlon_path}")
    print(f"  Shape: {latlon_array.shape}")
    print(f"  Lat range: [{latlon_array[:, 0].min():.4f}, {latlon_array[:, 0].max():.4f}]")
    print(f"  Lon range: [{latlon_array[:, 1].min():.4f}, {latlon_array[:, 1].max():.4f}]")
else:
    print("✗ No latlon data extracted!")

print("\n=== Done! ===")
