import h5py
import hdf5plugin
import numpy as np
import os
from upath import UPath
from typing import Tuple, Dict, Union

modalities = ["lt1", "rgb", "sar", "sentinel1", "sentinel2_l2a"]
# training_modalities = ["landcover_1m", "landcover_30m", "landsat", "lt1", "rgb", "sar", "sentinel1", "sentinel2_l2a", "srtm", "worldcereal", "worldcover"]
MASK_GROUP_NAME = "missing_timesteps_masks"
THRESHOLD = 0.1
def read_h5(h5_file_path: Union[str, UPath]) -> Tuple[Dict, Dict]:
    """
        读取h5文件，返回各模态的数据和missing timestamps掩码
        
        Args:
            h5_file_path: HDF5文件路径
            
        Returns:
            sample_dict: 各模态的数据字典，形状为 [H, W, T, C]
            missing_timesteps_masks: 各模态的掩码字典，每个值为布尔数组
    """
    with h5_file_path.open("rb") as f:
        with h5py.File(f, "r") as h5file:
            # timestamps should not be a floating string
            sample_dict = {
                k: v[()] for k, v in h5file.items()
                if k != MASK_GROUP_NAME
            }
            # 读取掩码
            missing_timesteps_masks = {}
            if MASK_GROUP_NAME in h5file:
                missing_timesteps_masks = {
                    k: v[()] for k, v in h5file[MASK_GROUP_NAME].items()
                }
    
    return sample_dict, missing_timesteps_masks

def filter_by_zero_ratio(
    sample_dict: Dict,
    missing_timesteps_masks: Dict,
    threshold: float = THRESHOLD
) -> Tuple[Dict, Dict]:
    """
        根据零值比例过滤无效时刻，并同步更新掩码
        
        Args:
            sample_dict: 各模态的数据字典，形状为 [H, W, T, C]
            missing_timesteps_masks: 各模态的掩码字典
            threshold: 零值比例阈值，默认0.1
            
        Returns:
            过滤后的数据字典和掩码字典
    """
    filtered_sample_dict = {}
    filtered_masks_dict = {}
    for modality in missing_timesteps_masks.keys():
        # 跳过不在sample_dict中的模态
        if modality not in sample_dict:
            print(f"警告: 模态 {modality} 在数据中不存在，跳过")
            continue
            
        timestamps_mask = missing_timesteps_masks[modality]
        true_indices = np.where(timestamps_mask)[0]
        data = sample_dict[modality]  # [H, W, T, C]
        
        H, W, T, C = data.shape
        
        if len(true_indices) != T:
            raise ValueError(
                f"模态 {modality}: 数据时刻数 T={T} "
                f"与掩码中True数量 {len(true_indices)} 不一致"
            )
        
        # 计算每个时刻的零值比例（优化版本）
        # 方法1: 使用 reshape 避免 moveaxis 的开销
        # data_reshaped = data.reshape(H * W, T, C)  # 但这会改变内存布局
        # 更直接的方法：使用 np.sum 直接在原始维度上计算
        
        # 将数据转置为 (T, H, W, C) 以便统计
        data_t_first = np.transpose(data, (2, 0, 1, 3))  # (T, H, W, C)
        total_pixels = H * W * C
        
        # 计算每个时刻的零值比例
        zero_ratios = np.sum(data_t_first == 0, axis=(1, 2, 3)) / total_pixels
        
        # 生成保留掩码 (长度为T)
        keep = zero_ratios <= threshold
        
        # 更新全局掩码：将需要丢弃的时刻对应的位置设为False
        new_timestamps_mask = timestamps_mask.copy()
        for t, global_idx in enumerate(true_indices):
            if not keep[t]:
                new_timestamps_mask[global_idx] = False
        
        # 过滤数据
        if not np.any(keep):
            filtered_data = np.empty((H, W, 0, C), dtype=data.dtype)
        else:
            filtered_data = data[:, :, keep, :]
        
        filtered_sample_dict[modality] = filtered_data
        filtered_masks_dict[modality] = new_timestamps_mask
    
    # 保留那些在sample_dict中但不在missing_timesteps_masks中的模态（没有掩码，不进行过滤）
    for modality in sample_dict:
        if modality not in filtered_sample_dict:
            filtered_sample_dict[modality] = sample_dict[modality]
            # 如果有掩码但未处理，这里可以添加警告
    
    return filtered_sample_dict, filtered_masks_dict


def save_h5(
    sample_dict: Dict,
    missing_timesteps_masks: Dict,
    output_path: Union[str, UPath]
) -> None:
    """
    将过滤后的数据保存为HDF5文件，保持与原始文件相同的格式
    
    Args:
        sample_dict: 各模态的数据字典
        missing_timesteps_masks: 各模态的掩码字典
        output_path: 输出文件路径
    """
    # 转换路径为字符串
    out_path_str = str(output_path) if isinstance(output_path, UPath) else output_path
    
    with h5py.File(out_path_str, 'w') as h5file:
        # 保存各模态数据
        for modality, data in sample_dict.items():
            h5file.create_dataset(modality, data=data, compression='gzip')
        
        # 保存掩码组
        if missing_timesteps_masks:
            mask_group = h5file.create_group(MASK_GROUP_NAME)
            for modality, mask in missing_timesteps_masks.items():
                mask_group.create_dataset(modality, data=mask, compression='gzip')
    
    print(f"数据已保存到: {out_path_str}")
    print(f"  - 模态数: {len(sample_dict)}")
    print(f"  - 掩码组: {MASK_GROUP_NAME}")


def process_and_save(
    input_path: Union[str, UPath],
    output_path: Union[str, UPath],
    threshold: float = THRESHOLD
) -> Tuple[Dict, Dict]:
    """
    完整处理流程：读取 -> 过滤 -> 保存
    
    Args:
        input_path: 输入HDF5文件路径
        output_path: 输出HDF5文件路径
        threshold: 零值比例阈值
        
    Returns:
        过滤后的数据字典和掩码字典
    """
    print(f"读取文件: {input_path}")
    sample_dict, masks = read_h5(input_path)
    
    print(f"原始数据统计:")
    for modality, data in sample_dict.items():
        if modality in masks:
            t_original = data.shape[2] if len(data.shape) == 4 else data.shape[0]
            print(f"  - {modality}: 形状 {data.shape}, 有效时刻数 {np.sum(masks[modality])}")
        else:
            print(f"  - {modality}: 形状 {data.shape}")
    
    print(f"\n开始过滤 (阈值={threshold})...")
    filtered_dict, filtered_masks = filter_by_zero_ratio(sample_dict, masks, threshold)
    
    print(f"\n过滤后统计:")
    for modality, data in filtered_dict.items():
        if modality in filtered_masks:
            t_new = data.shape[2] if len(data.shape) == 4 else data.shape[0]
            kept = np.sum(filtered_masks[modality])
            print(f"  - {modality}: 形状 {data.shape}, 有效时刻数 {kept}")
        else:
            print(f"  - {modality}: 形状 {data.shape}")
    
    print(f"\n保存文件: {output_path}")
    save_h5(filtered_dict, filtered_masks, output_path)
    
    return filtered_dict, filtered_masks


# 使用示例
if __name__ == "__main__":
    # 输入输出路径
    path = "/mnt/qh2-nas3/data_verification/olmo2m0602"
    imgLists = os.listdir(path)
    unique_num = []
    key = "worldcereal"
    # key2 = "landcover_30m"
    for imgList in imgLists:
        if imgList[-3:] == ".h5":
            input_path = UPath(os.path.join(path, imgList))
            sample_dict, masks = read_h5(input_path)
            # if key in sample_dict.keys() and key2 in sample_dict.keys():
            if key in sample_dict.keys():
                print(np.unique(sample_dict[key]))
                # print(np.unique(sample_dict[key2]))
                print(sample_dict[key].shape)
                # print(np.array_equal(sample_dict[key],sample_dict[key2]))
                # unique_num = set(unique_num) | set(np.unique(sample_dict[key]))
    # print(unique_num)
    # input_path = UPath("/mnt/ht2_nas2/QH_Group/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/3996/sample_12.h5")
    # # input_path = UPath("/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/245/sample_9.h5")
    # output_path = UPath("/mnt/ht2_nas2/00-model/00-limx/data_h5")
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # output_path = output_path / "sample_120.h5"
    # # output_path = input_path.parent / "sample_9_filtered.h5"  # 保存到同目录
    
    # # 方式1: 分步执行
    # sample_dict, masks = read_h5(input_path)
    # filtered_dict, filtered_masks = filter_by_zero_ratio(sample_dict, masks)
    # save_h5(filtered_dict, filtered_masks, output_path)
    
    # # 方式2: 一键处理
    # # filtered_dict, filtered_masks = process_and_save(input_path, output_path)