import h5py
import hdf5plugin
import numpy as np
import os
from upath import UPath
from typing import Tuple, Dict, Union, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from pathlib import Path

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

        # 检查是否有保留的时刻
        if not np.any(keep):
            # 所有时刻都被丢弃，跳过这个模态（不添加到结果中）
            print(f"  {modality}: 所有时刻都被丢弃（{T}个时刻），从结果中移除")
            continue
        
        # 更新全局掩码：将需要丢弃的时刻对应的位置设为False
        new_timestamps_mask = timestamps_mask.copy()
        for t, global_idx in enumerate(true_indices):
            if not keep[t]:
                new_timestamps_mask[global_idx] = False
        
        # # 过滤数据
        # if not np.any(keep):
        #     filtered_data = np.empty((H, W, 0, C), dtype=data.dtype)
        # else:
        filtered_data = data[:, :, keep, :]
        
        filtered_sample_dict[modality] = filtered_data
        filtered_masks_dict[modality] = new_timestamps_mask
    
    # 保留那些在sample_dict中但不在missing_timesteps_masks中的模态（没有掩码，不进行过滤）
    for modality in sample_dict:
        if modality not in filtered_sample_dict and modality not in missing_timesteps_masks:
            filtered_sample_dict[modality] = sample_dict[modality]
            print(f"  {modality}: 无掩码信息，直接保留")
    # 输出统计信息
    print(f"\n过滤完成: 保留 {len(filtered_sample_dict)}/{len(sample_dict)} 个模态")


    # 二次校验
    for modality in filtered_masks_dict.keys():
        # 跳过不在sample_dict中的模态
        if modality not in filtered_sample_dict:
            print(f"警告: 模态 {modality} 在数据中不存在，跳过")
            continue
            
        timestamps_mask = filtered_masks_dict[modality]
        true_indices = np.where(timestamps_mask)[0]
        data = filtered_sample_dict[modality]  # [H, W, T, C]
        
        H, W, T, C = data.shape
        
        if len(true_indices) != T:
            raise ValueError(
                f"模态 {modality}: 数据时刻数 T={T} "
                f"与掩码中True数量 {len(true_indices)} 不一致"
            )
    
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


def save_h5_with_preserved_compression(
    sample_dict: Dict,
    missing_timesteps_masks: Dict,
    output_path: Union[str, UPath],
    compression_info: Optional[Dict] = None,
    force_compression: Optional[str] = None,  # 强制使用指定压缩
    force_no_compression: bool = False
) -> None:
    """
    保存HDF5文件，可以选择保持原始压缩或强制使用新压缩
    
    Args:
        sample_dict: 各模态的数据字典
        missing_timesteps_masks: 各模态的掩码字典
        output_path: 输出文件路径
        compression_info: 原始压缩信息（可选，如果不提供则使用gzip）
        force_compression: 强制使用指定的压缩算法 ('gzip', 'zstd', 'lz4', None)
        force_no_compression: 是否完全不压缩
    """
    out_path_str = str(output_path) if isinstance(output_path, UPath) else output_path
    
    # 确保输出目录存在
    from pathlib import Path
    Path(out_path_str).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(out_path_str, 'w') as h5file:
        # 保存各模态数据
        for modality, data in sample_dict.items():
            create_kwargs = {}
            
            if force_no_compression:
                # 不压缩
                create_kwargs["compression"] = None
                print(f"  {modality}: 不压缩")
                
            elif force_compression is not None:
                # 强制使用指定压缩
                if force_compression == "gzip":
                    create_kwargs["compression"] = "gzip"
                    create_kwargs["compression_opts"] = 4
                    create_kwargs["shuffle"] = True
                elif force_compression == "zstd":
                    create_kwargs["compression"] = hdf5plugin.Zstd(clevel=3)
                    create_kwargs["shuffle"] = True
                elif force_compression == "lz4":
                    create_kwargs["compression"] = hdf5plugin.LZ4()
                print(f"  {modality}: 强制使用 {force_compression} 压缩")
                
            elif compression_info and modality in compression_info:
                # 保持原始压缩
                orig_comp = compression_info[modality]
                if orig_comp['compression'] is not None:
                    # 处理原始压缩
                    if orig_comp['compression'] in ['gzip', 'lzf']:
                        # 原生压缩
                        create_kwargs["compression"] = orig_comp['compression']
                        if orig_comp['compression_opts']:
                            create_kwargs["compression_opts"] = orig_comp['compression_opts']
                    elif orig_comp['compression'] == 'zstd':
                        # 需要hdf5plugin
                        create_kwargs["compression"] = hdf5plugin.Zstd(
                            clevel=orig_comp['compression_opts'] if orig_comp['compression_opts'] else 3
                        )
                    else:
                        # 其他压缩或默认使用gzip
                        print(f"  警告: {modality} 原始压缩 {orig_comp['compression']} 不兼容，使用 gzip")
                        create_kwargs["compression"] = "gzip"
                        create_kwargs["compression_opts"] = 4
                    
                    if orig_comp.get('shuffle'):
                        create_kwargs["shuffle"] = True
                    if orig_comp.get('chunks'):
                        create_kwargs["chunks"] = orig_comp['chunks']
                    
                    print(f"  {modality}: 保持原始压缩 {orig_comp['compression']}")
                else:
                    print(f"  {modality}: 原始无压缩，保持无压缩")
                    create_kwargs["compression"] = None
            else:
                # 默认使用gzip压缩
                create_kwargs["compression"] = "gzip"
                create_kwargs["compression_opts"] = 4
                create_kwargs["shuffle"] = True
                print(f"  {modality}: 默认使用 gzip 压缩")
            
            # 创建数据集
            h5file.create_dataset(modality, data=data, **create_kwargs)
        
        # 保存掩码组（掩码通常不压缩）
        if missing_timesteps_masks:
            mask_group = h5file.create_group(MASK_GROUP_NAME)
            for modality, mask in missing_timesteps_masks.items():
                # 掩码不压缩，因为布尔值压缩效果差
                mask_group.create_dataset(modality, data=mask, compression=None)
    
    # 输出文件大小
    file_size_mb = Path(out_path_str).stat().st_size / (1024 * 1024)
    print(f"\n数据已保存到: {out_path_str}")
    print(f"文件大小: {file_size_mb:.2f} MB")

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


def process_single_file_with_progress(
    args: Tuple,
    progress_counter=None,
    lock=None
) -> Tuple[bool, str, str]:
    """
    带进度条的单文件处理函数（用于显示进度）
    
    Args:
        args: (input_path, output_path, threshold, file_index, total_files)
        progress_counter: 共享计数器
        lock: 进程锁
        
    Returns:
        (是否成功, 文件名, 错误信息或成功信息)
    """
    input_path, output_path, threshold, file_index, total_files = args
    
    try:
        filename = Path(str(input_path)).name
        
        # 读取数据
        sample_dict, masks = read_h5(input_path)
        
        # 过滤数据
        filtered_dict, filtered_masks = filter_by_zero_ratio(sample_dict, masks, threshold)
        
        # 保存数据
        # save_h5(filtered_dict, filtered_masks, output_path)
        
        # 更新进度
        if lock and progress_counter:
            with lock:
                progress_counter.value += 1
                current = progress_counter.value
                print(f"进度: [{current}/{total_files}] 完成 - {filename}")
        
        return True, filename, f"处理成功"
        
    except Exception as e:
        error_msg = str(e)
        
        # 更新进度
        if lock and progress_counter:
            with lock:
                progress_counter.value += 1
                current = progress_counter.value
                print(f"进度: [{current}/{total_files}] 失败 - {Path(str(input_path)).name}")
        
        return False, Path(str(input_path)).name, error_msg

def process_files_with_progress(
    input_dir: Union[str, UPath],
    output_dir: Union[str, UPath],
    threshold: float = THRESHOLD,
    num_workers: Optional[int] = None,
    file_pattern: str = ".h5"
) -> Dict:
    """
    带进度显示的多进程批量处理
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        threshold: 零值比例阈值
        num_workers: 工作进程数
        file_patter: 文件匹配模式
    Returns:
        处理结果统计字典
    """
    from multiprocessing import Manager
    # 转换路径
    input_dir = UPath(input_dir) if isinstance(input_dir, str) else input_dir
    output_dir = UPath(output_dir) if isinstance(output_dir, str) else output_dir
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    # 获取所有H5文件
    h5_files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and str(file_path).endswith(file_pattern):
            h5_files.append(file_path)
    
    if not h5_files:
        print(f"在目录 {input_dir} 中未找到 {file_pattern} 文件")
        return {"success": [], "failed": [], "total": 0}
    total_files = len(h5_files)
    print(f"找到 {total_files} 个H5文件")
    print(f"输出目录: {output_dir}")
    print(f"工作进程数: {num_workers if num_workers else cpu_count()}")
    print("-" * 80)

    # 准备任务参数
    tasks = []
    for idx, input_path in enumerate(h5_files, 1):
        output_path = output_dir / input_path.name
        tasks.append((input_path, output_path, threshold, idx, total_files))
    
    # 设置进程数
    if num_workers is None:
        num_workers = min(cpu_count(), total_files)
    
    # 创建共享变量用于进度跟踪
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # 创建部分函数
    process_func = partial(process_single_file_with_progress, 
                          progress_counter=progress_counter, 
                          lock=lock)
    
    # 执行多进程处理
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_func, tasks)
    
    elapsed_time = time.time() - start_time
    # 统计结果
    successful = []
    failed = []
    for success, filename, msg in results:
        if success:
            successful.append(filename)
        else:
            failed.append((filename, msg))
    
    # 输出总结
    print("=" * 80)
    print(f"处理完成！总耗时: {elapsed_time:.2f}秒")
    print(f"成功: {len(successful)}/{total_files} 个文件")
    print(f"失败: {len(failed)}/{total_files} 个文件")
    
    if failed:
        print("\n失败文件列表:")
        for filename, error_msg in failed:
            print(f"  - {filename}: {error_msg}")
    
    return {
        "success": successful,
        "failed": failed,
        "total": total_files,
        "elapsed_time": elapsed_time
    }

# 使用示例
if __name__ == "__main__":
    # 输入输出路径

    input_dir = "/mnt/qh2-nas3/data_verification/olmo2m0602"
    output_dir = "/mnt/ht2_nas2/00-model/00-limx/data_h5"
    
    # 方式1: 基本多进程处理
    results = process_files_with_progress(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold=THRESHOLD,
        num_workers=4,  # 指定使用4个进程，不指定则使用CPU核心数
        file_pattern=".h5"
    )
    # path = "/mnt/qh2-nas3/data_verification/olmo2m0602"
    # output_base_path = UPath("/mnt/ht2_nas2/00-model/00-limx/data_h5")
    # if not os.path.exists(output_base_path):
    #     os.makedirs(output_base_path)
    # imgLists = os.listdir(path)
    # for imgList in imgLists:
    #     if not imgList[-3:] == ".h5":
    #         continue
    #     input_path = UPath(os.path.join(path, imgList))
    #     output_path = output_base_path / imgList
    # input_path = UPath("/mnt/qh2-nas3/data_verification/olmo2m0602/sample_48.h5")
    # input_path = UPath("/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/245/sample_9.h5")
    
    # output_path = output_path / "sample_48.h5"
    # output_path = input_path.parent / "sample_9_filtered.h5"  # 保存到同目录
    
        # 方式1: 分步执行
        # sample_dict, masks = read_h5(input_path)
        # filtered_dict, filtered_masks = filter_by_zero_ratio(sample_dict, masks)
    # save_h5(filtered_dict, filtered_masks, output_path)
    # save_h5_with_preserved_compression(filtered_dict, filtered_masks, output_path, compression_info="zstd")
    
    # 方式2: 一键处理
    # filtered_dict, filtered_masks = process_and_save(input_path, output_path)