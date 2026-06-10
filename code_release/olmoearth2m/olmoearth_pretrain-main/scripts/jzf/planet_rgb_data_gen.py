import sys
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
import time
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 强制重新加载模块（避免缓存问题）
import importlib
import olmoearth_pretrain.data.constants
importlib.reload(olmoearth_pretrain.data.constants)

from pathlib import Path
from test_planet_rgbnir_real_data import super_resolution_test, TEST_DATA_DIR

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_single_file(h5_file: Path, output_dir: Optional[Path] = None, dry_run: bool = False) -> tuple[Path, bool, str]:
    """处理单个H5文件的包装函数，用于多进程调用。
    
    Args:
        h5_file: H5文件路径
        output_dir: 输出目录，如果为None则使用默认目录
        dry_run: 是否只执行不写入
        
    Returns:
        (文件路径, 是否成功, 状态消息) 的元组
    """
    try:
        # 检查输出文件是否已存在
        if output_dir is None:
            output_dir = TEST_DATA_DIR / "planet_rgbnir_output"
        else:
            output_dir = Path(output_dir)
        
        output_file = output_dir / h5_file.name
        
        if not dry_run and output_file.exists():
            return (h5_file, True, "SKIPPED")
        
        success = super_resolution_test(h5_file, output_dir=output_dir, dry_run=dry_run)
        status = "SUCCESS" if success else "FAILED"
        return (h5_file, success, status)
    except Exception as e:
        logger.error(f"处理文件 {h5_file.name} 时出错: {e}", exc_info=True)
        return (h5_file, False, f"ERROR: {str(e)}")


def main_parallel(num_workers: Optional[int] = None, output_dir: Optional[Path] = None, dry_run: bool = False, force: bool = False):
    """使用多进程并行处理所有H5文件。
    
    Args:
        num_workers: 并行工作进程数，默认为CPU核心数
        output_dir: 输出目录，如果为None则使用默认目录（输入文件同级的"planet_rgbnir_output"）
        dry_run: 是否只执行不写入
        force: 是否强制重新处理已存在的文件
    """
    # 获取所有H5文件
    all_h5_files = sorted(TEST_DATA_DIR.glob("*.h5"))
    
    if not all_h5_files:
        logger.warning(f"在 {TEST_DATA_DIR} 中未找到H5文件")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = TEST_DATA_DIR / "planet_rgbnir_output"
    else:
        output_dir = Path(output_dir)
    
    # 确保输出目录存在
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 过滤需要处理的文件
    if force or dry_run:
        h5_files = all_h5_files
        skipped_count = 0
    else:
        h5_files = []
        skipped_count = 0
        for h5_file in all_h5_files:
            output_file = output_dir / h5_file.name
            if output_file.exists():
                skipped_count += 1
            else:
                h5_files.append(h5_file)
    
    logger.info(f"总共找到 {len(all_h5_files)} 个文件")
    if skipped_count > 0:
        logger.info(f"跳过 {skipped_count} 个已处理的文件")
    logger.info(f"待处理 {len(h5_files)} 个文件")
    
    if not h5_files:
        logger.info("✅ 所有文件已处理完成，无需重复处理")
        return
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = min(cpu_count(), len(h5_files))
    
    logger.info(f"使用 {num_workers} 个并行进程")
    if dry_run:
        logger.info("⚠️  Dry run模式 - 不会写入文件")
    if force:
        logger.info("⚠️  Force模式 - 将重新处理所有文件")
    
    # 创建进程池并并行处理
    process_func = partial(process_single_file, output_dir=output_dir, dry_run=dry_run)
    
    start_time = time.time()
    results = []
    
    # 使用 imap_unordered 实现实时进度显示
    with Pool(processes=num_workers) as pool:
        total = len(h5_files)
        with tqdm(total=total, desc="处理进度", unit="file") as pbar:
            for result in pool.imap_unordered(process_func, h5_files, chunksize=1):
                results.append(result)
                h5_file, success, status = result
                
                # 更新进度条描述
                if status == "SKIPPED":
                    pbar.set_postfix({"状态": "跳过"})
                elif success:
                    pbar.set_postfix({"状态": "成功"})
                else:
                    pbar.set_postfix({"状态": "失败", "文件": h5_file.name[:20]})
                
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for _, success, _ in results if success)
    failed_results = [(path, status) for path, success, status in results if not success]
    
    # 计算平均处理速度
    files_per_second = len(results) / elapsed_time if elapsed_time > 0 else 0
    estimated_total_time = len(all_h5_files) / files_per_second if files_per_second > 0 else 0
    
    logger.info(f"\n{'='*80}")
    logger.info(f"处理完成!")
    logger.info(f"总计: {len(all_h5_files)} 个文件")
    logger.info(f"本次处理: {len(h5_files)} 个文件")
    logger.info(f"跳过: {skipped_count} 个文件")
    logger.info(f"成功: {success_count} 个文件")
    logger.info(f"失败: {len(failed_results)} 个文件")
    logger.info(f"耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    logger.info(f"处理速度: {files_per_second:.2f} 文件/秒")
    logger.info(f"预估全部处理时间: {estimated_total_time/60:.2f} 分钟")
    
    if failed_results:
        logger.warning("\n❌ 失败的文件列表:")
        for f, status in failed_results:
            logger.warning(f"  - {f.name}: {status}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="并行处理Planet RGB数据生成")
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None,
        help="并行工作进程数 (默认: CPU核心数)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录路径 (默认: <测试数据目录>/planet_rgbnir_output)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只执行超分辨率计算，不写入文件"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理所有文件，忽略已存在的输出"
    )
    
    args = parser.parse_args()
    
    main_parallel(
        num_workers=args.workers, 
        output_dir=args.output_dir, 
        dry_run=args.dry_run,
        force=args.force
    )
