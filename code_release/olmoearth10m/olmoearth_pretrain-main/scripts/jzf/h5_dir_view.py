import h5py
import numpy as np
import hdf5plugin
import os
import csv
from pathlib import Path

def analyze_h5_file(file_path):
    """
    分析单个 H5 文件，返回所有数据集的统计信息
    """
    results = []
    file_name = os.path.basename(file_path)

    try:
        with h5py.File(file_path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        # 读取数据
                        data = obj[()]
                        data_flat = data.flatten().astype(np.float64)
                        total = data_flat.size

                        # 基础统计
                        min_val = np.nanmin(data_flat)
                        max_val = np.nanmax(data_flat)
                        mean_val = np.nanmean(data_flat)
                        std_val = np.nanstd(data_flat)

                        # NaN 统计
                        nan_mask = np.isnan(data_flat)
                        nan_count = np.count_nonzero(nan_mask)
                        nan_ratio = (nan_count / total) * 100 if total > 0 else 0

                        # 0 值统计（排除 NaN）
                        valid_data = data_flat[~nan_mask]
                        zero_count = np.count_nonzero(valid_data == 0)
                        zero_ratio = (zero_count / len(valid_data)) * 100 if len(valid_data) > 0 else 0

                        # -99999 特殊值统计
                        special_val = -99999.0
                        special_count = np.count_nonzero(data_flat == special_val)
                        special_ratio = (special_count / total) * 100 if total > 0 else 0

                        # 存入结果
                        results.append({
                            "file": file_name,
                            "dataset": name,
                            "shape": str(data.shape),
                            "dtype": str(data.dtype),
                            "min": round(min_val, 6),
                            "max": round(max_val, 6),
                            "mean": round(mean_val, 6),
                            "std": round(std_val, 6),
                            "nan_count": nan_count,
                            "nan_ratio(%)": round(nan_ratio, 2),
                            "zero_count": zero_count,
                            "zero_ratio(%)": round(zero_ratio, 2),
                            "special_-99999_count": special_count,
                            "special_ratio(%)": round(special_ratio, 2),
                            "total_elements": total
                        })
                    except Exception as e:
                        print(f"⚠️  数据集 {name} 解析失败: {str(e)}")

            f.visititems(visit)
    except Exception as e:
        print(f"❌ 文件 {file_name} 打开失败: {str(e)}")

    return results


def batch_analyze_folder(folder_path, output_csv="h5_statistics_summary.csv"):
    """
    批量分析文件夹下所有 .h5 文件，并输出汇总 CSV
    """
    folder = Path(folder_path)
    all_files = list(folder.glob("*.h5")) + list(folder.glob("*.hdf5"))
    all_results = []

    print(f"📁 找到 {len(all_files)} 个 H5 文件，开始分析...\n")

    # 逐个分析文件
    for i, h5_file in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] 分析: {h5_file.name}")
        file_results = analyze_h5_file(str(h5_file))
        all_results.extend(file_results)

    if not all_results:
        print("❌ 未解析到任何有效数据集")
        return

    # ==================== 汇总统计 ====================
    print("\n📊 生成全文件汇总统计...")
    dataset_summary = {}
    for res in all_results:
        ds_name = res["dataset"]
        if ds_name not in dataset_summary:
            dataset_summary[ds_name] = {
                "files": [],
                "total_elements": 0,
                "nan_count": 0,
                "zero_count": 0,
                "special_count": 0,
                "all_values": []
            }
        dataset_summary[ds_name]["files"].append(res["file"])
        dataset_summary[ds_name]["total_elements"] += res["total_elements"]
        dataset_summary[ds_name]["nan_count"] += res["nan_count"]
        dataset_summary[ds_name]["zero_count"] += res["zero_count"]
        dataset_summary[ds_name]["special_count"] += res["special_-99999_count"]
        dataset_summary[ds_name]["all_values"].append(res["mean"])

    # 构建汇总行
    summary_rows = []
    for ds_name, info in dataset_summary.items():
        total_ele = info["total_elements"]
        nan_ratio = info["nan_count"] / total_ele * 100 if total_ele else 0
        zero_ratio = info["zero_count"] / total_ele * 100 if total_ele else 0
        special_ratio = info["special_count"] / total_ele * 100 if total_ele else 0

        summary_rows.append({
            "file": "=== 全文件汇总 ===",
            "dataset": ds_name,
            "shape": "-",
            "dtype": "-",
            "min": "-",
            "max": "-",
            "mean": "-",
            "std": "-",
            "nan_count": info["nan_count"],
            "nan_ratio(%)": round(nan_ratio, 2),
            "zero_count": info["zero_count"],
            "zero_ratio(%)": round(zero_ratio, 2),
            "special_-99999_count": info["special_count"],
            "special_ratio(%)": round(special_ratio, 2),
            "total_elements": total_ele
        })

    # ==================== 写入 CSV ====================
    headers = [
        "file", "dataset", "shape", "dtype",
        "min", "max", "mean", "std",
        "nan_count", "nan_ratio(%)",
        "zero_count", "zero_ratio(%)",
        "special_-99999_count", "special_ratio(%)",
        "total_elements"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results)
        writer.writerows(summary_rows)

    print(f"\n✅ 分析完成！结果已保存到: {output_csv}")
    print(f"📄 包含 {len(all_files)} 个文件 + 全文件汇总")


# ==================== 运行入口 ====================
if __name__ == "__main__":
    # 1. 设置要分析的文件夹
    TARGET_FOLDER = "/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/zhejiang_10_test"
    
    # 2. 输出 CSV 文件名
    OUTPUT_CSV = "h5_dataset_statistics.csv"
    
    # 3. 开始批量分析
    batch_analyze_folder(TARGET_FOLDER, OUTPUT_CSV)