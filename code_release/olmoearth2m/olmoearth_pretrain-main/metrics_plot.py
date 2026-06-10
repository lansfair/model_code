import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# 1. 终端命令行参数配置
parser = argparse.ArgumentParser(description="绘制训练指标曲线")
parser.add_argument("--csv", required=True, help="metrics.csv 文件路径")
args = parser.parse_args()

# 2. 自动获取保存路径（默认和 CSV 同文件夹）
csv_dir = os.path.dirname(args.csv)
save_path = os.path.join(csv_dir, "metrics_plot.png")

# 3. 读取数据
df = pd.read_csv(args.csv)

# 4. 绘图
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(df["step"], df["train/ModalityPatchDisc"], color="steelblue", linewidth=1.5)
axes[0].set_ylabel("ModalityPatchDiscMasked")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["step"], df["train/InfoNCE"], color="darkorange", linewidth=1.5)
axes[1].set_ylabel("InfoNCE")
axes[1].set_xlabel("Step")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Training Metrics", fontsize=13)
plt.tight_layout()

# 5. 保存图片
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✅ 图片已保存到: {save_path}")

# python /mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/olmoearth_pretrain-main/metrics_plot.py --csv
# /mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/olmoearth_pretrain-main/local_output/checkpoints/anonymous/split_debug/metrics.csv