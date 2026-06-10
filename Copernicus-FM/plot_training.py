import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def plot_training_log(log_path="output_dir/log.txt", output_dir="output_dir"):
    """读取 log.txt 并绘制训练曲线"""

    # 读取日志
    epochs = []
    lr = []
    loss = []
    loss_mae = []
    loss_distill = []

    with open(log_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            epochs.append(data['epoch'])
            lr.append(data['train_lr'])
            loss.append(data['train_loss'])
            loss_mae.append(data['train_loss_mae'])
            loss_distill.append(data['train_loss_distill'])

    epochs = np.array(epochs)
    lr = np.array(lr)
    loss = np.array(loss)
    loss_mae = np.array(loss_mae)
    loss_distill = np.array(loss_distill)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Loss 曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, loss, 'b-', linewidth=2, label='total loss')
    ax1.plot(epochs, loss_mae, 'g-', linewidth=2, label='mae loss')
    ax1.plot(epochs, loss_distill, 'r-', linewidth=2, label='distill loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Learning Rate 曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, lr, 'purple', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    # 科学计数法格式
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # 3. Loss (log scale)
    ax3 = axes[1, 0]
    ax3.semilogy(epochs, loss, 'b-', linewidth=2, label='total loss')
    ax3.semilogy(epochs, loss_mae, 'g-', linewidth=2, label='mae loss')
    ax3.semilogy(epochs, loss_distill, 'r-', linewidth=2, label='distill loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('Training Loss (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Loss 下降率
    ax4 = axes[1, 1]
    if len(loss) > 1:
        loss_ratio = loss / loss[0] * 100
        mae_ratio = loss_mae / loss_mae[0] * 100
        distill_ratio = loss_distill / loss_distill[0] * 100
        ax4.plot(epochs, loss_ratio, 'b-', linewidth=2, label=f'total: {loss_ratio[-1]:.1f}%')
        ax4.plot(epochs, mae_ratio, 'g-', linewidth=2, label=f'mae: {mae_ratio[-1]:.1f}%')
        ax4.plot(epochs, distill_ratio, 'r-', linewidth=2, label=f'distill: {distill_ratio[-1]:.1f}%')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss % (relative to epoch 0)')
    ax4.set_title('Loss Reduction Progress')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 110)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    # 保存图片
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存到: {output_path}")

    # 打印统计信息
    print(f"\n训练统计:")
    print(f"  Epochs: {epochs[-1] + 1}")
    print(f"  初始 Loss: {loss[0]:.4f} → 最终 Loss: {loss[-1]:.4f} ({(loss[-1]/loss[0]*100):.1f}%)")
    print(f"  初始 MAE: {loss_mae[0]:.4f} → 最终 MAE: {loss_mae[-1]:.4f} ({(loss_mae[-1]/loss_mae[0]*100):.1f}%)")
    print(f"  初始 Distill: {loss_distill[0]:.4f} → 最终 Distill: {loss_distill[-1]:.4f} ({(loss_distill[-1]/loss_distill[0]*100):.1f}%)")
    print(f"  峰值 LR: {lr.max():.6f}")

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    # parser.add_argument("--log", default="output_dir/log.txt", help="log.txt 路径")
    parser.add_argument("--log", default="/mnt/ht2_nas2/00-model/00-ds/Copernicus-FM/h5towds/Copernicus-FM/Copernicus_ZJ.txt", help="log.txt 路径")
    parser.add_argument("--output", default="/mnt/ht2_nas2/00-model/00-ds/Copernicus-FM/h5towds/Copernicus-FM/src/output_dir_Copernicus_ZJ0610", help="图片输出目录")
    args = parser.parse_args()

    plot_training_log(args.log, args.output)