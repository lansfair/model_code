import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/olmoearth_pretrain-main/local_output/checkpoints/anonymous/dataset_debug_10m_02/metrics.csv")
df = pd.read_csv("/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/olmoearth_pretrain-main/scripts/jzf/metrics_ema.csv")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(df["step"], df["train/ModalityPatchDisc"], color="steelblue", linewidth=1.5)
axes[0].set_ylabel("ModalityPatchDisc")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["step"], df["train/InfoNCE"], color="darkorange", linewidth=1.5)
axes[1].set_ylabel("InfoNCE")
axes[1].set_xlabel("Step")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Training Metrics", fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/olmoearth_pretrain-main/scripts/jzf/metrics_plot_ema.png", dpi=150)
print("Saved metrics_plot.png")