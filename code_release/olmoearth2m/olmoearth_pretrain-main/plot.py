import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("loss.csv")

# ---------------------- 2. 创建3行1列子图 ----------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), dpi=120)

# ===== 子图 1：Total Loss =====
ax1.plot(df['step'], df['total_loss'], marker='o', color='#1f77b4', linewidth=2)
ax1.set_title('Total Loss', fontweight='bold', fontsize=12)
ax1.set_ylabel('Loss')
ax1.grid(alpha=0.3, linestyle='--')
# 只显示部分刻度（自动间隔，不拥挤）
ax1.locator_params(axis='x', nbins=4)

# ===== 子图 2：PatchDisc Loss =====
ax2.plot(df['step'], df['patchdisc'], marker='s', color='#ff7f0e', linewidth=2)
ax2.set_title('PatchDisc Loss', fontweight='bold', fontsize=12)
ax2.set_ylabel('Loss')
ax2.grid(alpha=0.3, linestyle='--')
ax2.locator_params(axis='x', nbins=4)

# ===== 子图 3：InfoNCE Loss =====
ax3.plot(df['step'], df['infonce_loss'], marker='^', color='#2ca02c', linewidth=2)
ax3.set_title('InfoNCE Loss', fontweight='bold', fontsize=12)
ax3.set_xlabel('Step')
ax3.set_ylabel('Loss')
ax3.grid(alpha=0.3, linestyle='--')
ax3.locator_params(axis='x', nbins=4)

# 整体布局
plt.suptitle('Training Loss Curves', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# ---------------------- 3. 保存并关闭 ----------------------
plt.savefig("loss.png", dpi=150, bbox_inches='tight')
plt.close()

print("✅ 图片已保存：loss.png")