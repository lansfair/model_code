# Planet RGBNIR 超分辨率 - Notebook使用说明

## 📋 快速开始

### 1. 在Notebook中运行测试

在 `pretrain_playground.ipynb` 中添加一个新的代码单元格,然后运行:

```python
# 运行超分辨率测试脚本(Dry Run模式)
%run test_planet_rgbnir_real_data.py
```

### 2. 实际处理数据(写入新文件)

Dry Run测试通过后,在Notebook中运行:

```python
from pathlib import Path
from test_planet_rgbnir_real_data import super_resolution_test, TEST_DATA_DIR

# 获取测试文件
h5_files = sorted(TEST_DATA_DIR.glob("*.h5"))
test_file = h5_files[0]

# 执行超分辨率并写入新文件
success = super_resolution_test(test_file, dry_run=False)

if success:
    output_path = test_file.parent / "planet_rgbnir_output" / test_file.name
    print(f"✅ 成功! 输出文件: {output_path}")
```

### 3. 批量处理所有文件

```python
from pathlib import Path
from test_planet_rgbnir_real_data import super_resolution_test, TEST_DATA_DIR

# 获取所有H5文件
h5_files = sorted(TEST_DATA_DIR.glob("*.h5"))
print(f"找到 {len(h5_files)} 个文件")

# 批量处理
success_count = 0
for h5_file in h5_files:
    print(f"\n处理: {h5_file.name}")
    if super_resolution_test(h5_file, dry_run=False):
        success_count += 1

print(f"\n完成! 成功: {success_count}/{len(h5_files)}")
```

或者直接在命令行运行:

```bash
cd /mnt/ht2-nas2/00-model/00-jiangzf/coderepo/filetrans/olmoearth_pretrain-main/scripts/jzf
python test_planet_rgbnir_real_data.py
```

## 📁 文件说明

- **test_planet_rgbnir_real_data.py**: 使用真实测试数据的测试脚本
- **测试数据路径**: `/mnt/ht2-nas2/00-model/00-jiangzf/coderepo/3996`
- **输出目录**: `{输入文件目录}/planet_rgbnir_output/`
- **输出模态**: `planet_rgbnir` (添加到新的H5文件中)

## 🔍 测试流程

1. **检查H5文件结构** - 查看现有的数据集
2. **Dry Run测试** - 执行超分辨率但不写入(验证逻辑)
3. **写入新文件** - 创建包含原始数据和planet_rgbnir的新H5文件

## ⚙️ 核心功能

### 超分辨率函数

```python
def super_resolution_bicubic(rgb_10m, scale_factor):
    """使用双三次插值进行超分辨率"""
    # 对每个时间步独立处理
    for i in range(t):
        rgb_upsampled = cv2.resize(
            rgb_timestep,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC
        )
    return rgb_hr
```

### 数据处理流程

1. 读取 `sentinel2_l2a` 数据 (10m分辨率)
2. 提取RGB波段 (B04, B03, B02)
3. 计算缩放因子 (基于tile size)
4. 使用双三次插值上采样
5. **创建新H5文件**,复制所有原始数据
6. 在新文件中添加 `planet_rgbnir` 模态

## 📊 预期结果

```
================================================================================
Planet RGBNIR 超分辨率 - 真实数据测试
================================================================================

测试数据目录: /mnt/ht2-nas2/00-model/00-jiangzf/coderepo/3996
找到 X 个H5文件

选择测试文件: sample_XXXXX.h5

================================================================================
检查文件: sample_XXXXX.h5
================================================================================

数据集列表:
  - sentinel2_l2a: shape=(256, 256, 12, 12), dtype=int16, compression=zstd
  - timestamps: shape=(12, 3), dtype=float64, compression=None
  ...

✅ 加载 sentinel2_l2a: shape=(256, 256, 12, 12), dtype=int16
✅ RGB波段索引: R=B04(2), G=B03(1), B=B02(0)
✅ 提取RGB: shape=(256, 256, 12, 3)
✅ 缩放因子: 3.33 (256 -> 853)

🔄 执行超分辨率: (256, 256) -> (853, 853)
✅ 超分辨率完成: shape=(853, 853, 12, 3), dtype=int16
   数据范围: [min_value, max_value]

⏭️  Dry run模式,跳过写入

================================================================================
✅ 测试成功!
💡 要实际写入新文件,请使用:
   super_resolution_test(test_file, dry_run=False)
   输出将保存到: /path/to/planet_rgbnir_output/sample_XXXXX.h5
================================================================================
```

## 📂 输出文件结构

```
/mnt/ht2-nas2/00-model/00-jiangzf/coderepo/3996/
├── sample_00001.h5              # 原始文件(不变)
├── sample_00002.h5              # 原始文件(不变)
└── planet_rgbnir_output/        # 新建的输出目录
    ├── sample_00001.h5          # 包含原始数据 + planet_rgbnir
    └── sample_00002.h5          # 包含原始数据 + planet_rgbnir
```

每个输出文件包含:
- 所有原始数据集(sentinel2_l2a, timestamps等)
- 新增的 `planet_rgbnir` 数据集(超分辨率后的RGB)

## ⚠️ 注意事项

1. ✅ **不会修改原文件** - 所有输出都写入新文件
2. ✅ **首次运行使用Dry Run** - 先用 `dry_run=True` 测试
3. ⚠️ **磁盘空间** - 新文件会比原文件大约11倍(因为添加了超分辨率数据)
4. ⚠️ **内存使用** - 确保有足够内存处理大文件
5. ✅ **压缩设置** - 默认使用zstd压缩(level=3)

## 🛠️ 自定义参数

### 指定输出目录

```python
from pathlib import Path

# 自定义输出目录
custom_output_dir = Path("/path/to/custom/output")
success = super_resolution_test(
    test_file, 
    output_dir=custom_output_dir,
    dry_run=False
)
```

### 修改其他参数

如果需要修改源模态、目标模态或压缩设置,可以编辑 `test_planet_rgbnir_real_data.py` 中的相应部分。

## 📈 性能优化建议

对于大批量处理:

```python
from tqdm import tqdm

# 批量处理所有文件
h5_files = sorted(TEST_DATA_DIR.glob("*.h5"))
for h5_file in tqdm(h5_files, desc="Processing"):
    super_resolution_test(h5_file, dry_run=False)
```

## 🔗 相关资源

- **完整实现**: 参见notebook中的 `generate_planet_rgbnir.py` 代码块
- **模态定义**: `olmoearth_pretrain/data/constants.py` 中的 `PLANET_RGBNIR`
- **H5读取**: `olmoearth_pretrain/data/dataset.py` 中的 `read_h5_file` 方法
