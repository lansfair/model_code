# Split H5 Dataset

将大尺寸 H5 数据集按空间维度拆分为多个小尺寸 H5 文件（真正的多进程并行处理）。

## 背景

OlmoEarth 预训练/推理使用 H5 格式的多模态地球观测数据，空间分辨率通常为 256×256。在某些场景下（如显存受限、小区域推理），需要将大图拆分为多个小图处理。

## 脚本

[scripts/jzf/split_h5.py](split_h5.py)

## 用法

### 基本用法

```bash
# 将 256×256 图像拆分为 4 个 128×128 的 H5 文件，并行处理
python scripts/jzf/split_h5.py \
    --src_dir data/dataset2 \
    --crop_size 128 \
    --output_dir data/dataset2_split
```

### 矩形裁剪

```bash
# 裁剪为 128×64 的矩形块
python scripts/jzf/split_h5.py \
    --src_dir data/dataset2 \
    --crop_size 128 64 \
    --output_dir data/dataset2_split
```

### 带重叠的裁剪

```bash
# 128×128 裁剪，步长 64（相邻块有 50% 重叠）
python scripts/jzf/split_h5.py \
    --src_dir data/dataset2 \
    --crop_size 128 \
    --stride 64 \
    --output_dir data/dataset2_split
```

### 过滤低质量切片

```bash
# 丢弃零值/缺失值超过 80% 的切片
python scripts/jzf/split_h5.py \
    --src_dir data/dataset2 \
    --crop_size 128 \
    --output_dir data/dataset2_split \
    --missing_threshold 0.8
```

### 筛选特定文件

```bash
python scripts/jzf/split_h5.py \
    --src_dir data/dataset2 \
    --crop_size 128 \
    --output_dir data/dataset2_split \
    --pattern "sample_0.h5"
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--src_dir` | str | 必填 | 源 H5 文件目录 |
| `--crop_size` | int [int] | 必填 | 逻辑空间（factor=1）的裁剪尺寸，单值为正方形，两值为 H×W |
| `--stride` | int [int] | 与 crop_size 相同 | 裁剪步长，设小值产生重叠块 |
| `--output_dir` | str | 必填 | 输出目录 |
| `--pattern` | str | `sample_*.h5` | 文件匹配 glob pattern |
| `--num_workers` | int | CPU 核心数 | 并行进程数 |
| `--missing_threshold` | float | `0.9` | 丢弃阈值：某模态切片中零值+缺失值占比超过此值则丢弃整个切片 |

## image_tile_size_factor 感知裁剪

`--crop_size` 以**逻辑（factor=1）像素**为单位指定。脚本自动根据每个模态的 `image_tile_size_factor` 计算实际裁剪像素数：

| 模态 | image_tile_size_factor | 逻辑 crop=128 时实际裁剪 |
|------|------------------------|--------------------------|
| sentinel2_l2a | 1 | 128×128 |
| srtm / worldcover | 1 | 128×128 |
| planet_rgbnir | 4 | 512×512 |
| naip_10 | 4 | 512×512 |

未在 `Modality` 类中注册的模态将按实际像素比例推断裁剪范围。

## 输出文件命名

每个输入文件产生 N 个输出文件，命名格式：

```
{原始文件名}_r{行号:02d}_c{列号:02d}.h5
```

例如 `sample_0.h5` 拆分为 2×2 网格时：
```
sample_0_r00_c00.h5    # 左上角
sample_0_r00_c01.h5    # 右上角
sample_0_r01_c00.h5    # 左下角
sample_0_r01_c01.h5    # 右下角
```

## 行为说明

### 空间数据

所有具有空间维度（H, W > 1）的 dataset 会根据裁剪坐标进行切片，并按各模态的 `image_tile_size_factor` 自动缩放坐标。`get_expected_tile_size() ≤ 1` 的模态（如 ERA5_10）视为非空间数据，直接复制。

### 非空间数据

以下数据直接复制，不做裁剪：
- `timestamps` — 时间戳，形状 (T, 3)
- `latlon` — 经纬度，形状 (2,)
- `missing_timesteps_masks` — 缺失时间步掩码组

### 质量过滤

每个切片写入前，对所有空间模态检查零值和 `MISSING_VALUE`（-99999）的占比。若任意模态超过 `--missing_threshold`，整个切片被丢弃（不写入磁盘）。最终统计会报告 written / skipped 数量。

### 边缘处理

当图像尺寸不能被步长整除时，最后一行/列的切片对齐到边缘（与前一个切片可能有重叠），确保全图覆盖无遗漏。步长大于裁剪尺寸时会打印警告（切片之间有间隙）。

### 压缩

输出文件保留源文件的压缩设置（zstd/lz4 等）。

## 示例：256 → 128 拆分

```
输入: data/dataset2/
  sample_0.h5  (256×256)
  sample_1.h5  (256×256)

输出: data/dataset2_split/
  sample_0_r00_c00.h5  (128×128)
  sample_0_r00_c01.h5  (128×128)
  sample_0_r01_c00.h5  (128×128)
  sample_0_r01_c01.h5  (128×128)
  sample_1_r00_c00.h5  (128×128)
  ...
```

含 planet_rgbnir（factor=4）的文件中，对应模态实际裁剪尺寸为 512×512。
