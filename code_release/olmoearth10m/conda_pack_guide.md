# conda-pack 使用指南

## 概述

`conda-pack` 是一个用于将 conda 环境打包成归档文件（如 `.tar.gz`）的工具，方便将环境迁移到其他机器或集群上使用。它解决了在不同机器间复制 conda 环境时路径依赖的问题。

**当前版本**: 0.9.1

---

## 安装

```bash
# 通过 conda 安装
conda install conda-pack

# 通过 pip 安装
pip install conda-pack
```

---

## 基本用法

### 1. 打包环境

```bash
# 打包当前激活的环境
conda pack

# 打包指定名称的环境
conda pack -n my_env

# 打包指定路径的环境
conda pack -p /path/to/my_env

# 指定输出文件名
conda pack -n my_env -o my_env.tar.gz

# 使用更多线程加速打包
conda pack -n my_env -j 4
```

### 2. 解压并使用环境

在目标机器上：

```bash
# 创建目标目录
mkdir -p ~/miniconda3/envs/my_env

# 解压归档到该目录
tar -xzf my_env.tar.gz -C ~/miniconda3/envs/my_env

# 激活环境（需要先运行 conda-unpack 修正路径）
conda activate my_env
conda-unpack
```

> **注意**: `conda-unpack` 是打包时自动生成的脚本，用于修正硬编码的路径。每次激活环境后只需运行一次。

---

## 常用选项详解

| 选项 | 简写 | 说明 |
|------|------|------|
| `--name` | `-n` | 要打包的环境名称 |
| `--prefix` | `-p` | 要打包的环境路径 |
| `--output` | `-o` | 输出文件路径 |
| `--format` | | 归档格式：`tar.gz`(默认), `zip`, `tar.bz2`, `tar.zst`, `tar.xz`, `tar`, `squashfs` 等 |
| `--compress-level` | | 压缩级别 0-9（zstd 支持到 19），默认 4 |
| `--n-threads` | `-j` | 线程数，`-1` 表示使用所有 CPU |
| `--force` | `-f` | 覆盖已存在的输出文件 |
| `--quiet` | `-q` | 静默模式，不显示进度 |
| `--exclude` | | 排除匹配模式的文件（可多次使用） |
| `--include` | | 重新包含被排除的文件（可多次使用） |
| `--dest-prefix` | `-d` | 指定目标路径前缀（不生成 conda-unpack 脚本） |
| `--ignore-editable-packages` | | 忽略可编辑包检查 |
| `--ignore-missing-files` | | 忽略缺失文件检查 |

---

## 实际应用场景

### 场景 1：将环境迁移到无网络的集群

```bash
# 在能联网的机器上
conda create -n my_project python=3.10 pytorch torchvision -c pytorch
conda pack -n my_project -o my_project.tar.gz

# 将 my_project.tar.gz 传到目标集群
# 在目标集群上
mkdir -p ~/miniconda3/envs/my_project
tar -xzf my_project.tar.gz -C ~/miniconda3/envs/my_project
conda activate my_project
conda-unpack
```

### 场景 2：打包为 ZIP 格式（Windows 兼容）

```bash
conda pack -n my_env --format zip -o my_env.zip
```

### 场景 3：排除不需要的大文件

```bash
# 排除 .pyc 缓存文件和 __pycache__ 目录
conda pack -n my_env --exclude '*.pyc' --exclude '__pycache__' -o my_env.tar.gz
```

### 场景 4：指定目标路径前缀（用于已知的固定路径）

```bash
# 如果目标机器的环境路径已知且固定
conda pack -n my_env -d /opt/conda/envs/my_env -o my_env.tar.gz
# 使用 -d 后，解压后不需要运行 conda-unpack
```

### 场景 5：使用 SquashFS 格式（高性能只读挂载）

```bash
conda pack -n my_env --format squashfs -o my_env.squashfs

# 在目标机器上挂载使用
sudo mount -o loop my_env.squashfs /mnt/my_env
# 然后通过 --prefix 参数使用该环境
conda run --prefix /mnt/my_env python script.py
```

### 场景 6：在 Slurm 集群提交任务时使用

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.log

# 在作业脚本中解压并使用打包的环境
tar -xzf ${SLURM_TMPDIR}/my_env.tar.gz -C ${SLURM_TMPDIR}/my_env
source ${SLURM_TMPDIR}/my_env/bin/activate
conda-unpack

# 运行你的程序
python train.py
```

---

## 注意事项

### 1. 跨平台限制
- **Linux → Linux**: ✅ 完全支持
- **macOS → macOS**: ✅ 完全支持
- **Windows → Windows**: ✅ 支持（建议使用 ZIP 格式）
- **跨平台（如 Linux → macOS）**: ❌ **不支持**，因为二进制文件不兼容

### 2. 路径问题
- 打包的环境包含硬编码的绝对路径
- 解压后必须运行 `conda-unpack` 来修正路径（除非使用了 `-d` 参数）
- `conda-unpack` 会修正 `bin/` 目录下脚本中的 shebang 路径和 `.conda` 元数据中的路径

### 3. 大小优化
- 默认包含所有包文件，包括缓存和 `.pyc` 文件
- 可以使用 `--exclude` 排除不必要的文件来减小归档体积
- 考虑使用更高的压缩级别（如 `--compress-level 9`）来减小文件大小

### 4. 环境激活
- 解压后首次激活环境必须运行 `conda-unpack`
- 之后正常使用 `conda activate` 即可
- 如果环境被复制到新位置，需要重新运行 `conda-unpack`

### 5. 与 conda create --clone 的区别

| 特性 | `conda pack` | `conda create --clone` |
|------|-------------|----------------------|
| 需要网络 | ❌ 不需要 | ✅ 需要（下载包） |
| 跨机器迁移 | ✅ 适合 | ❌ 不适合 |
| 离线使用 | ✅ 支持 | ❌ 不支持 |
| 文件大小 | 压缩归档，较小 | 完整环境，较大 |
| 速度 | 快（打包+解压） | 慢（需重新下载） |

---

## 常见问题

### Q: 解压后 `conda activate` 报错 "Not a conda environment"
**A**: 确保解压路径正确，且目录结构包含 `bin/`, `lib/`, `conda-meta/` 等子目录。

### Q: 打包时提示 "No such file or directory" 但文件确实存在
**A**: 可能是符号链接问题，尝试使用 `--ignore-missing-files` 跳过检查。

### Q: 如何查看打包后的环境内容？
**A**: 可以使用 `tar -tzf my_env.tar.gz | head -20` 查看归档中的文件列表。

### Q: 打包的环境比原始环境大很多？
**A**: 检查是否包含了缓存文件，使用 `--exclude '*.pyc'` 等排除规则。

---

## 参考链接

- [conda-pack 官方文档](https://conda.github.io/conda-pack/)
- [conda-pack GitHub 仓库](https://github.com/conda/conda-pack)
