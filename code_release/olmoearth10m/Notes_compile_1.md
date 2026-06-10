# OLMoEarth 从 10m 训练体系改为原生 2m 训练：改哪些文件、原理、实施计划

## 0. 结论先说

你现在这套代码里，分辨率体系是围绕 `BASE_RESOLUTION=0.625` 和 `resolution_factor(int)` 建起来的。  
`2m / 0.625 = 3.2` 不是整数，所以**“原生2m”在当前实现里无法被精确表达**（尤其在 `//` 整除与 factor 推导链路上会出问题）。

所以要做原生 2m，建议走：

1. **方案A（推荐，真正原生2m）**：把分辨率基准改成 `0.5m`，让 `2m=4x` 变成整数体系。  
2. **方案B（过渡）**：不改基准，先用 `2.5m`（现成 `rgb2_5` / `rgb2_5_10`），快速验证训练链路。

你明确说要“原生2m”，下面重点给方案A。

---

## 1. 原理：为什么 10m -> 2m 不是只改一个数字

当前 OLMoEarth 的分辨率是三层耦合：

1. **数据网格层**（窗口/CSV/GeoTIFF命名）  
   `ModalitySpec.tile_resolution_factor` + `BASE_RESOLUTION` 决定 tile 的真实米级分辨率。

2. **像素尺寸层**（同地理范围下不同模态像素宽高）  
   `image_tile_size_factor` 决定同一地理窗口下该模态的像素宽高倍数（例如 `rgb2_5_10` 是 4 倍）。

3. **token/patch层**（token数、mask、pos encoding）  
   patch 实际像素尺度由 `patch_size_at_16 * image_tile_size_factor` 决定；  
   位置编码里的 GSD 比例由 `input_res * patch_size / BASE_GSD` 决定。

所以切到原生 2m，会同时影响：

- 数据生成窗口分组（`res_10` -> `res_2`）
- 模态规格表达（factor 必须可整除）
- H5 转换时 expected size 校验
- 训练 token 预算（2m token 会显著变多）
- 位置编码尺度（`BASE_GSD` 若仍是10，会有尺度偏移）

---

## 2. 关键文件：必须改 / 建议改

### 2.1 必改（方案A原生2m）

1. `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/constants.py`
- `BASE_RESOLUTION`: `0.625 -> 0.5`（核心改动）
- 新增原生2m模态定义（建议新名字，避免污染已有实验）：
  - `RGB2_NATIVE2M`（或你自定义）
  - `tile_resolution_factor=4`（2m）
  - `band_sets=[BandSet(["R","G","B"], 4)]`
  - `image_tile_size_factor=1`（原生2m，不再4倍扩图）
- `BASE_GSD`: 建议从 `10 -> 2`（让位置编码尺度以2m为基准）

2. `olmoearth_pretrain-main-zj/olmoearth_pretrain/datatypes.py`
- 给 `OlmoEarthSample` / `MaskedOlmoEarthSample` / `TokensAndMasks` 增加新模态字段（若你定义新模态名）。

3. `olmoearth_pretrain-main-zj/data/rslearn_dataset_configs/*.json`
- 复制 `config_rgb2_5_10.json` 改为 `config_rgb2_native2m.json`
- 每个 layer 的 `alias` 改新模态名
- `zoom_offset` 重新按 2m 数据源配置（不能沿用 2.5m 的 `zoom_offset=2`）

4. `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset_creation/rslearn_to_olmoearth/`
- 新建或复制 `rgb2_5_10.py` 为 `rgb2_native2m.py`
- `LAYER_*` 与 `Modality.*` 改为新模态
- `dataset.load_windows(... groups=["res_10"])` 改为 `groups=["res_2"]`

5. `olmoearth_pretrain-main-zj/docs/数据生成流程.md`
- 全流程命令改成 `--resolution 2` 和 `res_2` 组
- `run_h5_conversion` 的 `supported_modality_names` 加入新模态

6. `olmoearth_pretrain-main-zj/scripts/official/script.py`（及你实际启动脚本）
- `training_modalities` 用新 2m 模态（或只保留2m单模态做首轮）
- `token_budget/global_batch_size/sampled_hw_p_list/min_patch_size/max_patch_size` 按2m重调

### 2.2 必改（通常会被忽略）

7. `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/norm_configs/predefined.json`
8. `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/norm_configs/computed.json`
- 给新模态补每个 band 的归一化统计（R/G/B）

### 2.3 建议改（稳定性与可维护性）

9. `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset_creation/rslearn_to_olmoearth/multitemporal_raster.py`
- 当前大量 `//` 与“factor为整数”的隐含假设；如果未来还要支持非整倍数分辨率，需抽象成浮点比例+插值采样。

10. `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset/sample.py`
- 同样存在整除/幂次假设，建议加断言与错误提示，防止 silent mismatch。

---

## 3. 详细实施计划（可执行）

## Phase 1：分辨率体系改造（1-2天）

目标：让代码“表达层”能合法表示 2m。

1. 改 `BASE_RESOLUTION=0.5`，新增 2m 模态 spec。
2. 改 `BASE_GSD=2`（配合位置编码）。
3. datatypes 补齐新模态字段。
4. 运行最小单测/导入测试，确认 `Modality.get(new_name)`、shape 推导、mask 流程正常。

验收：
- 无 `resolution_factor` 整除报错；
- `get_expected_tile_size()` 与 H5 实际shape一致。

## Phase 2：数据链路打通（2-4天）

目标：拿到可训练的 2m H5。

1. 新建 `config_rgb2_native2m.json`，按2m数据源校准 `zoom_offset`。
2. 用 `create_windows` 生成 `--resolution 2` 窗口。
3. `prepare/ingest/materialize` 全链路使用 `--group res_2`。
4. 新建 `rgb2_native2m.py` 完成 rslearn -> olmoearth 转换。
5. 运行 `run_h5_conversion`，生成H5。
6. 补 `sample_metadata.csv` 对应列（新模态 presence）。

验收：
- 随机抽查 H5：`[H, W, T, 3]`、timestamps、missing mask 三者一致；
- 至少100样本能被 DataLoader 连续读取。

## Phase 3：训练配置重平衡（1-2天）

目标：2m 下训练不OOM、token统计合理。

1. 首轮只训练 `2m RGB`（去掉多模态）做 smoke test。
2. 调小 token 压力：
   - 降低 `sampled_hw_p_list` 上限；
   - 适度提高 `min_patch_size`；
   - 必要时降低 `global_batch_size`；
   - 视显存调整 `token_budget`。
3. 跑 `scripts/estimate_token_ratios.py` 估计 token 占比。

验收：
- 连续训练 >=10k steps 不崩；
- GPU利用率和吞吐可接受。

## Phase 4：多模态融合（可选，2-5天）

目标：2m 主模态 + 10m/20m 辅模态共训。

1. 明确融合策略：
   - 策略1：同地理范围，不同像素密度（推荐）
   - 策略2：把其他模态重采样到2m（成本高，不推荐首版）
2. 重新校准 masking 的 only_decode/encode 比例。
3. 重跑归一化统计与对齐抽检。

---

## 4. 参数建议（首版稳妥配置）

1. 首跑只开新 2m RGB 模态，不混其他模态。  
2. `global_batch_size` 先降到当前的 `1/4 ~ 1/8`。  
3. `sampled_hw_p_list` 先用 `range(1, 9)`，避免 token 激增。  
4. `min_patch_size` 可从 `2` 起试（不是1），稳定后再放开。  
5. 先固定 `token_budget`，观察 OOM 与 step time，再迭代。

---

## 5. 风险与回退

高风险点：

1. 改 `BASE_RESOLUTION` 会影响所有模态的 resolution 语义（全局影响）。  
2. 旧 checkpoint 的尺度先验与新 2m 体系不完全一致。  
3. 数据转换阶段最容易出现 silently wrong（shape对但地理不对）。

回退策略：

1. 新建分支，保留现有 10m 训练脚本不动。  
2. 新模态用新名字（不要覆盖 `rgb2_5_10`）。  
3. 先在“小数据+单模态”闭环后，再接入主训练。

---

## 6. 你现在就可以执行的最小任务清单

1. 在 `constants.py` 新增原生2m模态，并把 `BASE_RESOLUTION` 改到 `0.5`（单独分支）。  
2. 复制 `rgb2_5_10.py` -> `rgb2_native2m.py`，把 `res_10` 改为 `res_2`。  
3. 新建 `config_rgb2_native2m.json`，按2m源数据改 `zoom_offset`。  
4. 跑一批窗口并生成 100 个 H5 样本做 DataLoader 验证。  
5. 用单模态配置启动一次训练 smoke test（1-2k steps）。

---

## 7. 方案B（仅供过渡，不是原生2m）

如果你短期只要跑通训练而不追求“严格2m”：

1. 直接用现有 `rgb2_5` / `rgb2_5_10`。  
2. 不改 `BASE_RESOLUTION`。  
3. 先完成模型和损失验证，再回到方案A做原生2m。

这条路径改动小、成功率高，但不是你要的“原生2m”。

---

## 8. 从 2m 训练视角梳理信息流（含 Python 文件映射）

下面按两条链路梳理：

1. **离线数据构建链路**（GeoTIFF/rslearn -> H5）
2. **在线训练链路**（H5 -> token -> loss -> 反传）

### 8.1 离线数据构建链路（2m）

#### Stage A：窗口定义（地理采样）
- 输入：经纬度列表、`--resolution 2`、dataset config。  
- 处理：创建 rslearn windows（应落到 `res_2` 分组）。  
- 输出：`dataset/windows/res_2/...` 元数据。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset_creation/create_windows/from_lon_lat_list.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset_creation/create_windows/util.py`

#### Stage B：rslearn prepare/ingest/materialize
- 输入：windows + 各模态数据源配置（2m 模态需新 config）。  
- 处理：按 layer 下载/拼接/物化栅格。  
- 输出：每 window 的 raster 目录与 item 元数据。
- 关键文件（OLMoEarth侧转换会依赖这些产物）：
  - `olmoearth_pretrain-main-zj/data/rslearn_dataset_configs/config_rgb2_5_10.json`（你将复制为 2m 版本）

#### Stage C：rslearn -> OLMoEarth tile目录
- 输入：rslearn windows（`res_2`）  
- 处理：按月频/高频读取栅格，做 band-set 对齐，写出 OLMoEarth 目录结构。  
- 输出：`<olmoearth_path>` 下的按 modality/time_span 组织的 tif + csv。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset_creation/rslearn_to_olmoearth/rgb2_5_10.py`（2m时复制改名）
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset_creation/rslearn_to_olmoearth/multitemporal_raster.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset_creation/util.py`

#### Stage D：tile目录 -> H5 训练样本
- 输入：OLMoEarth tile目录、支持模态列表、压缩参数、`tile_size`。  
- 处理：`parse_dataset -> image_tiles_to_samples -> load_image_for_sample -> 写H5`。  
- 输出：`h5py_data_w_missing_timesteps...`、`sample_metadata.csv`、`latlon_distribution.npy`。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/internal/run_h5_conversion.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset/convert_to_h5py.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset/parse.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/dataset/sample.py`

---

### 8.2 在线训练链路（单个 step）

#### Stage 0：实验配置装配
- 输入：训练脚本 overrides（modalities、token_budget、patch范围、h5路径等）。  
- 处理：组装 model/dataset/dataloader/train_module/trainer 配置。  
- 输出：`OlmoEarthExperimentConfig`。
- 关键文件：
  - `olmoearth_pretrain-main-zj/scripts/official/base.py`
  - `olmoearth_pretrain-main-zj/scripts/official/script.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/internal/experiment.py`

#### Stage 1：Dataset 读取 + 归一化 + 子采样
- 输入：`sample_i.h5`。  
- 处理：
  - 按 `training_modalities` 读取 ndarray
  - 用 `predefined/computed` 归一化
  - 根据 `patch_size + sampled_hw_p + token_budget` 做时空裁剪（含 `image_tile_size_factor`）
- 输出：`(patch_size, OlmoEarthSample)`。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/dataset.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/normalize.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/norm_configs/predefined.json`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/norm_configs/computed.json`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/datatypes.py`

#### Stage 2：DataLoader 批处理 + transform + masking
- 输入：一批 `(patch_size, OlmoEarthSample)`。  
- 处理：
  - collate 成 batched sample
  - 应用 transform（flip/rotate/mixup/no_transform）
  - 应用 masking（通常输出双视图 A/B）
- 输出：`(patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b)`。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/dataloader.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/collate.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/transform.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/train/masking.py`

#### Stage 3：TrainModule 训练循环（microbatch）
- 输入：masked batch（A/B 两视图）。  
- 处理：
  - `split_masked_batch` 切 microbatches
  - 每个 microbatch 跑 `model_forward`
  - 聚合 base/contrastive/regularizer/mae loss
  - backward + optimizer step（在父类流程）
- 输出：参数更新、日志指标。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/train/train_module/contrastive_latentmim.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/train/train_module/train_module.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/train/utils.py`

#### Stage 4：模型前向（LatentMIM 主链）
- 输入：`MaskedOlmoEarthSample`, `patch_size`。  
- 处理：
  - online encoder：编码 masked 输入
  - decoder：预测需 decode 的 token
  - target encoder：编码 unmasked 输入作为 teacher target
  - （可选）reconstructor 做 MAE 重建
- 输出：`decoded`, `target_output`, `pooled embedding` 等。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/nn/latent_mim.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/nn/flexi_vit.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/nn/utils.py`

#### Stage 5：Patch/token 级编码细节（2m最相关）
- 输入：各模态张量 + mask + patch_size。  
- 处理：
  - `FlexiPatchEmbed` 按 `patch_size * image_tile_size_factor` patchify
  - 各 modality/bandset token 拼接
  - 加 channel/time/month/spatial encodings
  - spatial encoding 里通过 `calculate_gsd_ratio(input_res, patch_size)` 注入尺度
- 输出：Transformer 输入 token 序列。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/nn/flexi_patch_embed.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/nn/flexi_vit.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/nn/encodings.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/data/constants.py`

#### Stage 6：损失计算
- 输入：`decoded` 与 `target_output`（TokensAndMasks）。  
- 处理：按 mask 选取 decoder token，计算 patch discrimination / all discrimination / contrastive 等。  
- 输出：标量 loss（回传到 encoder+decoder，target encoder通常EMA更新）。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/train/loss.py`

#### Stage 7：目标编码器 EMA 与训练控制
- 输入：online encoder 参数、step/epoch 状态。  
- 处理：
  - `update_target_encoder()` 做 EMA 同步
  - trainer 负责 checkpoint / wandb / speed monitor / evaluator callback
- 输出：稳定 teacher 分支与训练状态持久化。
- 关键文件：
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/train/train_module/train_module.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/train/callbacks/*.py`
  - `olmoearth_pretrain-main-zj/olmoearth_pretrain/internal/experiment.py`

---

### 8.3 2m 训练排障时，优先检查的“信息流断点”

1. `windows group` 是否真是 `res_2`（不是 `res_10`）  
2. H5 中 2m 模态 shape 是否与 `ModalitySpec` 的 `image_tile_size_factor` 匹配  
3. `dataset.py` 中裁剪后 shape 与 `patch_size` 是否可整除  
4. `masking.py` 的 mask shape 是否和 tokenization/bandset 一致  
5. `flexi_vit.py` 中 `calculate_gsd_ratio` 输入分辨率是否与你的2m设定一致  
6. `loss.py` 是否出现“某模态 decoded token = 0”（会造成训练信号退化）
