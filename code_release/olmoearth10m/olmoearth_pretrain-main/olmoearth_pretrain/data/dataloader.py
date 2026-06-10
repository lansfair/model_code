"""OlmoEarth Pretrain DataLoader."""

import functools
import logging
import math
import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.utils import get_rng, memmap_to_write
from olmo_core.distributed.utils import (
    barrier,
    get_fs_local_rank,
    get_rank,
    get_world_size,
)
from olmo_core.utils import get_default_device
from torch.utils.data import default_collate
from upath import UPath

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.collate import (
    collate_double_masked_batched,
    collate_single_masked_batched,
)
from olmoearth_pretrain.data.concat import OlmoEarthConcatDataset
from olmoearth_pretrain.data.constants import IMAGE_TILE_SIZE, Modality
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthSample,
    subset_sample_default,
)
from olmoearth_pretrain.data.transform import Transform, TransformConfig
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig, MaskingStrategy

logger = logging.getLogger(__name__)


class OlmoEarthDataLoader(DataLoaderBase):
    """OlmoEarth Pretrain dataloader.

    This dataloader is adapted from OLMo-core's TextDataLoaderBase and NumpyDataLoaderBase,
    incorporating their core functionality for DDP, multi-threading, and multi-processing.
    
    【中文说明】为什么需要这个DataLoader?
    
    Dataset只负责返回单个样本,而DataLoader负责:
    1. **Batch构建**: 将多个样本组合成一个batch
    2. **分布式训练支持**: 在DDP环境中正确分配数据给不同GPU
    3. **并行加载**: 使用多进程加速数据读取
    4. **动态采样策略**: 每个batch随机选择patch_size和空间维度
    5. **Masking应用**: 在dataload时应用掩码策略(而非在模型中)
    6. **断点续训**: 保存/恢复训练进度(shuffle状态、已处理batch数)
    
    核心设计哲学:
    - Dataset: "给我第i个样本" → 返回单个OlmoEarthSample
    - DataLoader: "给我一个batch" → 返回(batch_size个样本的堆叠 + masking)
    """

    def __init__(
        self,
        dataset: OlmoEarthDataset | OlmoEarthConcatDataset,
        work_dir: UPath,
        global_batch_size: int,
        min_patch_size: int,
        max_patch_size: int,
        sampled_hw_p_list: list[int],
        token_budget: int | None = None,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        collator: Callable = default_collate,
        target_device_type: str = "cpu",
        drop_last: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn",
        num_dataset_repeats_per_epoch: int = 1,
        # Dataloader-side masking
        transform: Transform | None = None,
        masking_strategy: MaskingStrategy | None = None,
        masking_strategy_b: MaskingStrategy | None = None,
        num_masked_views: int = 1,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the OlmoEarthDataLoader.

        Args:
            dataset: The dataset to load from.
            work_dir: The working directory for storing indices.
            global_batch_size: The global batch size across all workers.
            min_patch_size: Minimum patch size for training.
            max_patch_size: Maximum patch size for training.
            sampled_hw_p_list: List of possible height/width in patches to sample.
            token_budget: Optional token budget per instance.
            dp_world_size: Data parallel world size.
            dp_rank: Data parallel rank.
            fs_local_rank: File system local rank.
            seed: Random seed.
            shuffle: Whether to shuffle the data.
            num_workers: Number of dataloader workers.
            prefetch_factor: Prefetch factor for dataloader.
            collator: Collation function.
            target_device_type: Target device type ("cpu" or "cuda").
            drop_last: Whether to drop the last incomplete batch.
            persistent_workers: Whether to keep workers alive between epochs.
            multiprocessing_context: Multiprocessing context ("spawn" or "forkserver").
            num_dataset_repeats_per_epoch: Number of times to repeat the dataset per epoch.
            transform: Optional transform to apply in the dataloader workers.
            masking_strategy: Masking strategy to apply in the dataloader workers.
            masking_strategy_b: Optional second masking strategy for Galileo-style training.
            num_masked_views: Number of masked views to return (1=single, 2=double).
            tokenization_config: Optional tokenization config for custom band groupings.
        
        【关键参数中文说明】
        
        **分布式训练相关**:
        - global_batch_size: 全局batch size(所有GPU加起来)
          * 例如: 8卡训练,每卡batch_size=4 → global_batch_size=32
        - dp_world_size/dp_rank: 数据并行世界大小和当前rank
        - fs_local_rank: 文件系统本地rank(用于缓存文件去重)
        
        **采样策略相关**:
        - min/max_patch_size: patch尺寸范围(如8-32像素)
          * 每个batch会随机选择一个patch_size
          * 小patch→更多token但更细粒度; 大patch→更少token但更粗粒度
        - sampled_hw_p_list: 可能的空间维度列表(以patch为单位)
          * 例如: [8, 16, 32] 表示可能采样8×8, 16×16, 32×32个patch
          * 实际像素数 = hw_p × patch_size
        - token_budget: 每个样本的token上限
          * 如果None,使用所有像素
          * 如果设置,会根据budget动态调整时间步数
        
        **性能优化相关**:
        - num_workers: DataLoader工作进程数
          * 0=主进程加载(慢); 4-8=推荐值
          * 太多会导致内存溢出
        - prefetch_factor: 预取因子
          * 每个worker预先加载prefetch_factor个batch
          * 默认=2×num_workers
        - persistent_workers: worker进程是否持久化
          * True=epoch间复用worker(避免反复创建进程)
        
        **Masking相关**:
        - masking_strategy: 掩码策略(如随机mask 40%的token)
        - num_masked_views: 返回几个masked版本
          * 1=单视图(标准MAE风格)
          * 2=双视图(Galileo风格,两个不同的mask)
        - transform: 可选的数据变换(如颜色抖动)
        
        **其他**:
        - num_dataset_repeats_per_epoch: 每个epoch重复数据集几次
          * 1=标准; >1=增加epoch内样本数(相当于减小epoch长度)
        """
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        self.dataset = dataset
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        if token_budget is None:
            logger.warning("No token budget provided ALL PIXELS WILL BE USED")
        self.token_budget = token_budget
        self.patch_sizes = np.arange(min_patch_size, max_patch_size + 1)
        self.sampled_hw_p_list = sampled_hw_p_list
        self.collator = collator
        self.seed = seed
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type
        self.drop_last = drop_last
        self._global_indices: np.ndarray | None = None
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.num_dataset_repeats_per_epoch = num_dataset_repeats_per_epoch

        # Dataloader-side masking configuration
        self.transform = transform
        self.masking_strategy = masking_strategy
        self.masking_strategy_b = masking_strategy_b
        self.num_masked_views = num_masked_views
        self.tokenization_config = tokenization_config

        # Validate configuration
        if masking_strategy is None:
            raise ValueError("masking_strategy must be provided")
        if num_masked_views not in (1, 2):
            raise ValueError(f"num_masked_views must be 1 or 2, got {num_masked_views}")

        if self.num_workers > 0 and self.multiprocessing_context == "forkserver":
            # Overhead of loading modules on import by preloading them
            mp.set_forkserver_preload(["torch", "rasterio"])

    @property
    def total_unique_batches(self) -> int:
        """The total number of unique batches in an epoch.
        
        【中文说明】一个epoch中的唯一batch总数。
        计算公式: len(dataset) // global_batch_size
        注意: 这是"唯一"batch数,不考虑num_dataset_repeats_per_epoch
        """
        return len(self.dataset) // (self.global_batch_size)

    @property
    def total_unique_size(self) -> int:
        """The total number of unique instances in an epoch.
        
        【中文说明】一个epoch中的唯一样本总数。
        计算公式: total_unique_batches × global_batch_size
        """
        return self.total_unique_batches * self.global_batch_size

    @property
    def total_batches(self) -> int:
        """The total number of batches in an epoch.
        
        【中文说明】一个epoch中的实际batch总数(含重复)。
        计算公式: total_unique_batches × num_dataset_repeats_per_epoch
        
        示例: 
        - 1000个样本, global_batch_size=32 → 31个唯一batch
        - num_dataset_repeats_per_epoch=2 → 62个总batch
        """
        return self.total_unique_batches * self.num_dataset_repeats_per_epoch

    @property
    def total_size(self) -> int:
        """The total number of instances in an epoch.
        
        【中文说明】一个epoch中的实际样本总数(含重复)。
        计算公式: total_batches × global_batch_size
        """
        return self.total_batches * self.global_batch_size

    @property
    def _global_indices_file(self) -> UPath:
        """Global indices file."""
        global_indices_fname = self._format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,  # type: ignore
            size=self.total_size,
        )
        return (
            Path(self.work_dir)
            / f"dataset-{self.dataset.fingerprint}"
            / f"{global_indices_fname}.npy"
        )

    def _build_global_indices(self) -> np.ndarray:
        """Build global indices.
        
        【中文说明】构建全局索引数组。
        
        目的: 确定一个epoch中所有样本的访问顺序。
        
        工作流程:
        1. 创建基础索引 [0, 1, 2, ..., len(dataset)-1]
        2. 如果shuffle=True,根据(seed + epoch)随机打乱
           * 每个epoch的shuffle结果不同,但可复现
        3. 裁剪到total_unique_size(去除尾部不完整batch)
        4. 如果num_dataset_repeats_per_epoch > 1,重复多次
        
        为什么需要这个?
        - 分布式训练中,所有GPU需要知道全局的数据顺序
        - 每个GPU根据自己的dp_rank从global_indices中取子集
        - 确保不同GPU处理不同的数据,不重复不遗漏
        
        示例:
        - dataset有1000个样本, global_batch_size=32
        - total_unique_size = 31 × 32 = 992
        - global_indices形状: (992,) 或 (992×repeats,)
        """
        assert len(self.dataset) < np.iinfo(np.uint32).max

        rng: np.random.Generator | None = None
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = get_rng(self.seed + self.epoch)  # type: ignore
        indices_list = []
        for _ in range(self.num_dataset_repeats_per_epoch):
            indices = np.arange(len(self.dataset), dtype=np.uint32)
            if rng is not None:
                rng.shuffle(indices)
            # Remove tail of data to make it evenly divisible
            cropped_indices = indices[: self.total_unique_size]
            indices_list.append(cropped_indices)
        indices = np.concatenate(indices_list)
        return indices

    def build_and_save_global_indices(self, in_memory: bool = False) -> None:
        """Build and save global indices.
        
        【中文说明】构建并保存全局索引。
        
        两种模式:
        
        1. **内存模式** (in_memory=True):
           - 直接将indices保存在self._global_indices中
           - 适合小数据集,避免磁盘I/O
        
        2. **文件模式** (in_memory=False, 默认):
           - 将indices保存到.npy文件
           - 只有fs_local_rank=0的进程执行保存(避免并发写入)
           - 其他进程等待barrier()同步
           - 后续epoch可直接加载文件,无需重新计算
        
        为什么需要文件缓存?
        - shuffle后的索引计算有一定开销
        - 断点续训时,需要恢复相同的索引顺序
        - 多进程环境中,文件是共享状态的可靠方式
        
        文件命名规则:
        global_indices_seed{seed}_epoch{epoch}_size{size}.npy
        """
        if in_memory:
            self._global_indices = self._build_global_indices()
        else:
            self._global_indices = None
            if self.fs_local_rank == 0:
                # Either load from file or build and save to file
                if self._global_indices_file.is_file():
                    logger.info(
                        f"Using existing global indices file for seed {self.seed} and epoch {self.epoch}"  # type: ignore
                        f"at:\n'{self._global_indices_file}'"
                    )
                else:
                    global_indices = self._build_global_indices()
                    assert (
                        len(global_indices) < np.iinfo(np.int32).max
                    )  # Note: OLMo uses uint32
                    with memmap_to_write(
                        self._global_indices_file,
                        shape=global_indices.shape,
                        dtype=np.int32,
                    ) as global_indices_mmap:
                        global_indices_mmap[:] = global_indices
                    logger.info(
                        f"Global data order indices saved to:\n'{self._global_indices_file}'"
                    )
        barrier()

    def reshuffle(self, epoch: int | None = None, in_memory: bool = False) -> None:
        """Reshuffle the data."""
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1  # type: ignore
        if epoch <= 0:
            raise ValueError(f"'epoch' must be at least 1, got {epoch}")
        self._epoch = epoch
        # Since epoch has been updated, we need to create new global indices
        self.build_and_save_global_indices(in_memory=in_memory)

    def get_global_indices(self) -> np.ndarray:
        """Get global indices."""
        # Either load from memory or file
        if self._global_indices is not None:
            return self._global_indices
        if not self._global_indices_file.is_file():
            raise RuntimeError(
                f"Missing global indices file {self._global_indices_file}, did you forget to call 'reshuffle()'?"
            )
        return np.memmap(self._global_indices_file, mode="r", dtype=np.uint32)

    def _iter_batches(self) -> Iterable[OlmoEarthSample]:
        """Iterate over the dataset in batches."""
        return torch.utils.data.DataLoader(
            _IterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            multiprocessing_context=(
                self.multiprocessing_context if self.num_workers > 0 else None
            ),
            timeout=0,
        )

    @property
    def worker_info(self):  # type: ignore
        """Get worker info."""
        return torch.utils.data.get_worker_info()

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        """Get local instance indices.
        
        【中文说明】从全局索引中提取当前进程/worker负责的局部索引。
        
        这是分布式数据并行的核心逻辑,分三步过滤:
        
        1️⃣ **Batch维度切分**: 
           - 将全局索引reshape为 (num_batches, global_batch_size)
           - 如果已处理过一些batch,跳过前面的部分(断点续训)
        
        2️⃣ **Worker维度切分**:
           - 如果有多个DataLoader workers(num_workers > 0)
           - 每个worker只取自己负责的部分: indices[worker_id::num_workers]
           - 例如: 4个workers → worker 0取[0,4,8,...], worker 1取[1,5,9,...]
        
        3️⃣ **DP Rank维度切分**:
           - 在分布式训练中,每个GPU rank只取自己的部分
           - indices[:, dp_rank::dp_world_size]
           - 例如: 8卡训练 → rank 0取第0列, rank 1取第1列, ...
        
        最终结果: 一个一维数组,包含当前worker在当前rank需要处理的所有样本索引
        
        为什么这么复杂?
        - 确保8卡×4 workers = 32个并行单元,每个处理完全不同的数据
        - 避免重复处理和遗漏
        - 支持断点续训(从任意batch继续)
        """
        # NOTE:'indices' are global instance indices.
        instances_per_batch = self.global_batch_size
        indices = indices.reshape(-1, instances_per_batch)

        if self.batches_processed > 0:  # type: ignore
            indices = indices[self.batches_processed :]  # type: ignore

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        # Finally step batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))
        return indices

    def _get_dataset_item(
        self, idx: int, patch_size: int, sampled_hw_p: int
    ) -> tuple[int, OlmoEarthSample]:
        """Get a dataset item."""
        args = GetItemArgs(
            idx=idx,
            patch_size=patch_size,
            sampled_hw_p=sampled_hw_p,
            token_budget=self.token_budget,
            tokenization_config=self.tokenization_config,
        )
        item = self.dataset[args]
        return item

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict.
        
        【中文说明】保存DataLoader的状态,用于断点续训。
        
        保存的关键信息:
        - dataset_fingerprint: 数据集指纹(验证数据是否变化)
        - batches_processed: 已处理的batch数(从哪继续)
        - seed: 随机种子(保证shuffle可复现)
        - epoch: 当前epoch数
        
        为什么需要保存这些?
        - 训练中断后,可以从exact位置恢复
        - 确保resume后的数据顺序与中断前一致
        - 如果数据集变化了,会发出警告
        """
        return {
            "dataset_fingerprint_version": self.dataset.fingerprint_version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "batches_processed": self.batches_processed,  # type: ignore
            "seed": self.seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict.
        
        【中文说明】从checkpoint恢复DataLoader状态。
        
        恢复流程:
        1. 验证数据集指纹是否匹配
           - 如果不匹配,发出警告(数据可能已变化)
        2. 如果seed不同,使用checkpoint中的seed
           - 保证shuffle顺序一致
        3. 恢复batches_processed和epoch
           - 从exact位置继续训练
        
        注意事项:
        - 如果更换了数据集但加载了旧checkpoint,会警告
        - seed不一致时,优先使用checkpoint的seed保证可复现性
        """
        if (
            state_dict["dataset_fingerprint_version"]
            != self.dataset.fingerprint_version
        ):
            logger.warning(
                "Dataset fingerprint version does not match the version in the checkpoint, "
                "this could mean the data has changed"
            )
        elif state_dict["dataset_fingerprint"] != self.dataset.fingerprint:
            logger.warning(
                "Restoring state from a different dataset! If this is not expected, please check the dataset fingerprint(fingerprint doesn't match)"
                f"old fingerprint: {state_dict['dataset_fingerprint']}, new fingerprint: {self.dataset.fingerprint}"
            )

        if state_dict["seed"] != self.seed:
            logger.warning(
                "Restoring data loading state with a different data seed, "
                "will use data seed from state dict for data order consistency."
            )
            self.seed = state_dict["seed"]

        self.batches_processed = state_dict["batches_processed"]
        self._epoch = state_dict["epoch"] or self._epoch  # type: ignore

    def _format_fname_from_fields(self, prefix: str, **fields: Any) -> str:
        parts = [prefix]
        for key in sorted(fields):
            value = fields[key]
            if value is not None:
                parts.append(f"{key}{value}")
        return "_".join(parts)

    def _get_mock_sample(self, rng: np.random.Generator) -> OlmoEarthSample:
        output_dict = {}
        standard_hw = 64
        if Modality.SENTINEL2_L2A.name in self.dataset.training_modalities:
            mock_sentinel2_l2a = rng.random(
                (standard_hw, standard_hw, 12, 12), dtype=np.float32
            )
            output_dict["sentinel2_l2a"] = mock_sentinel2_l2a
        if Modality.NAIP_10.name in self.dataset.training_modalities:
            mock_naip_10 = rng.random((1024, 1024, 1, 4), dtype=np.float32)
            output_dict["naip_10"] = mock_naip_10
        if Modality.SENTINEL1.name in self.dataset.training_modalities:
            mock_sentinel1 = rng.random(
                (standard_hw, standard_hw, 12, 2), dtype=np.float32
            )
            output_dict[Modality.SENTINEL1.name] = mock_sentinel1
        if Modality.WORLDCOVER.name in self.dataset.training_modalities:
            mock_worldcover = rng.random(
                (standard_hw, standard_hw, 1, 1), dtype=np.float32
            )
            output_dict["worldcover"] = mock_worldcover
        if Modality.LATLON.name in self.dataset.training_modalities:
            mock_latlon = rng.random((2,), dtype=np.float32)
            output_dict["latlon"] = mock_latlon
        if Modality.OPENSTREETMAP_RASTER.name in self.dataset.training_modalities:
            mock_openstreetmap_raster = rng.random(
                (standard_hw, standard_hw, 1, 30), dtype=np.float32
            )
            output_dict["openstreetmap_raster"] = mock_openstreetmap_raster
        if Modality.SRTM.name in self.dataset.training_modalities:
            mock_srtm = rng.random((standard_hw, standard_hw, 1, 1), dtype=np.float32)
            output_dict["srtm"] = mock_srtm
        if Modality.LANDSAT.name in self.dataset.training_modalities:
            mock_landsat = rng.random(
                (standard_hw, standard_hw, 12, Modality.LANDSAT.num_bands),
                dtype=np.float32,
            )
            output_dict["landsat"] = mock_landsat
        if Modality.GSE.name in self.dataset.training_modalities:
            mock_gse = rng.random(
                (standard_hw, standard_hw, 1, Modality.GSE.num_bands), dtype=np.float32
            )
            output_dict["gse"] = mock_gse
        if Modality.CDL.name in self.dataset.training_modalities:
            mock_cdl = rng.random(
                (standard_hw, standard_hw, 1, Modality.CDL.num_bands), dtype=np.float32
            )
            output_dict["cdl"] = mock_cdl
        if Modality.WORLDPOP.name in self.dataset.training_modalities:
            mock_worldpop = rng.random(
                (standard_hw, standard_hw, 1, Modality.WORLDPOP.num_bands),
                dtype=np.float32,
            )
            output_dict["worldpop"] = mock_worldpop
        if Modality.WRI_CANOPY_HEIGHT_MAP.name in self.dataset.training_modalities:
            mock_wri_canopy_height_map = rng.random(
                (standard_hw, standard_hw, 1, Modality.WRI_CANOPY_HEIGHT_MAP.num_bands),
                dtype=np.float32,
            )
            output_dict["wri_canopy_height_map"] = mock_wri_canopy_height_map
        if Modality.ERA5_10.name in self.dataset.training_modalities:
            mock_era5_10 = rng.random(
                (12, Modality.ERA5_10.num_bands), dtype=np.float32
            )
            output_dict["era5_10"] = mock_era5_10
        if Modality.EUROCROPS.name in self.dataset.training_modalities:
            mock_eurocrops = rng.random(
                (standard_hw, standard_hw, 1, Modality.EUROCROPS.num_bands),
                dtype=np.float32,
            )
            output_dict["eurocrops"] = mock_eurocrops
        # --- 新增部分开始 ---
        if Modality.RGB.name in self.dataset.training_modalities:
            # RGB is multitemporal (Time=12) and has 3 bands (R, G, B)
            # Shape: [H, W, T, C]
            mock_rgb = rng.random(
                (standard_hw*4, standard_hw*4, 12, 4), dtype=np.float32
            )
            output_dict[Modality.RGB.name] = mock_rgb
        if Modality.SAR.name in self.dataset.training_modalities:
            # RGB is multitemporal (Time=12) and has 3 bands (R, G, B)
            # Shape: [H, W, T, C]
            mock_sar = rng.random(
                (standard_hw*4, standard_hw*4, 12, 1), dtype=np.float32
            )
            output_dict[Modality.SAR.name] = mock_sar
        if Modality.LANDCOVER_1M.name in self.dataset.training_modalities:
            mock_landcover_1m = rng.random(
                (standard_hw, standard_hw, 1, 1), dtype=np.float32
            )
            output_dict[Modality.LANDCOVER_1M.name] = mock_landcover_1m
        if Modality.LANDCOVER_30M.name in self.dataset.training_modalities:
            mock_landcover_30m = rng.random(
                (standard_hw, standard_hw, 1, 1), dtype=np.float32
            )
            output_dict[Modality.LANDCOVER_30M.name] = mock_landcover_30m
        if Modality.LT1.name in self.dataset.training_modalities:
            mock_lt1 = rng.random(
                (standard_hw*4, standard_hw*4, 12, 1), dtype=np.float32
            )
            output_dict[Modality.LT1.name] = mock_lt1
        # --- 新增部分结束 ---

        days = rng.integers(0, 25, (12, 1))
        months = rng.integers(0, 12, (12, 1))
        years = rng.integers(2018, 2020, (12, 1))
        timestamps = np.concatenate([days, months, years], axis=1)  # shape: (12, 3)

        output_dict["timestamps"] = timestamps
        return OlmoEarthSample(**output_dict)

    def get_mock_batch(self) -> Any:
        """Get a mock batch, for dry-run of forward and backward pass.

        Returns the appropriate batch format based on num_masked_views:
        - 1: (patch_size, MaskedOlmoEarthSample) - single masked view
        - 2: (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample) - double masked
        
        【中文说明】生成mock batch,用于模型测试(不读取真实数据)。
        
        使用场景:
        - 验证模型forward/backward是否正常
        - 调试时快速检查batch形状
        - 不需要等待真实数据加载
        
        工作流程:
        1. 为每个训练模态生成随机数据(形状符合规范)
        2. 调用subset_sample_default进行采样
        3. 通过collator组装成batch(包含masking)
        
        注意: 
        - 这是合成数据,不包含真实地理信息
        - 仅用于验证代码逻辑,不能用于实际训练
        """
        logger.info("Getting mock batch NOT FROM DATASET")
        logger.info(f"Training modalities: {self.dataset.training_modalities}")
        logger.info(f"num_masked_views: {self.num_masked_views}")
        rng = get_rng(42)
        batch_size = self.global_batch_size // self.dp_world_size
        patch_size = 1

        # Generate mock samples
        mock_samples = [
            subset_sample_default(
                self._get_mock_sample(rng),
                patch_size=patch_size,
                max_tokens_per_instance=1500,
                sampled_hw_p=6,
                current_length=12,
            )
            for _ in range(batch_size)
        ]

        # Pass raw samples to the collator - the batched collators handle
        # transform + masking internally when num_masked_views > 0
        collated_sample = self.collator(
            [(patch_size, sample) for sample in mock_samples]
        )

        return collated_sample

    def fast_forward(self, global_step: int) -> np.ndarray:
        """Fast forward the data loader to a specific global step and return the batch_indices."""
        logger.warning(
            "Fast forward does not yet support returning to indices for multiple GPUs"
        )
        if get_world_size() > 1:
            raise NotImplementedError("Fast forward is not supported in DDP")
        # If the model was trained with multiple GPUS, this logic must be updated so that we grab from where all the ranks started
        self.batches_processed = global_step
        epoch = math.ceil(global_step / self.total_batches)
        step_in_epoch = global_step % self.total_batches
        logger.info(f"epoch: {epoch}, step in epoch: {step_in_epoch}")
        self.reshuffle(epoch=epoch)
        batch_start = int(self.get_global_indices()[step_in_epoch])
        batch_end = batch_start + self.global_batch_size
        sample_indices = np.arange(batch_start, batch_end)
        return sample_indices


def iter_batched(
    iterable: Iterable[tuple[int, OlmoEarthSample]],
    batch_size: int,
    drop_last: bool = True,
) -> Iterable[tuple[tuple[int, OlmoEarthSample], ...]]:
    """Iterate over the dataset in batches.

    This is a modified version of olmo_core.data.data_loader.iter_batched that creates batches
    of size local_batch_size for the local rank from an iterator of items.


    Args:
        iterable: The iterator of items to batch.
        batch_size: The size of the batches to create for the local rank.
        drop_last: Whether to drop the last batch if it's not full.

    Returns:
        An iterator of batches of items.
    """
    assert batch_size > 0
    batch: list[tuple[int, OlmoEarthSample]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()

    # If there's a partial batch left over, yield it if `drop_last` is False
    if not drop_last and batch:
        yield tuple(batch)


class _IterableDatasetWrapper(torch.utils.data.IterableDataset[OlmoEarthSample]):
    """Iterable dataset wrapper.

    This is a modified version of olmo_core.data.data_loader._IterableDatasetWrapper
    
    【中文说明】可迭代数据集包装器。
    
    为什么需要这个wrapper?
    
    PyTorch的DataLoader要求dataset实现__iter__()方法才能使用多进程加载。
    但OlmoEarthDataset是map-style(通过索引访问),不是iterable-style。
    
    这个wrapper的作用:
    1. 将map-style dataset转换为iterable-style
    2. 在每个worker进程中独立管理随机状态(rngs)
    3. 动态生成(idx, patch_size, sampled_hw_p)三元组
    4. 调用dataset.__getitem__获取样本
    5. 按batch_size分组后yield
    
    关键设计:
    - 每个worker有独立的rng,保证不同worker的采样不重复
    - patch_size和sampled_hw_p每rank_batch_size个样本更换一次
    - 这样同一个batch内的样本有相同的空间维度,可以stack
    """

    def __init__(self, data_loader: OlmoEarthDataLoader):
        """Initialize the IterableDatasetWrapper."""
        self.data_loader = data_loader
        workers = data_loader.num_workers or 1
        self.rngs = [
            get_rng(
                data_loader.seed + data_loader.epoch + data_loader.dp_rank * workers + i
            )
            for i in range(workers)
        ]
        # Dataloader-side masking configuration
        self.transform = data_loader.transform
        self.masking_strategy = data_loader.masking_strategy
        self.masking_strategy_b = data_loader.masking_strategy_b
        self.num_masked_views = data_loader.num_masked_views

    def _get_batch_item_params_iterator(
        self,
        indices: np.ndarray,
        patch_size_list: list[int],
        hw_p_to_sample: list[int],
        rank_batch_size: int,
    ) -> Iterator[tuple[int, int, int]]:
        """Get a generator that yields a tuple of (idx, patch_size, sampled_hw_p).

        Changes patch_size and sampled_hw_p every rank_batch_size.
        
        【中文说明】生成(idx, patch_size, sampled_hw_p)三元组的迭代器。
        
        这是DataLoader动态采样策略的核心!
        
        工作流程:
        1. 遍历当前worker负责的所有样本索引
        2. 每rank_batch_size个样本,随机选择一次:
           - patch_size: 从[min_patch_size, max_patch_size]中随机选
           - sampled_hw_p: 从sampled_hw_p_list中过滤出合法的,然后随机选
        3. yield (idx, patch_size, sampled_hw_p)
        
        为什么每batch更换patch_size?
        - 增加数据多样性: 模型看到不同粒度的特征
        - 小patch→细粒度但token多; 大patch→粗粒度但token少
        - 类似multi-scale training
        
        合法性检查:
        - IMAGE_TILE_SIZE / patch_size = 最大可能的hw_p
        - 只选择 <= 这个值的hw_p,避免超出图像边界
        
        示例:
        - patch_size_list = [8, 16, 32]
        - sampled_hw_p_list = [8, 16, 32, 64]
        - 如果选中patch_size=16, IMAGE_TILE_SIZE=256
        - 最大hw_p = 256/16 = 16
        - 合法hw_p = [8, 16](排除32, 64)
        """
        patch_size_array = np.array(patch_size_list)
        hw_p_to_sample_array = np.array(hw_p_to_sample)
        instances_processed = 0

        # TODO: We need to maintain state and reproducibility here
        worker_id = self.worker_info.id if self.worker_info is not None else 0
        rng = self.rngs[worker_id]

        for idx in indices:
            if instances_processed % rank_batch_size == 0:
                patch_size = rng.choice(patch_size_array)
                max_height_width_tokens = int(IMAGE_TILE_SIZE / patch_size)
                filtered_hw_p_to_sample_array = hw_p_to_sample_array[
                    hw_p_to_sample_array <= max_height_width_tokens
                ]
                filtered_hw_p_to_sample_array = filtered_hw_p_to_sample_array[
                    filtered_hw_p_to_sample_array > 0
                ]
                sampled_hw_p = rng.choice(filtered_hw_p_to_sample_array)
            yield idx, int(patch_size), int(sampled_hw_p)
            instances_processed += 1

    @property
    def dataset(self) -> OlmoEarthDataset:
        """Get the dataset."""
        return self.data_loader.dataset

    @property
    def worker_info(self):  # type: ignore
        """Get worker info."""
        return torch.utils.data.get_worker_info()

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset.

        Yields batches in one of two formats depending on num_masked_views:
        - 1: (patch_size, MaskedOlmoEarthSample) - single masked view
        - 2: (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample) - double masked views

        Transform and masking are applied in the batched collator for better vectorization.
        
        【中文说明】数据加载的主迭代器。
        
        这是整个DataLoader的核心执行流程!
        
        完整工作流程:
        
        1️⃣ **获取全局索引**: 
           - 从文件/内存加载global_indices
           - 包含一个epoch中所有样本的访问顺序
        
        2️⃣ **提取局部索引**:
           - 调用_get_local_instance_indices
           - 根据dp_rank和worker_id过滤出当前进程负责的索引
        
        3️⃣ **创建样本迭代器**:
           - _get_batch_item_params_iterator生成(idx, patch_size, sampled_hw_p)
           - _get_dataset_item根据这些参数调用dataset.__getitem__
           - 返回单个OlmoEarthSample
        
        4️⃣ **批量组装**:
           - iter_batched将样本按rank_batch_size分组
           - collator对每个batch应用transform和masking
           - 返回最终的batch
        
        关键设计:
        - Generator表达式实现懒加载,不一次性加载所有数据
        - Collator在batch级别应用masking,比逐个样本更高效
        - 支持单视图(num_masked_views=1)和双视图(num_masked_views=2)
        
        返回值格式:
        - num_masked_views=1: (patch_size, MaskedOlmoEarthSample)
        - num_masked_views=2: (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample)
        """
        global_indices = self.data_loader.get_global_indices()
        indices = self.data_loader._get_local_instance_indices(global_indices)

        # Create iterator that fetches samples from the dataset
        instance_iterator = (
            self.data_loader._get_dataset_item(int(idx), patch_size, sampled_hw_p)
            for idx, patch_size, sampled_hw_p in self._get_batch_item_params_iterator(
                indices,
                self.data_loader.patch_sizes,
                self.data_loader.sampled_hw_p_list,
                self.data_loader.rank_batch_size,
            )
        )

        return (
            self.data_loader.collator(batch)  # type: ignore[arg-type]
            for batch in iter_batched(
                instance_iterator,  # type: ignore[arg-type]
                self.data_loader.rank_batch_size,
                self.data_loader.drop_last,
            )
        )


@dataclass
class OlmoEarthDataLoaderConfig(Config):
    """Configuration for the OlmoEarthDataLoader.
    
    【中文说明】DataLoader配置类。
    
    为什么需要单独的配置类?
    
    1. **声明式配置**: 
       - 通过YAML/JSON配置文件定义所有参数
       - 避免在代码中硬编码
    
    2. **自动验证**:
       - validate()方法确保配置合法性
       - 提前发现错误(如min_patch_size > max_patch_size)
    
    3. **构建器模式**:
       - build()方法负责创建DataLoader实例
       - 自动处理依赖(如构建masking_strategy、collator)
    
    4. **类型安全**:
       - dataclass提供自动的类型检查
       - IDE可以自动补全和提示
    
    使用示例:
        config = OlmoEarthDataLoaderConfig(
            work_dir="/path/to/work",
            global_batch_size=32,
            min_patch_size=8,
            max_patch_size=32,
            sampled_hw_p_list=[8, 16, 32],
            masking_config=MaskingConfig(...),
            num_masked_views=1
        )
        dataloader = config.build(dataset)
    """
    work_dir: str
    global_batch_size: int
    min_patch_size: int
    max_patch_size: int
    sampled_hw_p_list: list[int]
    seed: int
    token_budget: int | None = None  # No subsetting if None
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int | None = None
    target_device_type: str | None = None
    drop_last: bool = True
    num_dataset_repeats_per_epoch: int = 1
    # New fields for dataloader-side masking
    transform_config: TransformConfig | None = None
    masking_config: MaskingConfig | None = None
    masking_config_b: MaskingConfig | None = None
    num_masked_views: int = 1  # 1 = single, 2 = double
    tokenization_config: TokenizationConfig | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if self.work_dir is None:
            raise ValueError("Work directory is not set")
        if self.min_patch_size > self.max_patch_size:
            raise ValueError("min_patch_size must be less than max_patch_size")
        if self.masking_config is None:
            raise ValueError("masking_config must be provided")
        if self.num_masked_views not in (1, 2):
            raise ValueError(
                f"num_masked_views must be 1 or 2, got {self.num_masked_views}"
            )

    @property
    def work_dir_upath(self) -> UPath:
        """Get the work directory."""
        return UPath(self.work_dir)

    def build(
        self,
        dataset: OlmoEarthDataset,
        dp_process_group: dist.ProcessGroup | None = None,
    ) -> "OlmoEarthDataLoader":
        """Build the OlmoEarthDataLoader.
        
        【中文说明】根据配置构建DataLoader实例。
        
        构建流程:
        
        1️⃣ **验证配置**: 调用validate()确保参数合法
        
        2️⃣ **准备数据集**: 调用dataset.prepare()
           - 过滤样本索引
           - 加载地理分布信息
        
        3️⃣ **构建Masking策略**:
           - 从masking_config创建masking_strategy对象
           - 如果num_masked_views=2,还需要masking_strategy_b
        
        4️⃣ **选择Collator**:
           - num_masked_views=1 → collate_single_masked_batched
           - num_masked_views=2 → collate_double_masked_batched
           - 使用functools.partial绑定transform和masking策略
        
        5️⃣ **创建DataLoader**: 
           - 传递所有参数给OlmoEarthDataLoader.__init__
           - 自动获取分布式环境信息(dp_world_size, dp_rank等)
        
        为什么collator要在这里选择?
        - Collator需要知道masking策略才能正确工作
        - 不同的num_masked_views需要不同的collator
        - 使用partial提前绑定参数,避免每次调用都传递
        
        返回值:
        - 完全配置好的OlmoEarthDataLoader实例,可以直接用于训练
        """
        self.validate()
        dataset.prepare()

        # Build transform and masking strategies
        transform = (
            self.transform_config.build() if self.transform_config is not None else None
        )
        # masking_config is required (validated above)
        assert self.masking_config is not None
        masking_strategy = self.masking_config.build()
        masking_strategy_b = (
            self.masking_config_b.build() if self.masking_config_b is not None else None
        )

        # Select appropriate collator based on num_masked_views
        # Use batched collators that apply transform + masking to the entire batch
        # at once for better vectorization
        collator: Callable
        if self.num_masked_views == 1:
            collator = functools.partial(
                collate_single_masked_batched,
                transform=transform,
                masking_strategy=masking_strategy,
            )
        else:  # num_masked_views == 2
            collator = functools.partial(
                collate_double_masked_batched,
                transform=transform,
                masking_strategy=masking_strategy,
                masking_strategy_b=masking_strategy_b,
            )

        return OlmoEarthDataLoader(
            dataset=dataset,
            work_dir=self.work_dir_upath,
            global_batch_size=self.global_batch_size,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            seed=self.seed,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type or get_default_device().type,
            collator=collator,
            drop_last=self.drop_last,
            min_patch_size=self.min_patch_size,
            max_patch_size=self.max_patch_size,
            sampled_hw_p_list=self.sampled_hw_p_list,
            token_budget=self.token_budget,
            num_dataset_repeats_per_epoch=self.num_dataset_repeats_per_epoch,
            transform=transform,
            masking_strategy=masking_strategy,
            masking_strategy_b=masking_strategy_b,
            num_masked_views=self.num_masked_views,
            tokenization_config=self.tokenization_config,
        )


HeliosDataLoader = _deprecated_class_alias(
    OlmoEarthDataLoader, "helios.data.dataloader.HeliosDataLoader"
)
HeliosDataLoaderConfig = _deprecated_class_alias(
    OlmoEarthDataLoaderConfig, "helios.data.dataloader.HeliosDataLoaderConfig"
)
