"""
OlmoEarth Pretrain 数据集模块。

本模块实现了基于 H5 文件格式的 OlmoEarthDataset 数据集类，用于加载和预处理
多模态地球观测数据。核心功能包括：
- 从 H5 文件中读取多模态栅格数据
- 处理缺失模态和缺失时间步的填充
- 数据归一化
- 支持默认矩形裁剪（subset_sample_default）和 CutMix 裁剪（subset_sample_cutmix）
- NDVI 指数计算
- 数据集指纹生成用于版本控制

主要类：
- OlmoEarthDataset: 核心数据集类，继承自 torch.utils.data.Dataset
- OlmoEarthDatasetConfig: 数据集配置类
- GetItemArgs: __getitem__ 方法的参数命名元组
"""

from __future__ import annotations

import hashlib
import io
import logging
import multiprocessing
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, NamedTuple

import h5py

# hdf5 plugin 需要导入以解压某些压缩类型的数据
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from olmo_core.data.utils import get_rng
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain._compat import (
    deprecated_class_alias as _deprecated_class_alias,
)
from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_SEQUENCE_LENGTH,
    MISSING_VALUE,
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.datatypes import (
    OlmoEarthSample,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.types import ArrayTensor

logger = logging.getLogger(__name__)


# =============================================================================
# 性能计时探针（Performance Timing Probes）
# =============================================================================


class _TimingProbe:
    """数据加载管线性能计时探针。

    在 __getitem__ 的每个关键步骤前后插入计时点，
    以固定间隔（默认每 1000 个样本）输出各步骤的平均耗时，
    同时将每次 __getitem__ 的详细耗时写入 CSV 文件，
    用于定位数据加载的性能瓶颈。

    支持通过 enabled 参数开关探针（生产环境可关闭以消除开销）。
    支持缓存命中/未命中统计，在日志和 CSV 中输出命中率。

    使用方式：
        probe = _TimingProbe(enabled=True, log_interval=1000, csv_path="timing_probe.csv")
        probe.start("total")
        ...
        probe.tick("h5_read")
        ...
        probe.end("total")
        probe.record(cache_hit=True)  # 或 probe.record(cache_hit=False)

    日志输出格式示例：
        [TimingProbe] samples=1000 | total=0.082s | h5_read=0.035s | ... | cache_hit_rate=87.3%

    CSV 输出格式示例：
        sample_idx,total,h5_read,...,cache_hit
        0,0.082,0.035,...,1
        1,0.079,0.032,...,0
    """

    def __init__(
        self,
        enabled: bool = True,
        log_interval: int = 1000,
        csv_path: str | None = "timing_probe.csv",
    ):
        """初始化计时探针。

        Args:
            enabled: 是否启用探针。若为 False，所有操作为 no-op，无性能开销。
            log_interval: 每隔多少个样本输出一次统计信息。
            csv_path: CSV 文件路径。若为 None 则不写文件。
        """
        self._enabled = enabled
        self._log_interval = log_interval
        self._sample_count = 0
        self._cumulative: dict[str, float] = {}
        self._marks: dict[str, float] = {}
        self._current: dict[str, float] = {}  # 当前样本各步骤耗时
        self._csv_path = csv_path
        self._csv_header_written = False
        self._csv_step_names: list[str] = []
        # 缓存命中统计
        self._cache_hits = 0
        self._cache_misses = 0
        # 跟踪当前样本中最近一次 tick 的时间，用于链式计时
        self._last_tick_time: float | None = None

    @property
    def enabled(self) -> bool:
        """探针是否启用。"""
        return self._enabled

    def start(self, name: str) -> None:
        """标记一个计时起点，同时重置链式计时。

        Args:
            name: 计时步骤名称。
        """
        if not self._enabled:
            return
        t = time.perf_counter()
        self._marks[name] = t
        self._last_tick_time = t

    def tick(self, name: str, since: str | None = None) -> None:
        """标记一个计时终点，并累计耗时。

        当 since 为 None 时，从上一次 tick/start 的时刻开始计时（链式计时），
        确保每个步骤的耗时独立于其他样本。

        Args:
            name: 计时步骤名称（累计耗时存入此名称）。
            since: 计时起点名称。若为 None，则从上一次 tick 的时刻开始计时。
        """
        if not self._enabled:
            return
        now = time.perf_counter()
        if since is not None:
            elapsed = now - self._marks.get(since, now)
        elif self._last_tick_time is not None:
            elapsed = now - self._last_tick_time
        else:
            elapsed = 0.0
        self._cumulative[name] = self._cumulative.get(name, 0.0) + elapsed
        self._current[name] = elapsed
        self._marks[name] = now
        self._last_tick_time = now

    def end(self, name: str, since: str | None = None) -> None:
        """标记一个计时终点（与 tick 相同，但不更新 marks）。

        Args:
            name: 计时步骤名称。
            since: 计时起点名称。若为 None，则使用 name 作为起点。
        """
        if not self._enabled:
            return
        now = time.perf_counter()
        start_mark = since if since is not None else name
        elapsed = now - self._marks.get(start_mark, now)
        self._cumulative[name] = self._cumulative.get(name, 0.0) + elapsed
        self._current[name] = elapsed

    def record(self, cache_hit: bool | None = None) -> None:
        """将当前样本的耗时写入 CSV 并累计统计。

        每调用一次代表一个样本处理完毕。
        CSV 按步骤顺序写入列，步骤名由首次 record 时确定。

        Args:
            cache_hit: 是否命中 H5 缓存。若为 None 则不统计缓存命中率。
        """
        if not self._enabled:
            return
        self._sample_count += 1

        # 缓存命中统计
        if cache_hit is not None:
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

        # 写 CSV
        if self._csv_path is not None and self._current:
            import csv as _csv
            import os

            step_names = list(self._current.keys())
            # 首次写入时添加 cache_hit 列
            if not self._csv_header_written:
                # 检查文件是否已存在且有内容（跨 epoch 追加）
                file_exists = os.path.exists(self._csv_path) and os.path.getsize(self._csv_path) > 0
                csv_dir = os.path.dirname(self._csv_path)
                if csv_dir and not os.path.exists(csv_dir):
                    os.makedirs(csv_dir, exist_ok=True)
                if not file_exists:
                    # 文件不存在或为空，写入 header
                    with open(self._csv_path, "w", newline="") as f:
                        writer = _csv.writer(f)
                        header = ["sample_idx"] + step_names
                        if cache_hit is not None:
                            header.append("cache_hit")
                        writer.writerow(header)
                self._csv_header_written = True
                self._csv_step_names = step_names

            with open(self._csv_path, "a", newline="") as f:
                writer = _csv.writer(f)
                row = [self._sample_count - 1]  # 0-indexed
                for step in self._csv_step_names:
                    row.append(f"{self._current.get(step, 0.0):.6f}")
                if cache_hit is not None:
                    row.append(1 if cache_hit else 0)
                writer.writerow(row)

        # 日志输出
        if self._sample_count % self._log_interval == 0:
            parts = [f"samples={self._sample_count}"]
            for name, total in self._cumulative.items():
                avg = total / self._log_interval
                parts.append(f"{name}={avg:.4f}s")
            # 缓存命中率
            total_cache_ops = self._cache_hits + self._cache_misses
            if total_cache_ops > 0:
                hit_rate = self._cache_hits / total_cache_ops * 100
                parts.append(f"cache_hit_rate={hit_rate:.1f}%")
                parts.append(f"cache_hits={self._cache_hits}")
                parts.append(f"cache_misses={self._cache_misses}")
            logger.info(f"[TimingProbe] {', '.join(parts)}")
            self._cumulative.clear()
            self._cache_hits = 0
            self._cache_misses = 0


# =============================================================================
# 滑动窗口预加载（Sliding Window Preload）
# =============================================================================


class _SlidingWindowPreloader:
    """滑动窗口 H5 压缩字节预加载器。

    在 epoch 开始时，DataLoader 已知每个 worker 的完整索引序列。
    本类将索引序列分成窗口，当前窗口处理数据时，后台线程预加载
    下一个窗口的 H5 压缩字节，重叠 I/O 与计算。

    内存估算（window_size=500, 6MB/样本）：
    - 每窗口: 500 × 6MB = 3GB
    - 两窗口/worker: 6GB（当前 + 后台预加载）
    - 16 workers 总计: ~96GB

    优化：支持提前预加载第一个窗口，避免epoch开始时的同步等待。

    使用方式：
        preloader = _SlidingWindowPreloader(dataset, indices, window_size=500)
        preloader.preload_first_window()  # 提前异步加载第一个窗口
        preloader.start()  # 直接使用已加载的数据（无需等待）
        h5_bytes = preloader.get_next_bytes()  # 每次调用返回下一个样本
        preloader.stop()   # 清理

    关键属性:
        _current_window: 当前正在消费的窗口（dict[逻辑索引 → bytes]）
        _next_window: 后台线程正在加载的下一个窗口
        _bg_thread: 后台加载线程
        _next_pos: 当前消费位置（顺序计数器）
        _preloaded_first_window: 提前预加载的第一个窗口数据
    """

    def __init__(
        self,
        dataset: OlmoEarthDataset,
        indices: np.ndarray,
        window_size: int,
    ):
        """初始化预加载器。

        Args:
            dataset: OlmoEarthDataset 实例（用于访问 sample_indices 和 h5py_dir）。
            indices: 本 worker 的逻辑索引数组（从 _get_local_instance_indices 获取）。
            window_size: 每个窗口包含的样本数。
        """
        self._dataset = dataset
        self._indices = indices  # 逻辑索引（args.idx 值）
        self._window_size = window_size
        self._current_window: dict[int, bytes] = {}
        self._next_window: dict[int, bytes] = {}
        self._current_window_start = 0  # 当前窗口在 _indices 中的起始位置
        self._next_pos = 0  # 顺序消费计数器
        self._bg_thread: threading.Thread | None = None
        self._bg_error: Exception | None = None
        self._started = False
        self._preloaded_first_window: dict[int, bytes] | None = None
        self._preload_thread: threading.Thread | None = None

    def start(self) -> None:
        """加载第一个窗口（同步或使用预加载缓存），并启动后台线程加载第二个窗口。

        必须在调用 get_next_bytes() 之前调用一次。
        如果之前调用了 preload_first_window()，会等待预加载线程完成（如果还在运行），
        然后使用预加载的数据，减少同步等待时间。
        如果未预加载，则第一个窗口的加载是同步的，会产生较长延迟。
        """
        if self._started:
            return
        self._started = True

        end = min(self._window_size, len(self._indices))

        # 等待预加载线程完成（如果有）
        if self._preload_thread is not None and self._preload_thread.is_alive():
            logger.info(
                f"[SlidingWindowPreloader] Waiting for preload thread to complete..."
            )
            self._preload_thread.join()
            self._preload_thread = None

        # 检查预加载是否有错误
        if self._bg_error is not None:
            error = self._bg_error
            self._bg_error = None
            raise error

        # 使用预加载的第一个窗口数据（如果有）
        if self._preloaded_first_window is not None:
            logger.info(
                f"[SlidingWindowPreloader] Using preloaded first window: "
                f"indices[0:{end}] ({end} samples)"
            )
            self._current_window = self._preloaded_first_window
            self._preloaded_first_window = None  # 清空引用，释放内存
        else:
            logger.info(
                f"[SlidingWindowPreloader] Loading first window (sync): "
                f"indices[0:{end}] ({end} samples), "
                f"estimated {end * 6}MB"
            )
            self._current_window = self._load_window(0, end)

        self._current_window_start = 0

        # 启动后台线程加载第二个窗口
        next_end = min(end + self._window_size, len(self._indices))
        self._start_bg_load(end, next_end)

    def preload_first_window(self, worker_id: int = 0, num_workers: int = 1) -> None:
        """提前异步加载第一个窗口数据。

        在 epoch 开始前（如 DataLoader.__iter__ 开始时但尚未迭代数据）调用，
        后台线程加载第一个窗口，减少 start() 时的同步等待时间。

        此方法是可选的，如果不调用，start() 会同步加载第一个窗口。

        Args:
            worker_id: 当前 worker 的 ID（用于错峰加载）。
            num_workers: 总 worker 数量。
        """
        if self._started or self._preloaded_first_window is not None:
            return  # 已经启动或已预加载

        end = min(self._window_size, len(self._indices))

        # 错峰策略：每个 worker 延迟 worker_id * stagger 秒后再启动预加载
        # 这样 64 个 worker 不会同时争抢网络 I/O
        stagger_per_worker = 0.3  # 每个 worker 间隔 0.3 秒
        stagger_delay = worker_id * stagger_per_worker

        logger.info(
            f"[SlidingWindowPreloader] Preloading first window in background: "
            f"indices[0:{end}] ({end} samples), worker_id={worker_id}, "
            f"stagger_delay={stagger_delay:.1f}s"
        )

        # 使用后台线程异步加载，保存线程引用以便 start() 可以等待
        def _preload_worker():
            if stagger_delay > 0:
                time.sleep(stagger_delay)
            try:
                self._preloaded_first_window = self._load_window(0, end)
            except Exception as e:
                self._bg_error = e

        self._preload_thread = threading.Thread(target=_preload_worker, daemon=True)
        self._preload_thread.start()
        # 注意：不等待线程完成，让加载在后台进行

    def get_next_bytes(self) -> bytes | None:
        """按顺序返回下一个样本的 H5 压缩字节。

        必须按 _indices 的顺序调用（与 DataLoader 的迭代顺序一致）。
        到达窗口边界时自动切换到下一个窗口（等待后台线程完成）。
        如果预加载器尚未启动，会自动调用 start()（等待预加载完成）。

        Returns:
            H5 文件压缩字节。若未预加载则返回 None（__getitem__ 会 fallback 到磁盘）。
        """
        if self._next_pos >= len(self._indices):
            return None

        # 自动启动：如果 preload_first_window() 已调用但 start() 尚未调用
        if not self._started:
            self.start()

        # 检查是否需要切换到下一个窗口
        window_end = self._current_window_start + self._window_size
        if self._next_pos >= window_end:
            self._advance_window()

        logical_idx = int(self._indices[self._next_pos])
        self._next_pos += 1
        return self._current_window.get(logical_idx)

    def _advance_window(self) -> None:
        """切换窗口：等待后台线程完成，将 next_window 变为 current_window，
        并启动新的后台线程加载下一个窗口。"""
        if self._bg_thread is not None:
            logger.debug(
                f"[SlidingWindowPreloader] Waiting for background load to complete "
                f"at position {self._next_pos}"
            )
            self._bg_thread.join()
            self._bg_thread = None

        if self._bg_error is not None:
            error = self._bg_error
            self._bg_error = None
            raise error

        self._current_window = self._next_window
        self._next_window = {}
        self._current_window_start += self._window_size

        # 启动加载下一个窗口
        next_start = self._current_window_start
        next_end = min(next_start + self._window_size, len(self._indices))
        logger.info(
            f"[SlidingWindowPreloader] Window swap: "
            f"now serving indices[{next_start}:{next_end}] ({next_end - next_start} samples)"
        )
        self._start_bg_load(next_start, next_end)

    def _load_window(self, start: int, end: int) -> dict[int, bytes]:
        """同步加载 H5 文件字节。

        将 _indices[start:end] 中的逻辑索引映射到实际 H5 文件索引，
        然后读取每个文件的原始字节。

        使用 ThreadPoolExecutor 并行读取多个文件，显著加速网络存储 I/O。

        Args:
            start: _indices 数组的起始位置。
            end: _indices 数组的结束位置（不含）。

        Returns:
            dict[逻辑索引 → H5 压缩字节]。
        """
        window: dict[int, bytes] = {}

        def _read_single_file(i: int) -> tuple[int, bytes] | None:
            """读取单个 H5 文件，返回 (logical_idx, bytes) 或 None（失败时）。"""
            logical_idx = int(self._indices[i])
            # 通过 sample_indices 映射到实际 H5 文件索引
            if (
                self._dataset.sample_indices is not None
                and logical_idx < len(self._dataset.sample_indices)
            ):
                actual_idx = int(self._dataset.sample_indices[logical_idx])
            else:
                actual_idx = logical_idx

            h5_file_path = self._dataset.h5py_dir / ConvertToH5py.sample_file_pattern.format(
                index=actual_idx
            )
            try:
                with h5_file_path.open("rb") as f:
                    return (logical_idx, f.read())
            except Exception as e:
                logger.warning(
                    f"Failed to preload sample {logical_idx} (actual {actual_idx}): {e}"
                )
                return None

        # 使用 ThreadPoolExecutor 并行读取文件
        # 线程数 = min(文件数, 32) 避免过多线程
        num_threads = min(end - start, 32)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(_read_single_file, range(start, end))
            for result in results:
                if result is not None:
                    logical_idx, data = result
                    window[logical_idx] = data

        return window

    def _start_bg_load(self, start: int, end: int) -> None:
        """启动后台线程加载下一个窗口。

        Args:
            start: _indices 数组的起始位置。
            end: _indices 数组的结束位置（不含）。
        """
        if start >= end:
            self._bg_thread = None
            return

        self._bg_error = None

        def _worker():
            try:
                self._next_window = self._load_window(start, end)
            except Exception as e:
                self._bg_error = e

        self._bg_thread = threading.Thread(target=_worker, daemon=True)
        self._bg_thread.start()

    def stop(self) -> None:
        """清理预加载器：等待后台线程完成，释放内存。"""
        if self._bg_thread is not None:
            self._bg_thread.join()
            self._bg_thread = None
        self._current_window.clear()
        self._next_window.clear()
        self._started = False


# =============================================================================
# 子采样函数（Subsetting Functions）
# =============================================================================


def _get_max_t_within_token_budget(
    sample: OlmoEarthSample,
    h_w_p: int,
    max_tokens_per_instance: int,
    tokenization_config: TokenizationConfig | None = None,
) -> int:
    """在 token 预算内计算允许的最大时间步数。

    给定采样的 h_w_p（高度和宽度方向上的 token 数），
    返回在 max_tokens 预算内允许的最大时间步数 t，
    使得 patchify 后的 OlmoEarthSample 的 token 总数不超过 max_tokens。

    本函数假设采用 (H, W, T=1 patchifying) 的分块方式。

    Args:
        sample: 待子采样的 OlmoEarthSample 实例。
        h_w_p: 高度和宽度方向上的 patch 数量。
        max_tokens_per_instance: 每个 instance 的最大 token 预算。
        tokenization_config: 可选的 tokenization 配置，用于自定义波段分组。

    Returns:
        在 token 预算内允许的最大时间步数 t。
    """
    from math import floor

    used_tokens = 0  # 已使用的 token 数（静态模态）
    time_multiply_tokens = 0  # 随时间线性增长的 token 数（多时相模态）
    for attribute in sample.as_dict().keys():
        if attribute in ("timestamps", "latlon"):
            continue  # 时间戳和经纬度不占 token
        modality_spec = Modality.get(attribute)
        num_band_sets = (
            tokenization_config.get_num_bandsets(attribute)
            if tokenization_config is not None
            else modality_spec.num_band_sets
        )
        if modality_spec.is_spacetime_varying:
            # 时空变化模态：token 数 = h_w_p^2 * band_sets数 * t
            time_multiply_tokens += (h_w_p**2) * num_band_sets
        elif modality_spec.is_space_only_varying:
            # 仅空间变化模态：token 数 = h_w_p^2 * band_sets数
            used_tokens += (h_w_p**2) * num_band_sets
        elif modality_spec.is_time_only_varying:
            # 仅时间变化模态：token 数 = band_sets数 * t
            time_multiply_tokens += num_band_sets
        elif modality_spec.is_static_in_space_and_time:
            # 空间和时间均不变模态：token 数 = band_sets数
            used_tokens += num_band_sets
    if time_multiply_tokens == 0:
        return 1  # 没有多时相模态，t 默认为 1
    remaining_tokens = max_tokens_per_instance - used_tokens  # 剩余 token 预算
    max_t_within_budget = remaining_tokens / time_multiply_tokens  # 最大允许 t
    if max_t_within_budget < 1:
        raise ValueError(
            f"patch_size too small for this sample and budget, h_w_p: {h_w_p}, max_tokens: {max_tokens_per_instance}"
        )

    return min(floor(max_t_within_budget), sample.time)  # 取预算和实际时间步的较小值


def get_valid_start_ts(
    missing_timesteps: dict[str, Any], max_t: int, current_length: int
) -> list[int]:
    """获取有效的时间步起始位置列表。

    在子采样时，需要选择一个起始时间步 t，使得从该位置开始的 max_t 个时间步
    都是有效的（非缺失的）。

    Args:
        missing_timesteps: 字典，键为模态名，值为该模态的缺失时间步掩码。
        max_t: 最大时间步数。
        current_length: 当前序列长度。

    Returns:
        有效起始时间步索引的有序列表。
    """
    if current_length > max_t:
        if not missing_timesteps:
            # 无缺失信息时，所有位置均可作为起始
            valid_start_ts = list(range(current_length - max_t + 1))
        else:
            # 有缺失信息时，需要找到所有模态都有有效数据的位置
            start_ts = set()
            for modality in missing_timesteps:
                valid_timesteps = np.flatnonzero(missing_timesteps[modality])
                # 筛选从该起始位置开始的 max_t 个时间步都在有效范围内
                valid_timesteps = valid_timesteps[
                    valid_timesteps + max_t <= current_length
                ]
                start_ts.update(valid_timesteps)
            valid_start_ts = list(start_ts)
    else:
        # 当前序列长度不超过 max_t，只能从位置 0 开始
        valid_start_ts = [0]
    if len(valid_start_ts) == 0:
        logger.warning(
            f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
        )
        raise ValueError(
            f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
        )
    return sorted(valid_start_ts)


def subset_sample_default(
    sample: OlmoEarthSample,
    patch_size: int,
    max_tokens_per_instance: int | None,
    sampled_hw_p: int,
    current_length: int,
    missing_timesteps_masks: dict[str, Any] | None = None,
    tokenization_config: TokenizationConfig | None = None,
) -> OlmoEarthSample:
    """使用默认矩形裁剪方式对 OlmoEarthSample 进行子采样。

    从样本中随机裁剪一块矩形区域（空间维度）和一段连续时间步（时间维度），
    使得 patchify 后的 token 总数不超过预算。

    Args:
        sample: 待子采样的 OlmoEarthSample 实例。
        patch_size: 当前样本的 patch 大小。
        max_tokens_per_instance: 每个 instance 的 token 预算。若为 None，则不做子采样。
        sampled_hw_p: 高度和宽度方向上的 patch 数量。
        current_length: 当前样本的最大序列长度。
        missing_timesteps_masks: 缺失时间步掩码字典。
        tokenization_config: 可选的 tokenization 配置。

    Returns:
        子采样后的 OlmoEarthSample。
    """
    if max_tokens_per_instance is None:
        return sample  # 无 token 预算限制，不做子采样
    if missing_timesteps_masks is None:
        missing_timesteps_masks = {}

    # 计算 token 预算内允许的最大时间步数
    max_t = _get_max_t_within_token_budget(
        sample, sampled_hw_p, max_tokens_per_instance, tokenization_config
    )
    max_t = min(max_t, MAX_SEQUENCE_LENGTH)
    # 获取有效起始时间步并随机选择一个
    valid_start_ts = get_valid_start_ts(missing_timesteps_masks, max_t, current_length)
    start_t = np.random.choice(valid_start_ts)
    new_data_dict: dict[str, ArrayTensor] = {}

    # 计算采样的空间范围（像素单位）
    sampled_hw = sampled_hw_p * patch_size
    # 随机选择空间起始位置
    start_h = np.random.choice(sample.height - sampled_hw + 1)
    start_w = np.random.choice(sample.width - sampled_hw + 1)

    for attribute, modality in sample.as_dict().items():
        assert modality is not None
        if attribute == "timestamps":
            # 时间戳按时间维度裁剪
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
            continue
        if attribute == "latlon":
            # 经纬度不参与空间裁剪
            new_data_dict[attribute] = modality
            continue
        modality_spec = Modality.get(attribute)
        if modality_spec.is_spacetime_varying:
            # 时空变化模态：裁剪空间和时间维度，注意 image_tile_size_factor 缩放
            new_data_dict[attribute] = modality[
                start_h * modality_spec.image_tile_size_factor : (start_h + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_w * modality_spec.image_tile_size_factor : (start_w + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_t : start_t + max_t,
            ]
        elif modality_spec.is_space_only_varying:
            # 仅空间变化模态：只裁剪空间维度
            new_data_dict[attribute] = modality[
                start_h * modality_spec.image_tile_size_factor : (start_h + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_w * modality_spec.image_tile_size_factor : (start_w + sampled_hw)
                * modality_spec.image_tile_size_factor,
            ]
        elif modality_spec.is_time_only_varying:
            # 仅时间变化模态：只裁剪时间维度
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
        elif modality_spec.is_static_in_space_and_time:
            # 空间和时间均不变的模态：不做裁剪
            new_data_dict[attribute] = modality

    return OlmoEarthSample(**new_data_dict)


def subset_sample_cutmix(
    sample: OlmoEarthSample,
    patch_size: int,
    max_tokens_per_instance: int | None,
    sampled_hw_p: int,
    current_length: int,
    missing_timesteps_masks: dict[str, Any] | None = None,
    tokenization_config: TokenizationConfig | None = None,
) -> OlmoEarthSample:
    """使用 CutMix patch 采样方式对 OlmoEarthSample 进行子采样。

    与默认矩形裁剪不同，CutMix 在空间维度上随机选择不连续的 patch，
    实现更丰富的空间数据增强。

    Args:
        sample: 待子采样的 OlmoEarthSample 实例。
        patch_size: 当前样本的 patch 大小。
        max_tokens_per_instance: 每个 instance 的 token 预算。若为 None，则不做子采样。
        sampled_hw_p: 高度和宽度方向上的 patch 数量。
        current_length: 当前样本的最大序列长度。
        missing_timesteps_masks: 缺失时间步掩码字典。
        tokenization_config: 可选的 tokenization 配置。

    Returns:
        子采样后的 OlmoEarthSample（使用 CutMix 采样）。
    """
    if max_tokens_per_instance is None:
        return sample  # 无 token 预算限制，不做子采样
    if missing_timesteps_masks is None:
        missing_timesteps_masks = {}

    # 计算 token 预算内允许的最大时间步数
    max_t = _get_max_t_within_token_budget(
        sample, sampled_hw_p, max_tokens_per_instance, tokenization_config
    )
    # 获取有效起始时间步并随机选择一个
    valid_start_ts = get_valid_start_ts(missing_timesteps_masks, max_t, current_length)
    start_t = np.random.choice(valid_start_ts)
    new_data_dict: dict[str, ArrayTensor] = {}

    # 计算高度和宽度方向的 patch 数量
    height_p, width_p = sample.height // patch_size, sample.width // patch_size
    # 随机选择不连续的 patch 索引（无放回采样）
    h_p_indices = np.random.choice(height_p, size=sampled_hw_p, replace=False)
    w_p_indices = np.random.choice(width_p, size=sampled_hw_p, replace=False)
    # 将 patch 索引展平为像素索引
    h_indices = [
        i
        for h_p in h_p_indices
        for i in range(h_p * patch_size, (h_p + 1) * patch_size)
    ]
    w_indices = [
        i
        for w_p in w_p_indices
        for i in range(w_p * patch_size, (w_p + 1) * patch_size)
    ]
    # 构建二维网格索引，用于高级索引
    hh, ww = np.meshgrid(h_indices, w_indices, indexing="ij")

    for attribute, modality in sample.as_dict().items():
        assert modality is not None
        if attribute == "timestamps":
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
            continue
        if attribute == "latlon":
            new_data_dict[attribute] = modality
            continue
        modality_spec = Modality.get(attribute)
        if modality_spec.is_spacetime_varying:
            # 使用高级索引选取非连续的 patch
            new_data_dict[attribute] = modality[
                hh * modality_spec.image_tile_size_factor,
                ww * modality_spec.image_tile_size_factor,
                start_t : start_t + max_t,
            ]
        elif modality_spec.is_space_only_varying:
            new_data_dict[attribute] = modality[
                hh * modality_spec.image_tile_size_factor,
                ww * modality_spec.image_tile_size_factor,
            ]
        elif modality_spec.is_time_only_varying:
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
        elif modality_spec.is_static_in_space_and_time:
            new_data_dict[attribute] = modality

    return OlmoEarthSample(**new_data_dict)


class _SharedH5BytesCache:
    """跨 worker 进程共享的 H5 压缩字节缓存。

    使用 multiprocessing.Manager 管理的共享字典，所有 worker 进程共享同一份缓存。
    缓存 H5 文件的原始压缩字节（zstd 压缩后通常 1-3MB/样本），
    而非解压后的 numpy 数组（通常 10-25MB/样本），以减少内存占用。

    优势：
    - 所有 worker 共享同一份缓存（不像 per-worker 缓存那样内存 × num_workers）
    - 缓存压缩字节比缓存解压数据节省约 10 倍内存
    - 读取时跳过磁盘 I/O（网络存储的延迟通常 10-100ms），
      只需内存读取 + zstd 解压（通常 1-5ms），加速约 10-50 倍

    注意：
    - 首次读取时需要加锁，可能成为瓶颈。但实际中读比写多得多，影响很小。
    - multiprocessing.Manager 有序列化开销，但 bytes 对象序列化很快。
    """

    def __init__(self, max_size: int):
        """初始化共享缓存。

        Args:
            max_size: 最大缓存样本数。
        """
        self._max_size = max_size
        try:
            self._manager = multiprocessing.Manager()
            self._cache_dict = self._manager.dict()
            self._cache_keys = self._manager.list()
            self._lock = self._manager.Lock()
            self._is_shared = True
            logger.info(
                f"Initialized shared H5 bytes cache with max_size={max_size}"
            )
        except Exception as e:
            # 如果共享内存初始化失败（如某些环境不支持），回退到 per-worker 缓存
            logger.warning(
                f"Failed to initialize shared cache ({e}), falling back to per-worker cache"
            )
            self._cache_dict = {}
            self._cache_keys = []
            self._is_shared = False

    def get(self, index: int) -> bytes | None:
        """从缓存获取 H5 压缩字节。

        Args:
            index: 样本索引。

        Returns:
            H5 文件的压缩字节数据，若未命中返回 None。
        """
        return self._cache_dict.get(index)

    def put(self, index: int, data: bytes) -> None:
        """将 H5 压缩字节放入缓存。

        使用 LRU 淘汰策略，当缓存满时移除最早插入的条目。

        Args:
            index: 样本索引。
            data: H5 文件的压缩字节数据。
        """
        if index in self._cache_dict:
            return  # 已缓存
        if self._is_shared:
            with self._lock:
                if index in self._cache_dict:
                    return  # 双重检查
                if len(self._cache_keys) >= self._max_size:
                    # FIFO 淘汰最早的条目
                    oldest = self._cache_keys[0]
                    del self._cache_dict[oldest]
                    del self._cache_keys[0]
                self._cache_dict[index] = data
                self._cache_keys.append(index)
        else:
            # per-worker 回退模式
            if len(self._cache_keys) >= self._max_size:
                oldest = self._cache_keys.pop(0)
                del self._cache_dict[oldest]
            self._cache_dict[index] = data
            self._cache_keys.append(index)

    @property
    def size(self) -> int:
        """当前缓存中的样本数。"""
        return len(self._cache_dict)


class GetItemArgs(NamedTuple):
    """OlmoEarthDataset.__getitem__ 方法的参数命名元组。

    属性:
        idx: 样本索引。
        patch_size: patch 大小（像素）。
        sampled_hw_p: 采样的高度和宽度方向上的 patch 数量。
        token_budget: 可选的 token 预算限制。
        tokenization_config: 可选的 tokenization 配置。
    """

    idx: int
    patch_size: int
    sampled_hw_p: int
    token_budget: int | None = None
    tokenization_config: TokenizationConfig | None = None


# TODO: training_modalities 应该是 str 还是 modality_spec？
class OlmoEarthDataset(Dataset):
    """OlmoEarth Pretrain 数据集类，基于 H5 文件格式加载多模态地球观测数据。

    该数据集支持：
    - 从 H5 文件中读取多种模态的栅格数据
    - 处理缺失模态和缺失时间步的填充
    - 数据归一化（预定义和计算两种策略）
    - 支持默认矩形裁剪和 CutMix 裁剪
    - NDVI 指数计算
    - 数据集指纹生成用于版本控制
    - 可选的数据缓存和读取限速

    关键属性:
        h5py_dir: H5 文件目录路径
        training_modalities: 训练使用的模态列表
        dtype: 数据类型
        normalize: 是否应用归一化
        sample_indices: 过滤后的样本索引数组
        latlon_distribution: 样本的地理分布（经纬度数组）

    使用场景:
        用于 OlmoEarth 预训练任务的数据加载，通常与 OlmoEarthDataLoader 配合使用。
    """

    def __init__(
        self,
        h5py_dir: UPath,
        training_modalities: list[str],
        dtype: np.dtype,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        normalize: bool = True,
        cache_dir: UPath | None = None,
        samples_per_sec: float | None = None,
        dataset_percentage: float = 1.0,
        seed: int = 0,
        apply_cutmix: bool = False,
        filter_idx_file: str | None = None,
        h5_read_cache_size: int = 0,
        timing_probe_enabled: bool = True,
        preload_window_size: int = 0,
    ):
        """初始化数据集。

        使用已有的 H5 目录时，设置 h5py_dir 为 H5 目录路径。
        使用原始瓦片目录时，设置 tile_path 为瓦片目录路径，将在训练前的准备步骤中创建 H5 文件。

        来自 OLMo-core 的警告：
            在分布式设置中，确保 work_dir 在所有本地 rank 之间共享，
            并相应设置 fs_local_rank。设置这些字段后，应在做任何其他操作之前
            在主进程中调用 prepare() 方法。

        Args:
            h5py_dir: 包含预处理数据的 H5 目录路径。
            training_modalities: 训练使用的模态名称列表。
            dtype: 数据的 numpy dtype。
            max_sequence_length: 所有时间维度填充到的最大序列长度。
            normalize: 是否对数据应用归一化。
            cache_dir: 可选的本地缓存目录，用于缓存 H5 文件。
            samples_per_sec: 限制每秒读取的样本数（限速），仅在从 h5py_dir 读取时生效。
            dataset_percentage: 使用的数据集百分比（0.0~1.0）。
            seed: 选择数据集百分比时的随机种子。
            apply_cutmix: 是否在子采样时应用 CutMix 增强。
            filter_idx_file: 若非 None，则使用该 numpy 文件中的索引过滤样本。
            h5_read_cache_size: H5 文件压缩字节缓存大小（样本数）。
                缓存 H5 文件的原始压缩字节到共享内存，所有 worker 共享一份。
                zstd 压缩后每样本约 1-3MB，5000 样本只需 5-15GB。
                读取时跳过磁盘 I/O，但仍需解压（解压比 I/O 快得多）。
                设为 0 表示不缓存（默认）。推荐值：5000-20000。
        """
        self.h5py_dir = h5py_dir
        if not self.h5py_dir.exists():
            raise FileNotFoundError(f"H5PY directory does not exist: {self.h5py_dir}")
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录

        self.training_modalities = training_modalities

        self.dtype = dtype
        self.normalize = normalize
        self.dataset_percentage = dataset_percentage
        self.seed = seed
        if self.normalize:
            # 初始化两种归一化器：预定义（min-max）和计算（mean-std）
            self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        self.max_sequence_length = max_sequence_length

        if samples_per_sec is None:
            self.sec_per_sample = None
        else:
            self.sec_per_sample = 1 / samples_per_sec  # 计算每个样本的读取间隔
        self.last_read_time = time.time()

        self.sample_indices: np.ndarray | None = None  # 准备后设置的样本索引
        self.latlon_distribution: np.ndarray | None = None  # 样本的地理分布
        self.apply_cutmix = apply_cutmix
        self.filter_idx_file = filter_idx_file
        # H5 压缩字节共享缓存：所有 worker 进程共享，跳过磁盘 I/O
        # 使用 multiprocessing 管理的共享 OrderedDict，缓存 H5 文件的原始压缩字节
        self._h5_read_cache_size = h5_read_cache_size
        self._h5_bytes_cache: _SharedH5BytesCache | None = None
        # 性能计时探针：每个 worker 进程独立计数，CSV 按 worker 分别写入
        import os
        worker_id = os.getpid()
        csv_path = f"timing_probe_dataset/timing_probe_dataset_w{worker_id}.csv"
        self._timing_probe = _TimingProbe(
            enabled=timing_probe_enabled, log_interval=100, csv_path=csv_path
        )
        # 滑动窗口预加载器：epoch 开始时由 DataLoader 创建，在 __getitem__ 中消费
        # 当 preload_window_size > 0 时，禁用 _h5_bytes_cache（避免双重缓存）
        self._preload_window_size = preload_window_size
        self._preloader: _SlidingWindowPreloader | None = None
        if preload_window_size > 0 and h5_read_cache_size > 0:
            logger.warning(
                f"Both preload_window_size={preload_window_size} and "
                f"h5_read_cache_size={h5_read_cache_size} are set. "
                f"Disabling h5_read_cache_size to avoid double caching."
            )
            self._h5_read_cache_size = 0
        if filter_idx_file is not None:
            # 加载过滤索引文件
            self.indices_to_filter: np.ndarray | None = np.load(filter_idx_file)
            assert isinstance(self.indices_to_filter, np.ndarray), (
                f"Expected filter_idx_file to point to a np.ndarray, got {type(self.indices_to_filter)} instead."
            )
        else:
            self.indices_to_filter = None

    @property
    def fingerprint_version(self) -> str:
        """数据集指纹的版本号，用于版本控制。"""
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        """数据集指纹，可用于识别和比较数据集。

        基于瓦片路径、支持的模态、样本数量和数据类型生成 SHA256 哈希值。

        Returns:
            数据集的 SHA256 哈希字符串。

        Raises:
            RuntimeError: 如果数据集尚未准备。
        """
        if not self.is_dataset_prepared:
            raise RuntimeError("Dataset must be prepared before creating a fingerprint")
        sha256_hash = hashlib.sha256()
        # 从 h5py_dir 路径解析支持的模态信息
        supported_modalities_folder = self.h5py_dir.parent.name
        supported_modalities = supported_modalities_folder.split("_")
        # 将拆分后的模态名重新合并（如 sentinel2 + l2a -> sentinel2_l2a）
        if "l2a" in supported_modalities:
            supported_modalities.remove("l2a")
            supported_modalities.remove("sentinel2")
            supported_modalities.append("sentinel2_l2a")
        if "raster" in supported_modalities:
            supported_modalities.remove("raster")
            supported_modalities.remove("openstreetmap")
            supported_modalities.append("openstreetmap_raster")

        if "naip" in supported_modalities and "10" in supported_modalities:
            supported_modalities.remove("naip")
            supported_modalities.remove("10")
            supported_modalities.append("naip_10")

        if "landcover" in supported_modalities and "1m" in supported_modalities:
            supported_modalities.remove("landcover")
            supported_modalities.remove("1m")
            supported_modalities.append("landcover_1m")

        if "landcover" in supported_modalities and "30m" in supported_modalities:
            supported_modalities.remove("landcover")
            supported_modalities.remove("30m")
            supported_modalities.append("landcover_30m")

        # 经纬度随每个 h5py 文件保存
        supported_modalities.append("latlon")
        num_samples = int(self.h5py_dir.name)  # 目录名即为样本数量

        tile_path = self.h5py_dir.parent.parent.parent

        if self.filter_idx_file is not None:
            filter_file_string = f",filter_idx_file={self.filter_idx_file}"
        else:
            filter_file_string = ""

        # 基于关键信息生成哈希
        sha256_hash.update(
            f"tile_path={tile_path},"
            f"supported_modalities={sorted(supported_modalities)},"
            f"sample_size={num_samples},"
            f"dtype={self.dtype}"
            f"{filter_file_string}".encode()
        )
        return sha256_hash.hexdigest()

    @property
    def sample_metadata_path(self) -> UPath:
        """获取样本元数据文件的路径。"""
        return self.h5py_dir / ConvertToH5py.sample_metadata_fname

    @property
    def latlon_distribution_path(self) -> UPath:
        """获取经纬度分布文件的路径。"""
        return self.h5py_dir / ConvertToH5py.latlon_distribution_fname

    @property
    def is_dataset_prepared(self) -> bool:
        """检查数据集是否已准备（sample_indices 是否已设置）。"""
        return self.sample_indices is not None

    def _filter_sample_indices_for_training(self) -> None:
        """过滤训练用的样本索引。

        更新 sample_indices 数组，仅保留包含至少一个时空变化训练模态的样本。
        同时根据 filter_idx_file 进一步过滤索引。
        """
        # 读取元数据 CSV
        # TODO: Pandas 无法读取 GCS upaths
        metadata_df = pd.read_csv(str(self.sample_metadata_path))
        logger.info(f"Metadata CSV has {len(metadata_df)} samples")
        logger.info(f"columns: {metadata_df.columns}")

        # 获取不包含任何时空变化训练模态的样本索引，这些样本需要移除
        # 跳过派生模态（ignore_when_parsing=True），因为它们在元数据 CSV 中没有列
        spacetime_varying_training_modalities = [
            modality
            for modality in self.training_modalities
            if Modality.get(modality).is_spacetime_varying
            and not Modality.get(modality).ignore_when_parsing
        ]
        if len(spacetime_varying_training_modalities) == 0:
            raise ValueError(
                "no spacetime varying modalities are specified for training"
            )
        # 找到所有时空变化模态列之和为 0 的行（即无任何训练模态的样本）
        no_spacetime_varying_indices = metadata_df[
            metadata_df[spacetime_varying_training_modalities].sum(axis=1) == 0
        ].index

        # 从样本索引中移除这些无效样本
        logger.info(
            f"Filtering out {len(no_spacetime_varying_indices)} samples without any training modalities"
        )
        self.sample_indices = np.setdiff1d(
            self.sample_indices, no_spacetime_varying_indices
        )
        logger.info(
            f"Filtered {len(no_spacetime_varying_indices)} samples to {self.sample_indices.shape} samples"
        )
        # 如果提供了过滤索引文件，进一步取交集
        if self.indices_to_filter is not None:
            self.sample_indices = np.intersect1d(
                self.sample_indices, self.indices_to_filter
            )

            logger.info(
                f"Intersected {len(self.indices_to_filter)} samples to yield {self.sample_indices.shape} samples"
            )

    def _filter_sample_indices_by_dataset_percentage(self) -> None:
        """根据数据集百分比过滤样本索引。

        当 dataset_percentage < 1.0 时，随机选择一定比例的样本。

        Raises:
            AssertionError: 如果 sample_indices 尚未设置。
        """
        assert self.sample_indices is not None, (
            "Sample indices must be set before filtering by dataset percentage"
        )
        if self.dataset_percentage < 1.0:
            rng = get_rng(self.seed)  # 使用确定性随机数生成器
            num_samples = len(self.sample_indices)
            self.sample_indices = rng.choice(
                self.sample_indices,
                size=int(len(self.sample_indices) * self.dataset_percentage),
                replace=False,  # 无放回采样
            )
            logger.info(
                f"Picked {len(self.sample_indices)} samples from {num_samples} samples"
            )

    def prepare(self) -> None:
        """准备数据集。

        此方法应仅由主进程调用，且应在任何其他进程尝试使用数据集之前执行。
        准备步骤包括：加载地理分布、初始化样本索引、过滤无效样本、
        按百分比采样、更新地理分布。
        """
        logger.info("Preparing dataset...")
        if self.is_dataset_prepared:
            logger.info("Dataset is already prepared")
            return

        num_samples = int(self.h5py_dir.name)  # 从目录名获取样本数量
        self.latlon_distribution = self.get_geographic_distribution()
        self.sample_indices = np.arange(num_samples)  # 初始化所有样本索引
        self._filter_sample_indices_for_training()  # 过滤无效样本
        self._filter_sample_indices_by_dataset_percentage()  # 按百分比采样
        self.latlon_distribution = self.latlon_distribution[self.sample_indices]  # 同步更新地理分布

    def get_geographic_distribution(self) -> np.ndarray:
        """获取数据集的地理分布（经纬度坐标）。

        Returns:
            形状为 (N, 2) 的 numpy 数组，包含 N 个样本的 [纬度, 经度] 坐标。
        """
        if self.latlon_distribution_path.exists():
            with self.latlon_distribution_path.open("rb") as f:
                return np.load(f)

    def __len__(self) -> int:
        """获取数据集的样本数量。

        Raises:
            ValueError: 如果数据集尚未准备。
        """
        if self.sample_indices is None:
            raise ValueError("Dataset is not prepared")
        return self.sample_indices.shape[0]

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """对图像数据进行归一化。

        优先尝试计算策略（mean-std），若失败则回退到预定义策略（min-max）。

        Args:
            modality: 模态规格。
            image: 待归一化的图像数据。

        Returns:
            归一化后的图像数据。
        """
        # TODO: 可以后续将模态归一化策略设为可配置
        try:
            return self.normalizer_computed.normalize(modality, image)
        except Exception:
            return self.normalizer_predefined.normalize(modality, image)

    def _compute_ndvi(
        self,
        s2_data: np.ndarray,
        missing_modalities: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """从原始 Sentinel-2 L2A 波段计算 NDVI（归一化植被指数）。

        NDVI = (NIR - Red) / (NIR + Red)，其中 NIR=B08（索引 3），Red=B04（索引 2）。
        如果某个像素的 Red 或 NIR 波段为 MISSING_VALUE，则 NDVI 也设为 MISSING_VALUE。

        Args:
            s2_data: 原始（未归一化）的 S2 L2A 数据，形状为 [H, W, T, C]。
            missing_modalities: 完全缺失的模态列表。

        Returns:
            元组：(ndvi 数组 [H, W, T, 1], 更新后的 missing_modalities 列表)。
        """
        s2_band_order = Modality.SENTINEL2_L2A.band_order
        red = s2_data[..., s2_band_order.index("B04")]  # 红光波段
        nir = s2_data[..., s2_band_order.index("B08")]  # 近红外波段

        # 标记缺失像素
        missing = (red == MISSING_VALUE) | (nir == MISSING_VALUE)

        # 安全计算 NDVI，避免除零
        denom = nir + red
        safe_denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)  # 除零保护
        ndvi = (nir - red) / safe_denom
        ndvi = np.where(np.abs(denom) < 1e-10, 0.0, ndvi)  # 分母为零时 NDVI 设为 0
        ndvi = np.where(missing, MISSING_VALUE, ndvi)  # 缺失像素恢复为 MISSING_VALUE

        # 从 missing_modalities 中移除 "ndvi"，因为已计算
        updated_missing = [m for m in missing_modalities if m != "ndvi"]
        return ndvi[..., np.newaxis].astype(self.dtype), updated_missing

    def _fill_missing_timesteps(
        self,
        modality_data: np.ndarray,
        missing_timestep_mask: np.ndarray,
    ) -> np.ndarray:
        """用缺失值填充缺失的时间步。

        将模态数据的时间维度扩展到 max_sequence_length，
        仅在有效时间步位置填入实际数据，其余位置填入 MISSING_VALUE。

        Args:
            modality_data: 原始模态数据，形状为 [H, W, T, C]。
            missing_timestep_mask: 布尔掩码，True 表示该时间步有效。

        Returns:
            填充后的模态数据，形状为 [H, W, max_sequence_length, C]。
        """
        # 仅在类型不匹配时才转换，避免不必要的完整副本
        if modality_data.dtype != self.dtype:
            modality_data = modality_data.astype(self.dtype)
        h, w, t, c = modality_data.shape

        # 快速路径：如果所有时间步都有效且长度已匹配，直接返回
        if t == self.max_sequence_length and np.all(missing_timestep_mask):
            return modality_data

        # 快速路径：如果所有时间步都有效但需要填充
        if np.all(missing_timestep_mask[:t]) and t < self.max_sequence_length:
            full_timesteps_data = np.full(
                (h, w, max(self.max_sequence_length, missing_timestep_mask.shape[0]), c),
                MISSING_VALUE,
                dtype=self.dtype,
            )
            full_timesteps_data[:, :, :t, :] = modality_data
            return full_timesteps_data

        # 创建全为 MISSING_VALUE 的完整时间步数组
        full_timesteps_data = np.full(
            (h, w, max(self.max_sequence_length, missing_timestep_mask.shape[0]), c),
            MISSING_VALUE,
            dtype=self.dtype,
        )

        # 将有效数据复制到对应的时间步位置
        present_indices = np.where(missing_timestep_mask)[0]
        num_to_copy = min(len(present_indices), t)
        if num_to_copy > 0:
            # 优化：如果有效索引是连续的，使用切片替代高级索引（切片比fancy indexing快得多）
            idx_diff = np.diff(present_indices[:num_to_copy])
            if num_to_copy == 1 or np.all(idx_diff == 1):
                start_idx = present_indices[0]
                end_idx = start_idx + num_to_copy
                full_timesteps_data[:, :, start_idx:end_idx, :] = modality_data[
                    :, :, :num_to_copy, :
                ]
            else:
                full_timesteps_data[:, :, present_indices[:num_to_copy], :] = modality_data[:, :, :num_to_copy, :]

        return full_timesteps_data

    def _fill_missing_modality(
        self, modality: str, height: int | None, width: int | None, time: int
    ) -> np.ndarray:
        """用缺失值填充整个模态数组。

        当某个模态完全缺失时，创建一个全为 MISSING_VALUE 的数组来占位。

        Args:
            modality: 模态名称。
            height: 空间高度（像素）。
            width: 空间宽度（像素）。
            time: 时间步数。

        Returns:
            全为 MISSING_VALUE 的模态数组。
        """
        expected_shape = OlmoEarthSample.compute_expected_shape(
            modality, height, width, time
        )
        logger.debug(f"Filling {modality} with shape {expected_shape}")
        return np.full(
            expected_shape,
            fill_value=MISSING_VALUE,
            dtype=self.dtype,
        )

    @staticmethod
    def extract_hwt_from_sample_dict(
        sample_dict: dict[str, Any],
    ) -> tuple[int, int, int]:
        """从样本字典中提取高度（h）、宽度（w）和时间步数（t）。

        遍历样本字典中的模态数据，找到第一个空间模态来获取 h 和 w，
        并从 timestamps 中获取 t。

        Args:
            sample_dict: 包含各模态数据和 timestamps 的字典。

        Returns:
            元组 (height, width, time)。

        Raises:
            ValueError: 如果样本字典中没有空间模态。
        """
        time = sample_dict["timestamps"].shape[0]
        for mod_name, mod_data in sample_dict.items():
            if mod_name == "timestamps":
                continue
            mod_spec = Modality.get(mod_name)
            if mod_spec.is_spatial and mod_data is not None:
                # 形状为 (H, W, T, C)，无 batch 维度
                height = mod_data.shape[0] // mod_spec.image_tile_size_factor
                width = mod_data.shape[1] // mod_spec.image_tile_size_factor
                return height, width, time
        raise ValueError("Expected sample dict to have at least one spatial modality")

    def fill_sample_with_missing_values(
        self, sample_dict: dict[str, Any], missing_timesteps_masks: dict[str, Any]
    ) -> tuple[OlmoEarthSample, list[str]]:
        """用缺失值填充样本中缺失的模态和时间步。

        遍历所有训练模态：
        - 对于完全缺失的模态，用全 MISSING_VALUE 数组填充
        - 对于部分时间步缺失的模态，用 MISSING_VALUE 填充缺失时间步

        Args:
            sample_dict: 包含各模态数据的字典。
            missing_timesteps_masks: 缺失时间步掩码字典，True 表示有效。

        Returns:
            元组：(填充后的 OlmoEarthSample, 缺失模态名称列表)。
        """
        # 【修改点1】：允许时间步长大于等于max_sequence_length
        current_time_len = sample_dict["timestamps"].shape[0]
        assert current_time_len >= self.max_sequence_length, (
            f"Timestamps shape {sample_dict['timestamps'].shape[0]} does not match max_sequence_length {self.max_sequence_length}"
        )
        missing_modalities = []
        height, width, time = self.extract_hwt_from_sample_dict(sample_dict)
        sample_keys = sample_dict.keys()

        for modality in self.training_modalities:
            # 模态完全缺失：用 MISSING_VALUE 填充
            if modality not in sample_keys:
                sample_dict[modality] = self._fill_missing_modality(
                    modality, height, width, time
                )
                missing_modalities.append(modality)
                continue

            # 多时相模态：处理缺失时间步
            # missing_timesteps_masks 中 True 表示有效，False 表示缺失
            if modality in missing_timesteps_masks:
                mask = missing_timesteps_masks[modality]
                modality_data = sample_dict[modality]

                # 快速路径：如果所有时间步都有效且长度已匹配，仅做类型转换
                if len(mask) == self.max_sequence_length and np.all(mask):
                    if modality_data.dtype != self.dtype:
                        sample_dict[modality] = modality_data.astype(self.dtype)
                    continue

                # 仅在类型不匹配时才转换，避免不必要的完整副本
                if modality_data.dtype != self.dtype:
                    modality_data = modality_data.astype(self.dtype)

                # 如果存在缺失时间步或时间步数不足，用 MISSING_VALUE 填充
                has_missing_timesteps = (
                    not np.all(mask) or len(mask) < self.max_sequence_length
                )
                if has_missing_timesteps:
                    modality_data = self._fill_missing_timesteps(modality_data, mask)
                # 更新样本字典
                sample_dict[modality] = modality_data
        return OlmoEarthSample(**sample_dict), missing_modalities

    def _pad_timestamps(
        self, sample_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """将时间戳填充到 max_sequence_length。

        如果当前时间步数不足，则在末尾复制最后一个时间步进行填充（edge padding）。

        Args:
            sample_dict: 包含 timestamps 数据的字典。

        Returns:
            元组：(更新后的 sample_dict, 填充前的原始序列长度)。
        """
        timestamps_data = sample_dict["timestamps"]
        current_length = timestamps_data.shape[0]
        if current_length < self.max_sequence_length:
            pad_width = ((0, self.max_sequence_length - current_length), (0, 0))
            # 在末尾用最后一个时间步的副本填充
            padded_timestamps = np.pad(
                timestamps_data, pad_width=pad_width, mode="edge"
            )
            sample_dict["timestamps"] = padded_timestamps
        return sample_dict, current_length

    def _apply_throttling(self) -> None:
        """应用读取限速。

        当从 h5py_dir 读取样本时调用，确保不超过配置的每秒读取速率。
        仅对 h5py_dir 读取生效，缓存读取不限速。
        """
        if self.sec_per_sample is None:
            return  # 无限速要求
        elapsed = time.time() - self.last_read_time
        time_to_sleep = self.sec_per_sample - elapsed
        self.last_read_time = time.time()
        logger.info(f"{elapsed} elapsed since last read, sleeping for {time_to_sleep}")
        if time_to_sleep <= 0:
            return  # 已超过限速间隔，无需等待
        time.sleep(time_to_sleep)

    def read_h5_file(
        self, h5_file_path: UPath, cache_index: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """读取 H5 文件，返回样本数据和缺失时间步掩码。

        如果配置了缓存目录，会先将文件缓存到本地再读取，以提高后续读取速度。
        缓存使用原子重命名以避免并发问题。
        如果配置了共享字节缓存，会将原始压缩字节缓存到共享内存。

        Args:
            h5_file_path: H5 文件路径。
            cache_index: 用于共享字节缓存的样本索引。若为 None 则不缓存。

        Returns:
            元组：(sample_dict 包含各模态数据, missing_timesteps_masks 字典)。
        """
        if self.cache_dir is not None:
            cache_file_path = self.cache_dir / h5_file_path.name
            logger.debug(f"Caching H5 file {h5_file_path} to {cache_file_path}")
            if not cache_file_path.exists():
                self._apply_throttling()  # 从远程读取时应用限速
                # 先复制到临时文件，然后原子重命名以避免并发问题
                tmp_file_path = self.cache_dir / (h5_file_path.name + ".tmp")
                with h5_file_path.open("rb") as src, tmp_file_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_file_path.rename(cache_file_path)
            h5_file_path = cache_file_path  # 使用缓存文件路径

        else:
            self._apply_throttling()  # 无缓存时直接限速

        # 读取文件原始字节，用于共享缓存
        h5_bytes: bytes | None = None
        sample_dict = {}
        with h5_file_path.open("rb") as f:
            h5_bytes = f.read()
            with h5py.File(io.BytesIO(h5_bytes), "r") as h5file:
                logger.debug(
                    f"Reading h5 file {h5_file_path} with keys {h5file.keys()}"
                )
                # 读取训练模态的数据和时间戳
                sample_dict = {
                    k: v[()]
                    for k, v in h5file.items()
                    if k in self.training_modalities
                    # TODO: 修复浮动字符串问题
                    or k in ["timestamps"]
                }

                # 读取缺失时间步掩码
                if (
                    missing_mask_group_name
                    := ConvertToH5py.missing_timesteps_mask_group_name
                ) in h5file:
                    missing_timesteps_masks = {
                        k: v[()]
                        for k, v in h5file[missing_mask_group_name].items()
                        if k in self.training_modalities
                    }
                else:
                    # 兼容旧版本：如果文件中不存在掩码组，设为空字典
                    missing_timesteps_masks = {}

        # 将压缩字节缓存到共享内存，供其他 worker 复用
        if (
            self._h5_bytes_cache is not None
            and cache_index is not None
            and h5_bytes is not None
        ):
            self._h5_bytes_cache.put(cache_index, h5_bytes)

        return sample_dict, missing_timesteps_masks

    def _get_h5_file_path(self, index: int) -> UPath:
        """根据索引获取 H5 文件路径。"""
        return self.h5py_dir / ConvertToH5py.sample_file_pattern.format(index=index)

    def _init_h5_bytes_cache(self) -> None:
        """延迟初始化 H5 字节共享缓存。

        必须在 worker 进程中调用（不能在 __init__ 中调用），
        因为 Manager 对象不能跨进程传递。
        """
        if self._h5_read_cache_size > 0 and self._h5_bytes_cache is None:
            self._h5_bytes_cache = _SharedH5BytesCache(self._h5_read_cache_size)

    def _read_h5_from_bytes(
        self, h5_bytes: bytes
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """从内存中的 H5 字节数据解析样本。

        跳过磁盘 I/O，直接从内存中的压缩字节流读取 H5 文件。

        Args:
            h5_bytes: H5 文件的原始压缩字节数据。

        Returns:
            元组：(sample_dict 包含各模态数据, missing_timesteps_masks 字典)。
        """
        sample_dict = {}
        with h5py.File(io.BytesIO(h5_bytes), "r") as h5file:
            sample_dict = {
                k: v[()]
                for k, v in h5file.items()
                if k in self.training_modalities or k in ["timestamps"]
            }
            if (
                missing_mask_group_name
                := ConvertToH5py.missing_timesteps_mask_group_name
            ) in h5file:
                missing_timesteps_masks = {
                    k: v[()]
                    for k, v in h5file[missing_mask_group_name].items()
                    if k in self.training_modalities
                }
            else:
                missing_timesteps_masks = {}
        return sample_dict, missing_timesteps_masks

    @staticmethod
    def _crop_timestamps_and_masks(
        timestamps: np.ndarray, missing_timesteps_masks: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """将时间戳和掩码裁剪到现存模态的首尾有效时间步之间。

        Args:
            timestamps: 时间戳数组。
            missing_timesteps_masks: 缺失时间步掩码字典。

        Returns:
            元组：(裁剪后的 timestamps, 裁剪后的 missing_timesteps_masks)。
        """
        # 假设 missing_timesteps_masks 已经过滤为仅包含训练模态
        if not missing_timesteps_masks:
            first_valid_timestep = 0
            last_valid_timestep = MAX_SEQUENCE_LENGTH
        else:
            # 找到所有模态中最早和最晚的有效时间步
            first_valid_timestep = MAX_SEQUENCE_LENGTH
            last_valid_timestep = 0
            for timestep_mask in missing_timesteps_masks.values():
                valid_timesteps = np.where(timestep_mask)[0]
                if len(valid_timesteps) > 0:
                    first_valid_timestep = min(first_valid_timestep, valid_timesteps[0])
                    last_valid_timestep = max(last_valid_timestep, valid_timesteps[-1])
        # 裁剪时间戳和掩码
        timestamps = timestamps[first_valid_timestep : last_valid_timestep + 1]
        for modality, timestep_mask in missing_timesteps_masks.items():
            missing_timesteps_masks[modality] = timestep_mask[
                first_valid_timestep : last_valid_timestep + 1
            ]
        return timestamps, missing_timesteps_masks

    def __getitem__(self, args: GetItemArgs) -> tuple[int, OlmoEarthSample]:
        """获取指定索引的样本。

        完整的数据加载流程：
        1. 将 args.idx 映射到过滤后的样本索引
        2. 读取 H5 文件获取原始数据
        3. 裁剪时间戳和掩码到有效范围
        4. 填充时间戳到 max_sequence_length
        5. 用缺失值填充缺失的模态和时间步
        6. 子采样（矩形裁剪或 CutMix）
        7. 计算派生模态（如 NDVI）
        8. 归一化

        Args:
            args: GetItemArgs 命名元组，包含 idx、patch_size、sampled_hw_p 等。

        Returns:
            元组：(patch_size, OlmoEarthSample)。
        """
        probe = self._timing_probe
        probe.start("total")

        if hasattr(self, "sample_indices") and self.sample_indices is not None:
            index = self.sample_indices[args.idx]  # 使用过滤后的索引
        else:
            index = args.idx

        # 延迟初始化共享缓存（必须在 worker 进程中）
        if self._h5_read_cache_size > 0:
            self._init_h5_bytes_cache()

        # 尝试从滑动窗口预加载器获取（最高优先级）
        sample_dict = None
        cache_hit = False
        if self._preloader is not None:
            h5_bytes = self._preloader.get_next_bytes()
            if h5_bytes is not None:
                sample_dict, missing_timesteps_masks = self._read_h5_from_bytes(h5_bytes)
                cache_hit = True

        # 尝试从共享压缩字节缓存读取
        if sample_dict is None and self._h5_bytes_cache is not None:
            h5_bytes = self._h5_bytes_cache.get(index)
            if h5_bytes is not None:
                sample_dict, missing_timesteps_masks = self._read_h5_from_bytes(
                    h5_bytes
                )
                cache_hit = True

        if sample_dict is None:
            h5_file_path = self._get_h5_file_path(index)
            # 读取 H5 文件（会自动将原始字节缓存到共享内存）
            sample_dict, missing_timesteps_masks = self.read_h5_file(
                h5_file_path, cache_index=index
            )
        probe.tick("h5_read")

        # 裁剪时间戳和掩码
        timestamps, missing_timesteps_masks = self._crop_timestamps_and_masks(
            sample_dict["timestamps"], missing_timesteps_masks
        )
        sample_dict["timestamps"] = timestamps
        probe.tick("crop_timestamps")

        # 填充时间戳到 max_sequence_length
        sample_dict, current_length = self._pad_timestamps(sample_dict)
        probe.tick("pad_timestamps")

        # 用缺失值填充缺失的模态和时间步（当前耗时约 0.08 秒，可能成为小模型的瓶颈）
        sample, missing_modalities = self.fill_sample_with_missing_values(
            sample_dict, missing_timesteps_masks
        )
        probe.tick("fill_missing")

        # 子采样
        if self.apply_cutmix:
            subset_sample = subset_sample_cutmix(
                sample,
                patch_size=args.patch_size,
                max_tokens_per_instance=args.token_budget,
                sampled_hw_p=args.sampled_hw_p,
                current_length=current_length,
                missing_timesteps_masks=missing_timesteps_masks,
                tokenization_config=args.tokenization_config,
            )
        else:
            subset_sample = subset_sample_default(
                sample,
                patch_size=args.patch_size,
                max_tokens_per_instance=args.token_budget,
                sampled_hw_p=args.sampled_hw_p,
                current_length=current_length,
                missing_timesteps_masks=missing_timesteps_masks,
                tokenization_config=args.tokenization_config,
            )
        probe.tick("subset_sample")

        sample_dict = subset_sample.as_dict()

        # 如果请求了 NDVI 且有 S2 L2A 数据，从原始（未归一化）波段计算 NDVI
        if (
            "ndvi" in sample_dict
            and "sentinel2_l2a" in sample_dict
            and "sentinel2_l2a" not in missing_modalities
        ):
            sample_dict["ndvi"], missing_modalities = self._compute_ndvi(
                sample_dict["sentinel2_l2a"], missing_modalities
            )
        probe.tick("ndvi")

        if self.normalize:
            missing_modalities_set = set(missing_modalities)
            for modality_name in sample_dict.keys():
                if modality_name == "timestamps":
                    continue
                if modality_name in missing_modalities_set:
                    continue
                modality_data = sample_dict[modality_name]
                # 记录缺失值位置
                missing_mask = modality_data == MISSING_VALUE
                normalized_data = self.normalize_image(
                    Modality.get(modality_name), modality_data
                )
                # 归一化后恢复缺失值标记
                sample_dict[modality_name] = np.where(
                    missing_mask, modality_data, normalized_data
                ).astype(self.dtype)
        probe.tick("normalize")

        probe.end("total", since="total")
        probe.record(cache_hit=cache_hit if self._h5_bytes_cache is not None else None)

        return args.patch_size, OlmoEarthSample(**sample_dict)


@dataclass
class OlmoEarthDatasetConfig(Config):
    """OlmoEarthDataset 的配置类。

    属性:
        h5py_dir: H5 文件目录路径字符串。
        training_modalities: 训练使用的模态名称列表。
        dtype: 数据类型字符串（如 "float32"）。
        normalize: 是否应用归一化。
        cache_dir: 可选的本地缓存目录路径字符串。
        samples_per_sec: 每秒读取样本数的限速。
        dataset_percentage: 使用的数据集百分比。
        seed: 随机种子。
        apply_cutmix: 是否应用 CutMix 增强。
        filter_idx_file: 可选的索引过滤文件路径。
    """

    h5py_dir: str
    training_modalities: list[str]
    dtype: str = "float32"
    normalize: bool = True
    cache_dir: str | None = None
    samples_per_sec: float | None = None
    dataset_percentage: float = 1.0
    seed: int = 0
    apply_cutmix: bool = False
    filter_idx_file: str | None = None
    h5_read_cache_size: int = 0
    timing_probe_enabled: bool = True
    preload_window_size: int = 0  # 滑动窗口预加载大小，0 = 禁用

    def get_numpy_dtype(self) -> np.dtype:
        """获取 numpy 数据类型。

        Returns:
            对应的 numpy.dtype。

        Raises:
            ValueError: 如果 dtype 不受支持。
        """
        if self.dtype == "float16":
            return np.float16
        elif self.dtype == "float32":
            return np.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def validate(self) -> None:
        """验证配置参数的合法性。

        Raises:
            ValueError: 如果 training_modalities 不是列表。
        """
        if not isinstance(self.training_modalities, list):
            raise ValueError("training_modalities must be a list")

    @property
    def h5py_dir_upath(self) -> UPath:
        """获取 H5 目录的 UPath 对象。"""
        return UPath(self.h5py_dir)

    @property
    def cache_dir_upath(self) -> UPath:
        """获取缓存目录的 UPath 对象。"""
        return UPath(self.cache_dir)

    def build(self) -> OlmoEarthDataset:
        """构建 OlmoEarthDataset 实例。

        Returns:
            配置好的 OlmoEarthDataset 实例。
        """
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["h5py_dir"] = self.h5py_dir_upath  # 转换为 UPath
        kwargs["cache_dir"] = (
            self.cache_dir_upath if self.cache_dir is not None else None
        )
        kwargs["dtype"] = self.get_numpy_dtype()  # 转换为 numpy dtype
        logger.info(f"OlmoEarthDataset kwargs: {kwargs}")
        return OlmoEarthDataset(**kwargs)


# 向后兼容的废弃别名
HeliosSample = _deprecated_class_alias(
    OlmoEarthSample, "helios.data.dataset.HeliosSample"
)
HeliosDataset = _deprecated_class_alias(
    OlmoEarthDataset, "helios.data.dataset.HeliosDataset"
)
HeliosDatasetConfig = _deprecated_class_alias(
    OlmoEarthDatasetConfig, "helios.data.dataset.HeliosDatasetConfig"
)
