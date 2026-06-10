"""
OlmoEarth Pretrain 数据集的归一化模块。

本模块提供了对多模态栅格数据进行归一化的功能，支持两种策略：
- PREDEFINED: 使用预定义的 min/max 值进行线性归一化：(data - min) / (max - min)
- COMPUTED: 使用预计算的 mean/std 值，先转换为 min/max 范围后线性归一化

归一化配置从 JSON 文件中加载，映射格式为：模态名 -> 波段名 -> {min, max} 或 {mean, std}。
"""

import json
import logging
from enum import Enum
from importlib.resources import files

import numpy as np

from olmoearth_pretrain.data.constants import ModalitySpec

logger = logging.getLogger(__name__)

# 模块级缓存，避免每个 Normalizer 实例重复加载 JSON 配置文件
_predefined_config_cache: dict | None = None
_computed_config_cache: dict | None = None


def load_predefined_config() -> dict[str, dict[str, dict[str, float]]]:
    """加载预定义的归一化配置。

    配置格式：模态名 -> 波段名 -> {min, max}。

    Returns:
        预定义归一化配置字典。
    """
    global _predefined_config_cache
    if _predefined_config_cache is not None:
        return _predefined_config_cache
    with (
        files("olmoearth_pretrain.data.norm_configs") / "predefined.json"
    ).open() as f:
        _predefined_config_cache = json.load(f)
    return _predefined_config_cache


def load_computed_config() -> dict[str, dict]:
    """加载计算的归一化配置。

    配置格式：模态名 -> 波段名 -> {mean, std}。

    Returns:
        计算的归一化配置字典。
    """
    global _computed_config_cache
    if _computed_config_cache is not None:
        return _computed_config_cache
    with (files("olmoearth_pretrain.data.norm_configs") / "computed.json").open() as f:
        _computed_config_cache = json.load(f)
    return _computed_config_cache


class Strategy(Enum):
    """归一化策略枚举。

    - PREDEFINED: 使用预定义的 min/max 值进行归一化
    - COMPUTED: 使用预计算的 mean/std 值进行归一化
    """

    PREDEFINED = "predefined"
    COMPUTED = "computed"


class Normalizer:
    """数据归一化器。

    根据 Strategy 选择使用预定义或计算的归一化值，
    将各模态各波段的数据归一化到 [0, 1] 范围。

    关键属性:
        strategy: 归一化策略（PREDEFINED 或 COMPUTED）
        std_multiplier: 标准差乘数（仅 COMPUTED 策略），用于确定归一化范围
        norm_config: 加载的归一化配置字典
        _cache: 预计算的 min/max/inv_range 缓存，避免每次归一化时重复构建数组
    """

    def __init__(
        self,
        strategy: Strategy,
        std_multiplier: float | None = 2,
    ) -> None:
        """初始化归一化器。

        Args:
            strategy: 归一化策略（PREDEFINED 或 COMPUTED）。
            std_multiplier: 标准差乘数，仅用于 COMPUTED 策略。
                归一化范围为 [mean - std_multiplier*std, mean + std_multiplier*std]，
                默认值为 2，覆盖约 95% 的数据。
        """
        self.strategy = strategy
        self.std_multiplier = std_multiplier
        if self.strategy == Strategy.PREDEFINED:
            self.norm_config = load_predefined_config()
        elif self.strategy == Strategy.COMPUTED:
            self.norm_config = load_computed_config()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        # 预计算缓存：modality_name -> (min_array, inv_range_array)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _build_cache_for_modality(self, modality: ModalitySpec) -> tuple[np.ndarray, np.ndarray]:
        """为指定模态预计算并缓存 min 和 inv_range 数组。

        Args:
            modality: 模态规格。

        Returns:
            元组 (min_array, inv_range_array)，其中 inv_range = 1 / (max - min)。
        """
        if modality.name in self._cache:
            return self._cache[modality.name]

        modality_bands = modality.band_order
        modality_norm_values = self.norm_config[modality.name]

        if self.strategy == Strategy.PREDEFINED:
            min_vals = []
            max_vals = []
            for band in modality_bands:
                if band not in modality_norm_values:
                    raise ValueError(f"Band {band} not found in config")
                min_vals.append(modality_norm_values[band]["min"])
                max_vals.append(modality_norm_values[band]["max"])
            min_arr = np.array(min_vals, dtype=np.float32)
            max_arr = np.array(max_vals, dtype=np.float32)
        else:  # COMPUTED
            mean_vals = []
            std_vals = []
            for band in modality_bands:
                if band not in modality_norm_values:
                    raise ValueError(f"Band {band} not found in config")
                mean_vals.append(modality_norm_values[band]["mean"])
                std_vals.append(modality_norm_values[band]["std"])
            mean_arr = np.array(mean_vals, dtype=np.float32)
            std_arr = np.array(std_vals, dtype=np.float32)
            min_arr = mean_arr - self.std_multiplier * std_arr
            max_arr = mean_arr + self.std_multiplier * std_arr

        inv_range = np.reciprocal(max_arr - min_arr)
        self._cache[modality.name] = (min_arr, inv_range)
        return min_arr, inv_range

    def _normalize_predefined(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """使用预定义值进行归一化：normalized = (data - min) / (max - min)。

        Args:
            modality: 模态规格。
            data: 待归一化的数据，最后维度为通道（波段）数。

        Returns:
            归一化后的数据。
        """
        min_arr, inv_range = self._build_cache_for_modality(modality)
        return (data - min_arr) * inv_range

    def _normalize_computed(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """使用计算值进行归一化。

        将 mean 和 std 转换为 min 和 max 后线性归一化。

        Args:
            modality: 模态规格。
            data: 待归一化的数据。

        Returns:
            归一化后的数据。
        """
        min_arr, inv_range = self._build_cache_for_modality(modality)
        return (data - min_arr) * inv_range

    def normalize(self, modality: ModalitySpec, data: np.ndarray) -> np.ndarray:
        """对数据进行归一化。

        根据 strategy 选择对应的归一化方法。

        Args:
            modality: 模态规格。
            data: 待归一化的数据。

        Returns:
            归一化后的数据。

        Raises:
            ValueError: 如果策略无效。
        """
        if self.strategy == Strategy.PREDEFINED:
            return self._normalize_predefined(modality, data)
        elif self.strategy == Strategy.COMPUTED:
            return self._normalize_computed(modality, data)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
