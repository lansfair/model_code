"""Constants shared across the OlmoEarth Pretrain package.

Warning: this is only developed for raster data currently.
"""

from dataclasses import dataclass
from enum import Enum

# The highest resolution that we are working at.
# Everything else is a factor (which is a power of 2) coarser than this resolution.
BASE_RESOLUTION = 0.625

# The default image tile size.
# Some images may be smaller if they are stored at a coarser resolution compared to the
# resolution that the grid is based on.
IMAGE_TILE_SIZE = 256

PROJECTION_CRS = "EPSG:4326"

# Default missing value for raster data.
MISSING_VALUE = -99999

# Default maximum sequence length.
MAX_SEQUENCE_LENGTH = 12

# Resolution of the input data in meters
BASE_GSD = 10
# Default nodata value for Sentinel-1 data.
SENTINEL1_NODATA = -32768

# Number of timesteps for YEAR data.
YEAR_NUM_TIMESTEPS = 12


def get_resolution(resolution_factor: int) -> float | int:
    """Compute the resolution.

    If it is an integer, then we cast it to int so that it works with the raw OlmoEarth Pretrain
    dataset, where some files are named based on the integer. We may want to change
    this in the future to avoid the extra code here.
    """
    resolution = BASE_RESOLUTION * resolution_factor
    if float(int(resolution)) == resolution:
        return int(resolution)
    return resolution


@dataclass(frozen=True)
class BandSet:
    """A group of bands that is stored at the same resolution.

    Many modalities only have one band set, but some have different bands at different
    resolutions.
    """

    # List of band names.
    bands: list[str]

    # Resolution is BASE_RESOLUTION * resolution_factor.
    # If resolution == 0, this means the data
    # does not vary in space (e.g. latlons)
    resolution_factor: int

    def __hash__(self) -> int:
        """Hash this BandSet."""
        return hash((tuple(self.bands), self.resolution_factor))

    def get_resolution(self) -> float:
        """Compute the resolution."""
        return get_resolution(self.resolution_factor)

    def get_expected_image_size(self, modality_resolution_factor: int) -> int:
        """Get the expected size of images containing these bands.

        Args:
            modality_resolution_factor: the resolution factor of the modality.

        Returns:
            the expected image size.
        """
        return IMAGE_TILE_SIZE // (self.resolution_factor // modality_resolution_factor)


class TimeSpan(str, Enum):
    """Enum to distinguish data that is valid for different time ranges."""

    # Only one data point (not time series).
    STATIC = "static"

    # Monthly over one year.
    YEAR = "year"

    # Every data point in a two-week period.
    TWO_WEEK = "two_week"

    def get_suffix(self) -> str:
        """Returns the suffix used for this timespan in raw OlmoEarth Pretrain dataset."""
        if self == TimeSpan.STATIC:
            return ""
        if self == TimeSpan.YEAR:
            return "_monthly"
        if self == TimeSpan.TWO_WEEK:
            return "_freq"
        raise ValueError("invalid TimeSpan")


@dataclass(frozen=True)
class ModalitySpec:
    """Modality specification.
    
    模态规格配置类，定义每种遥感数据模态的关键属性。
    
    核心作用：
    1. 描述不同模态（如Sentinel-2、Landsat、SRTM等）的物理特性
    2. 控制数据加载时的分辨率对齐和tile尺寸调整
    3. 定义波段分组方式，影响token化策略
    4. 决定时间维度的处理方式
    
    使用场景：
    - 在FlexiPatchEmbed中，通过image_tile_size_factor调整实际patch size
    - 在数据加载时，根据tile_resolution_factor计算地理覆盖范围
    - 在模型配置中，通过band_sets确定输入通道数

    Args:
        name: the name of the modality.
            模态名称，唯一标识符
            例如："sentinel2_l2a", "landsat", "srtm", "worldcover"
            用于在配置文件中引用该模态，以及在H5文件中标识数据集
            
        tile_resolution_factor: the factor of how much more ground area is covered by the tile compared with a tile
                        of IMAGE_TILE_SIZE x IMAGE_TILE_SIZE pixels at the base resolution.
            Tile分辨率因子，表示相对于基准分辨率的地理覆盖倍数
            计算公式：实际地理面积 / (IMAGE_TILE_SIZE² × BASE_RESOLUTION²)
            
            关键理解：
            - BASE_RESOLUTION = 0.625米/像素（基准分辨率）
            - IMAGE_TILE_SIZE = 256像素（标准tile尺寸）
            - 如果某模态分辨率为10米/像素，则tile_resolution_factor = (10/0.625)² = 256
            - 这意味着同样256×256像素的tile，该模态覆盖的地理面积是基准的256倍
            
            应用场景：
            - 多模态对齐：确保不同分辨率的数据在相同地理范围内采样
            - Token预算控制：高分辨率模态需要更大的factor来平衡token数量
            
        band_sets: the band sets of the modality, ie the units of tokenization.
            波段集合列表，定义该模态的token化单元
            每个BandSet代表一组会被一起处理的波段
            
            设计目的：
            - 灵活组织波段：可以将相关波段分组（如RGB、NIR、SWIR）
            - 独立token化：每个BandSet生成独立的token序列
            - 选择性使用：训练时可以选择性地启用某些BandSet
            
            示例：
            ```python
            # Sentinel-2的多波段分组
            band_sets=[
                BandSet(name="rgb", bands=["B04", "B03", "B02"], ...),  # RGB真彩色
                BandSet(name="nir", bands=["B08"], ...),                 # 近红外
                BandSet(name="swir", bands=["B11", "B12"], ...)          # 短波红外
            ]
            # 这样Sentinel-2会产生3组独立的tokens
            ```
            
            注意事项：
            - BandSet中的波段数量必须与H5文件中对应数据的最后一个维度匹配
            - 所有BandSet的波段总数应等于该模态的实际波段数
            
        is_multitemporal: whether the modality is multitemporal.
            是否为多时相模态，决定是否有时间维度
            - True: 该模态包含多个时间步的数据（如Sentinel-2的时间序列）
            - False: 该模态是静态的或单时相的（如SRTM高程、WorldCover土地覆盖）
            
            影响：
            - 数据形状：True时为[batch, h, w, time, channels]，False时为[batch, h, w, channels]
            - 内存占用：多时相模态需要存储多个时间步，占用更多内存
            - 模型处理：编码器需要处理时间维度，可能使用时序注意力机制
            - Masking策略：可以对时间步进行mask，实现时序预测任务
            
            典型多时相模态：
            - Sentinel-2（光学影像，定期重访）
            - Sentinel-1（雷达影像，全天候观测）
            - Landsat（历史长时序数据）
            
            典型单时相模态：
            - SRTM（数字高程模型，静态）
            - WorldCover（土地覆盖分类，年度更新但视为静态）
            - OpenStreetMap（矢量数据栅格化，静态）
            
        ignore_when_parsing: whether to ignore the modality when parsing the data form the csv file.
            解析CSV文件时是否忽略该模态
            - True: 跳过该模态，不从CSV读取，不从H5文件加载
            - False: 正常处理该模态
            
            使用场景：
            1. 实验消融研究：临时禁用某些模态，评估其对模型性能的贡献
            2. 数据缺失处理：某些区域缺少特定模态数据时跳过
            3. 调试模式：快速测试时只加载部分模态加速迭代
            4. 资源限制：显存不足时减少模态数量
            
            注意：
            - 这是代码层面的过滤，不影响H5文件的物理存储
            - 如果设置为True，对应的模型输入通道数也会减少
            - 修改后需要重新初始化DataLoader以生效
            
        image_tile_size_factor: the factor of how much bigger the dimensions of the image tile are compared with the base tile size.
            图像tile尺寸因子，控制实际像素尺寸相对于基准的放大倍数
            默认值为1，表示使用标准的IMAGE_TILE_SIZE（如256×256）
            
            核心作用：
            - 在FlexiPatchEmbed中，实际patch_size = patch_size_at_16 × image_tile_size_factor
            - 允许不同分辨率的模态使用不同的tile像素尺寸
            
            设计原理：
            对于高分辨率模态（如2米/像素的Planet影像）：
            - 如果保持地理覆盖不变，需要更大的像素尺寸（如1024×1024）
            - 设置image_tile_size_factor = 4，则实际tile为1024×1024
            - 同时增大patch_size，避免token数量爆炸
            
            对于低分辨率模态（如30米/像素的Landsat）：
            - 可以使用较小的像素尺寸（如64×64）
            - 设置image_tile_size_factor = 0.25，则实际tile为64×64
            
            计算公式：
            ```
            实际tile像素尺寸 = IMAGE_TILE_SIZE × image_tile_size_factor
            实际patch_size = 基准patch_size × image_tile_size_factor
            ```
            
            示例配置：
            ```python
            # 高分辨率Planet影像（2米/像素）
            ModalitySpec(
                name="planet",
                image_tile_size_factor=4,  # 1024×1024像素
                ...
            )
            
            # 中等分辨率Sentinel-2（10米/像素）
            ModalitySpec(
                name="sentinel2",
                image_tile_size_factor=1,  # 256×256像素（默认）
                ...
            )
            
            # 低分辨率Landsat（30米/像素）
            ModalitySpec(
                name="landsat",
                image_tile_size_factor=0.5,  # 128×128像素
                ...
            )
            ```
            
            与tile_resolution_factor的区别：
            - tile_resolution_factor: 描述地理覆盖面积的相对大小（物理意义）
            - image_tile_size_factor: 描述像素尺寸的相对大小（计算意义）
            - 两者可以独立设置，但通常高分辨率模态会同时具有较大的两个因子
            
            性能影响：
            - 增大image_tile_size_factor会增加输入像素数，可能导致显存压力
            - 需要配合增大patch_size来控制token数量
            - 建议在token_budget允许的范围内调整
    """

    name: str
    tile_resolution_factor: int
    band_sets: list[BandSet]
    is_multitemporal: bool
    ignore_when_parsing: bool  # If true this modality is not parsed from the csv file and not loaded form a file
    image_tile_size_factor: int = 1

    def __hash__(self) -> int:
        """Hash this Modality."""
        return hash(self.name)

    def get_tile_resolution(self) -> float:
        """Compute the tile resolution."""
        return get_resolution(self.tile_resolution_factor)

    def bandsets_as_indices(self) -> list[list[int]]:
        """Return band sets as indices."""
        indices = []
        offset = 0
        for band_set in self.band_sets:
            num_bands = len(band_set.bands)
            indices.append(list(range(offset, offset + num_bands)))
            offset += num_bands
        return indices

    @property
    def band_order(self) -> list[str]:
        """Get all bands."""
        return sum((list(band_set.bands) for band_set in self.band_sets), [])

    @property
    def num_band_sets(self) -> int:
        """Get the number of band sets."""
        return len(self.band_sets)

    @property
    def num_bands(self) -> int:
        """Get the number of channels.

        The number of channels is the sum of the number of bands in all the band sets.
        """
        return sum(len(band_set.bands) for band_set in self.band_sets)

    def get_expected_tile_size(self) -> int:
        """Get the expected size of the tile."""
        if self.image_tile_size_factor < 0:
            return IMAGE_TILE_SIZE // abs(self.image_tile_size_factor)
        else:
            return IMAGE_TILE_SIZE * self.image_tile_size_factor

    @property
    def is_spatial(self) -> bool:
        """Does the modality have spatial data."""
        # Tile size must be greater than 1 to have spatial varying data.
        return self.get_tile_resolution() > 0 and self.get_expected_tile_size() > 1

    @property
    def is_spacetime_varying(self) -> bool:
        """Does the modality vary in space and time."""
        return self.is_spatial and self.is_multitemporal

    @property
    def is_space_only_varying(self) -> bool:
        """Does the modality vary in space and not time."""
        return self.is_spatial and not self.is_multitemporal

    @property
    def is_time_only_varying(self) -> bool:
        """Does the modality vary in time and not space."""
        return not self.is_spatial and self.is_multitemporal

    @property
    def is_static_in_space_and_time(self) -> bool:
        """Does the modality vary in neither space or space."""
        return not self.is_spatial and not self.is_multitemporal


class Modality:
    """Enum-like access to ModalitySpecs."""

    NAIP = ModalitySpec(
        name="naip",
        tile_resolution_factor=1,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # NAIP_10 is the NAIP data that covers the same extent as a IMAGE_TILE_SIZE x IMAGE_TILE_SIZE tile
    # at 10 m/pixel resolution but is still stored at NAIP resolution.
    NAIP_10 = ModalitySpec(
        name="naip_10",
        tile_resolution_factor=16,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
        ignore_when_parsing=False,
        # Currently this is set to 4x (2.5 m/pixel) so that it is more feasible to
        # train with NAIP_10. This way we end up with 512x512 NAIP images in the
        # 128x128 H5 files instead of 2048x2048, which slows down data loading.
        image_tile_size_factor=4,
    )

    SENTINEL1 = ModalitySpec(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["vv", "vh"], 16)],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    SENTINEL2 = ModalitySpec(
        name="sentinel2",
        tile_resolution_factor=16,
        band_sets=[
            # 10 m/pixel bands.
            BandSet(["B02", "B03", "B04", "B08"], 16),
            # 20 m/pixel bands.
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            # 60 m/pixel bands that we store at 40 m/pixel.
            BandSet(["B01", "B09", "B10"], 64),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    SENTINEL2_L2A = ModalitySpec(
        name="sentinel2_l2a",
        tile_resolution_factor=16,
        band_sets=[
            # 10 m/pixel bands.
            BandSet(["B02", "B03", "B04", "B08"], 16),
            # 20 m/pixel bands.
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            # 60 m/pixel bands that we store at 40 m/pixel.
            BandSet(["B01", "B09"], 64),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    LANDSAT = ModalitySpec(
        name="landsat",
        tile_resolution_factor=16,
        band_sets=[
            # 15 m/pixel bands that we store at 10 m/pixel.
            BandSet(["B8"], 16),
            # 30 m/pixel bands that we store at 20 m/pixel.
            BandSet(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 32),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    WORLDCOVER = ModalitySpec(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    WORLDCEREAL = ModalitySpec(
        name="worldcereal",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "tc-annual-temporarycrops-classification",
                    "tc-maize-main-irrigation-classification",
                    "tc-maize-main-maize-classification",
                    "tc-maize-second-irrigation-classification",
                    "tc-maize-second-maize-classification",
                    "tc-springcereals-springcereals-classification",
                    "tc-wintercereals-irrigation-classification",
                    "tc-wintercereals-wintercereals-classification",
                ],
                16,
            )
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    SRTM = ModalitySpec(
        name="srtm",
        tile_resolution_factor=16,
        band_sets=[BandSet(["srtm"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    OPENSTREETMAP = ModalitySpec(
        name="openstreetmap",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "aerialway_pylon",
                    "aerodrome",
                    "airstrip",
                    "amenity_fuel",
                    "building",
                    "chimney",
                    "communications_tower",
                    "crane",
                    "flagpole",
                    "fountain",
                    "generator_wind",
                    "helipad",
                    "highway",
                    "leisure",
                    "lighthouse",
                    "obelisk",
                    "observatory",
                    "parking",
                    "petroleum_well",
                    "power_plant",
                    "power_substation",
                    "power_tower",
                    "river",
                    "runway",
                    "satellite_dish",
                    "silo",
                    "storage_tank",
                    "taxiway",
                    "water_tower",
                    "works",
                ],
                1,
            )
        ],
        is_multitemporal=False,
        ignore_when_parsing=True,
    )

    OPENSTREETMAP_RASTER = ModalitySpec(
        name="openstreetmap_raster",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "aerialway_pylon",
                    "aerodrome",
                    "airstrip",
                    "amenity_fuel",
                    "building",
                    "chimney",
                    "communications_tower",
                    "crane",
                    "flagpole",
                    "fountain",
                    "generator_wind",
                    "helipad",
                    "highway",
                    "leisure",
                    "lighthouse",
                    "obelisk",
                    "observatory",
                    "parking",
                    "petroleum_well",
                    "power_plant",
                    "power_substation",
                    "power_tower",
                    "river",
                    "runway",
                    "satellite_dish",
                    "silo",
                    "storage_tank",
                    "taxiway",
                    "water_tower",
                    "works",
                ],
                4,
            )
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    ERA5 = ModalitySpec(
        name="era5",
        # 9 km/pixel bands that we store at 150 m/pixel.
        tile_resolution_factor=256,
        band_sets=[
            BandSet(
                [
                    "2m-temperature",
                    "2m-dewpoint-temperature",
                    "surface-pressure",
                    "10m-u-component-of-wind",
                    "10m-v-component-of-wind",
                    "total-precipitation",
                ],
                256,
            ),
        ],
        is_multitemporal=True,
        ignore_when_parsing=True,
    )

    ERA5_10 = ModalitySpec(
        name="era5_10",
        # 9 km/pixel bands that we store at 2.56 km/pixel.
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "2m-temperature",
                    "2m-dewpoint-temperature",
                    "surface-pressure",
                    "10m-u-component-of-wind",
                    "10m-v-component-of-wind",
                    "total-precipitation",
                ],
                4096,
            ),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
        image_tile_size_factor=-256,
    )

    LATLON = ModalitySpec(
        name="latlon",
        tile_resolution_factor=0,
        band_sets=[BandSet(["lat", "lon"], 0)],
        is_multitemporal=False,
        ignore_when_parsing=True,
    )

    GSE = ModalitySpec(
        name="gse",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [f"A{idx:02d}" for idx in range(64)],
                16,
            ),
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    CDL = ModalitySpec(
        name="cdl",
        tile_resolution_factor=16,
        band_sets=[BandSet(["cdl"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    WORLDPOP = ModalitySpec(
        name="worldpop",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    WRI_CANOPY_HEIGHT_MAP = ModalitySpec(
        name="wri_canopy_height_map",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    NDVI = ModalitySpec(
        name="ndvi",
        tile_resolution_factor=16,
        band_sets=[BandSet(["ndvi"], 16)],
        is_multitemporal=True,
        ignore_when_parsing=True,  # computed from S2 L2A, not loaded from file
    )

    EUROCROPS = ModalitySpec(
        name="eurocrops",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    ##############################################################################
    ## add more modalities here
    ###############################################################################
    RGB = ModalitySpec(
        name="rgb",
        tile_resolution_factor=4,
        band_sets=[BandSet(["B", "G", "R"], 4),
                BandSet(["NIR"], 4)                   
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
        image_tile_size_factor=4,
    )

    SAR = ModalitySpec(
        name="sar",
        tile_resolution_factor=4,
        band_sets=[BandSet(["sar"], 4)                   
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
        image_tile_size_factor=4,
    )

    LANDCOVER_1M = ModalitySpec(
        name="landcover_1m",
        tile_resolution_factor=16,
        band_sets=[BandSet(["landcover_1m"], 16)                   
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
        image_tile_size_factor=1,
    )

    LANDCOVER_30M = ModalitySpec(
        name="landcover_30m",
        tile_resolution_factor=16,
        band_sets=[BandSet(["landcover_30m"], 16)                   
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
        image_tile_size_factor=1,
    )

    LT1 = ModalitySpec(
        name="lt1",
        tile_resolution_factor=4,
        band_sets=[BandSet(["lt1"], 4)],
        is_multitemporal=True,
        ignore_when_parsing=False,
        image_tile_size_factor=4,
    )


    @classmethod
    def get(self, name: str) -> ModalitySpec:
        """Get the ModalitySpec with the specified name."""
        modality = getattr(Modality, name.upper())
        assert modality.name == name
        return modality

    @classmethod
    def values(self) -> list[ModalitySpec]:
        """Get all of the ModalitySpecs."""
        modalities = []
        for k in dir(Modality):
            modality = getattr(Modality, k)
            if not isinstance(modality, ModalitySpec):
                continue
            modalities.append(modality)
        return modalities

    @classmethod
    def names(self) -> list[str]:
        """Get all of the modality names."""
        return [modality.name for modality in self.values()]


# Latlon and timestamps
LATLON = ["lat", "lon"]
TIMESTAMPS = ["day", "month", "year"]


def get_modality_specs_from_names(names: list[str]) -> list[ModalitySpec]:
    """Get the modality specs from the names."""
    return [Modality.get(name) for name in names]
