"""Trying to prototype fitting everything into olmo core."""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 强制重新加载模块（避免缓存问题）
import importlib
import olmoearth_pretrain.data.constants
importlib.reload(olmoearth_pretrain.data.constants)

import logging

from scripts.official.script import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components for nano_jzf experiment with custom modalities."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    # 在这里添加你想要的模态，只对 nano_jzf 生效
    config.training_modalities = [
        # Modality.SENTINEL2_L2A.name,
        # Modality.SENTINEL1.name,
        # Modality.LANDSAT.name,
        # Modality.WORLDCOVER.name,
        # Modality.SRTM.name,
        # Modality.OPENSTREETMAP_RASTER.name,
        # Modality.WRI_CANOPY_HEIGHT_MAP.name,
        # Modality.CDL.name,
        # Modality.WORLDCEREAL.name,
        Modality.PLANET_RGBNIR.name,
        # 在这里添加新的模态，例如：
        # Modality.YOUR_NEW_MODALITY.name,
    ]
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS["nano"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_linear_patch_embed=False,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


# if __name__ == "__main__":
#     main(
#         common_components_builder=build_common_components,
#         model_config_builder=build_model_config,
#         train_module_config_builder=build_train_module_config,
#         dataset_config_builder=build_dataset_config,
#         dataloader_config_builder=build_dataloader_config,
#         trainer_config_builder=build_trainer_config,
#     )


if __name__ == "__main__":
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
    # 调试模式：硬编码参数
    debug_mode = True  # 调试时设为 True，正常训练时改为 False
    
    if debug_mode:
        # 直接设置 sys.argv
        sys.argv = [
            "scripts/official/nano.py",
            "train_single",
            "debug_run_new", 
            "local",
            "--dataset.h5py_dir=/mnt/ht2-nas2/00-model/00-jiangzf/coderepo/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/planet_rgbnir/3996",
            "--data_loader.global_batch_size=640",
            "--trainer.max_duration.value=1",
            "--data_loader.num_workers=0",
            "--trainer.callbacks.wandb.enabled=False",
            "--trainer.load_strategy=never",
        ]
    
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )
