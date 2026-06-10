"""Trying to prototype fitting everything into olmo core."""

import logging

from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

import os

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
            "--dataset.h5py_dir=/mnt/ht2-nas2/QH_Group/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/3996",
            "--data_loader.global_batch_size=640",
            "--trainer.max_duration.value=1",
            "--data_loader.num_workers=0",
            "--trainer.callbacks.wandb.enabled=False",
        ]
    
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )
