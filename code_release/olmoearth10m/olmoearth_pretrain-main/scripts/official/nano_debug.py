"""Trying to prototype fitting everything into olmo core."""

import logging
import os
import sys
# os.environ['PYTHONPATH']="/mnt/ht2-nas2/00-model/00-limx/Codes/olmoearth_pretrain-main_10m:{os.environ.get('PYTHONPATH','')}"
sys.path.insert(0, "/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/olmoearth_pretrain-main")
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
    import sys
    import torch

    if torch.cuda.is_available():
        torch.cuda.init()
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("CUDA is not available; training will run on CPU")
    
    
    # 调试模式：硬编码参数
    debug_mode = True  # 调试时设为 True，正常训练时改为 False
    
    if debug_mode:
        # 直接设置 sys.argv
        sys.argv = [
            "scripts/official/nano.py",
            "train_single",
            "debug_run_new_3", 
            "local",
            "--dataset.h5py_dir=/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/245",
            "--data_loader.global_batch_size=64",
            "--trainer.max_duration.value=10",
            "--trainer.load_strategy=if_available",
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
