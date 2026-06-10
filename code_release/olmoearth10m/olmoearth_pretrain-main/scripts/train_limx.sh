# !/bin/bash
OLMOEARTHPATH="/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/olmoearth_pretrain-main"
export NO_ALBUMENTATIONS_UPDATE=1
TRAIN_NAME="dataset_debug_10m_limx-2"
PYTHONWARNINGS="ignore::UserWarning:class_registry.entry_points:7"
export PYTHONPATH=${OLMOEARTHPATH}:$PYTHONPATH
export WANDB_API_KEY="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"
# export WANDB_API_KEY="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"
cd ${OLMOEARTHPATH}
# rm -r ./local_output/checkpoints/anonymous/${TRAIN_NAME}
# CUDA_VISIBLE_DEVICES=1 torchrun --master_port 29502
# python scripts/official/nano.py train_single ${TRAIN_NAME} local \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 39503 scripts/official/nano.py train ${TRAIN_NAME} local \
--dataset.h5py_dir=/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/2205 \
--train_module.rank_microbatch_size=32 \
--data_loader.global_batch_size=64 \
--trainer.callbacks.wandb.enabled=True \
--trainer.callbacks.wandb.entity=mengxuanli28- \
--trainer.callbacks.wandb.project=olmoearth \
--trainer.load_strategy=if_available \
--trainer.callbacks.checkpointer.save_interval=50 \
--trainer.callbacks.checkpointer.ephemeral_save_interval=25 \
--pretrained_weight_path="/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/model_weights/OlmoEarth-v1-Nano/weights.pth"


# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 scripts/official/nano.py train dataset_debug_10m_limx local   --dataset.h5py_dir=/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/2205   --data_loader.global_batch_size=64 --pretrained_weight_path=/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/model_weights/OlmoEarth-v1-Nano/weights.pth