OLMOEARTHPATH="/mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/olmoearth_pretrain-main"
export NO_ALBUMENTATIONS_UPDATE=1
TRAIN_NAME="dataset_debug_10m_4"
# rm -rf /mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/olmoearth_pretrain-main/local_output/checkpoints/anonymous/dataset_debug_10m_3
PYTHONWARNINGS="ignore::UserWarning"
export PYTHONPATH=${OLMOEARTHPATH}:$PYTHONPATH
export WANDB_API_KEY="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"
cd ${OLMOEARTHPATH}
# rm -r ./local_output/checkpoints/anonymous/${TRAIN_NAME}
# CUDA_VISIBLE_DEVICES=1 torchrun --master_port 29502
# python scripts/official/nano.py train_single ${TRAIN_NAME} local \
# /mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/2205
# /mnt/ht2_nas2/00-model/00-jiangzf/coderepo/H5_DIR/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/3996
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 39883 scripts/official/nano.py train ${TRAIN_NAME} local \
--dataset.h5py_dir=/mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/dataset/landcover_1m_landcover_30m_landsat_lt1_rgb_sar_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover/200 \
--train_module.rank_microbatch_size=16 \
--data_loader.global_batch_size=128 \
--trainer.callbacks.wandb.enabled=True \
--trainer.callbacks.wandb.entity=masonj-university-of-alberta \
--trainer.callbacks.wandb.project=olmoearth \
--trainer.load_strategy=if_available \
--data_loader.num_workers=0 \
--data_loader.prefetch_factor=0 \
--trainer.callbacks.checkpointer.save_interval=50 \
--trainer.callbacks.checkpointer.ephemeral_save_interval=25 \
--pretrained_weight_path="/mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth10m/model_weights/OlmoEarth-v1-Nano/weights.pth" \
--train_module.ema_decay="[0.99, 1.0]" \
--common.save_folder="/mnt/ht2-nas2/00-model/olmoearth/logs/${TRAIN_NAME}"