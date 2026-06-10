OLMOEARTHPATH="/mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/olmoearth_pretrain-main"
export NO_ALBUMENTATIONS_UPDATE=1
TRAIN_NAME="split_test"
export PYTHONPATH=${OLMOEARTHPATH}:$PYTHONPATH
export WANDB_API_KEY="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"
# export PYTHONWARNINGS="ignore::UserWarning"
cd ${OLMOEARTHPATH}
rm -r ./local_output/checkpoints/anonymous/${TRAIN_NAME}
# rm /mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/loss.csv
# CUDA_VISIBLE_DEVICES=1 torchrun --master_port 29502
# python scripts/official/nano.py train_single ${TRAIN_NAME} local \
# CUDA_VISIBLE_DEVICES=0,1,2,3 
CUDA_VISIBLE_DEVICES=0,1,2,4 torchrun --nproc_per_node=4 --master_port 29502 scripts/official/nano.py train ${TRAIN_NAME} local \
--dataset.h5py_dir=/mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/dataset/landcover_1m_landcover_30m_lt1_rgb_sar_srtm_worldcereal_worldcover/12393 \
--train_module.rank_microbatch_size=32 \
--data_loader.global_batch_size=256 \
--data_loader.num_workers=8 \
--trainer.callbacks.wandb.enabled=True \
--trainer.max_duration='{"unit": "epochs", "value": 500}'
# --trainer.callbacks.checkpointer.save_interval=50 \
# --trainer.callbacks.checkpointer.ephemeral_save_interval=25
