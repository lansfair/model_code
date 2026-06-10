OLMOEARTHPATH="/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/olmoearth_pretrain-main"
export NO_ALBUMENTATIONS_UPDATE=1
TRAIN_NAME="debug"
export PYTHONPATH=${OLMOEARTHPATH}:$PYTHONPATH
export WANDB_API_KEY="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"
export PYTHONWARNINGS="ignore::UserWarning"
cd ${OLMOEARTHPATH}
# rm -r ./local_output/checkpoints/anonymous/${TRAIN_NAME}
# rm /mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/loss.csv
# CUDA_VISIBLE_DEVICES=1 torchrun --master_port 29502
# python scripts/official/nano.py train_single ${TRAIN_NAME} local \
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port 29504 scripts/official/nano.py train ${TRAIN_NAME} local \
--dataset.h5py_dir=/mnt/ht2_nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/dataset/153 \
--train_module.rank_microbatch_size=32 \
--data_loader.global_batch_size=64 \
--trainer.max_duration='{"unit": "epochs", "value": 5000}' \
--trainer.callbacks.checkpointer.save_interval=50 \
--trainer.callbacks.checkpointer.ephemeral_save_interval=25
# --trainer.callbacks.wandb.enabled=False