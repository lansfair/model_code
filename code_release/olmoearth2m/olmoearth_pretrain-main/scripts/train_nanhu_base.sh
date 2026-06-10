echo "初始化conda环境："
export PATH="/root/miniconda3/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
conda init
source /root/miniconda3/bin/activate
conda activate olmoearth
echo "初始化结束"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "TQ_GPU_NUM: $TQ_GPU_NUM"
echo "MASTER_ADDR: $MASTER_ADDR"
printenv 
sleep 15s

cp -r /mnt/ht2-nas2/00-model/share/cartopy ~/.local/share/
ll ~/.local/share/cartopy/shapefiles/natural_earth/physical/ 

OLMOEARTHPATH="/mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/olmoearth_pretrain-main"
export OLMO_SHARED_FS=1
export NO_ALBUMENTATIONS_UPDATE=1
TRAIN_NAME="split_debug_nanhu_base1"
export PYTHONPATH=${OLMOEARTHPATH}:$PYTHONPATH
export WANDB_API_KEY="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"
export PYTHONWARNINGS="ignore::UserWarning"
cd ${OLMOEARTHPATH}
# rm -r ./local_output/checkpoints/anonymous/${TRAIN_NAME}
mkdir -p /mnt/si000523ygkv/00-model/olmoearth2m/logs
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK  --master_addr=$MASTER_ADDR --nproc_per_node=$TQ_GPU_NUM --master_port $MASTER_PORT \
scripts/official/base.py train ${TRAIN_NAME} local \
--dataset.h5py_dir=/mnt/ht2-nas2/00-model/guantp/olmoearth/code_release/olmoearth2m/dataset/landcover_1m_landcover_30m_lt1_rgb_sar_srtm_worldcereal_worldcover/12393 \
--data_loader.num_workers=8 \
--data_loader.prefetch_factor=2 \
--data_loader.token_budget=2250 \
--train_module.rank_microbatch_size=16 \
--data_loader.global_batch_size=1024 \
--trainer.callbacks.wandb.enabled=True \
--trainer.max_duration='{"unit": "epochs", "value": 300}' \
--common.save_folder="/mnt/si000523ygkv/00-model/olmoearth2m/logs/${TRAIN_NAME}"
# --trainer.callbacks.checkpointer.save_interval=5000 \
# --trainer.callbacks.checkpointer.ephemeral_save_interval=250