# #设置SLURM伪环境变量
# export SLURM_PROCID=0
# export SLURM_LOCALID=0
# export SLURM_NTASKS=1
# export SLURM_JOB_ID=12345
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29500

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# # GPU debugging version with minimal resources
# torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29501 main_pretrain.py \
# --data_mode webdataset \
# --trainshards /mnt/ht2-nas2/00-model/Copernicus_Zhejiang_Split/*_split_*.tar \
# --dataset_size 65 \
# --shuffle 10 \
# --output_dir ./checkpoints \
# --log_dir ./checkpoints/log \
# --model mae_vit_base_patch16 \
# --norm_pix_loss \
# --mask_ratio 0.75 \
# --num_workers 2 \
# --batch_size 16 \
# --epochs 1000 \
# --warmup_epochs 40 \
# --blr 1e-4 \
# --weight_decay 0.05 \
# --distill_size base
# # --dist_url $dist_url \
# # --dist_backend 'nccl' \


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
mkdir -p /mnt/si000523ygkv/00-model/Copernicus_zhejiang/checkpoints/20260609/log

SHARDS=$(ls /mnt/si000523ygkv/EO_Pretrian_Data_zhejiang_Copernicus_fmt/zhejiang_split/*_split_*.tar | tr '\n' ' ')
# GPU debugging version with minimal resources
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK  --master_addr=$MASTER_ADDR --nproc_per_node=$TQ_GPU_NUM --master_port $MASTER_PORT main_pretrain.py \
--data_mode webdataset \
--trainshards $SHARDS \
--dataset_size 1224 \
--shuffle 8 \
--output_dir /mnt/si000523ygkv/00-model/Copernicus_zhejiang/checkpoints/20260609 \
--log_dir /mnt/si000523ygkv/00-model/Copernicus_zhejiang/checkpoints/20260609/log \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--num_workers 2 \
--batch_size 8 \
--epochs 1000 \
--warmup_epochs 30 \
--blr 1e-4 \
--weight_decay 0.05 \
--distill_size base
# --dist_url $dist_url \
# --dist_backend 'nccl' \

