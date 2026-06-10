 # Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Yi Wang.
# All rights reserved.

# This code is adapted from https://github.com/facebookresearch/mae, which is 
# licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file except in compliance with the License.
# --------------------------------------------------------
import os
print("CUDA_VISIBLE_DEVICES",os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
print("Visible GPUs",torch.cuda.device_count())

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch
import models_mae_cfm as models_mae

import webdataset as wds
import random
import kornia
from kornia.augmentation import AugmentationSequential
from kornia.augmentation import AugmentationBase2D

from datetime import date

# import wandb
from util.pos_embed import interpolate_pos_embed

import glob

# 在 get_args_parser 之前或内部
# 拆分后的 tar 文件目录
base_path = "/mnt/ht2-nas2/00-model/Copernicus_Zhejiang_Split/"
# 匹配所有拆分后的 tar 文件 (example-*_split_*.tar)
all_tars = sorted(glob.glob(f"{base_path}/*_split_*.tar"))

def get_args_parser():
    parser = argparse.ArgumentParser('Copernicus-FM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_mode', default='webdataset', type=str,)
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    # webdataset
    # parser.add_argument('--trainshards', default='/mnt/qh2-nas3/EO_Pretrian_Data_zhejiang_Corpernius_fmt/test5/example-{000000..000000}.tar', nargs='+',
    #                     help='path/URL for dataset shards')
    parser.add_argument('--trainshards', default=all_tars, nargs='+',
                        help='path/URL for dataset shards')
    parser.add_argument('--shuffle', default=10, type=int,
                        help='webdataset shuffle buffer size')
    parser.add_argument('--dataset_size', default=65, type=int,
                        help='dataset size for webdataset')
    
    # training parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')

    # copernicusfm specific
    parser.add_argument('--drop_prob', default=0, type=float,
                        help='drop meta probability')
    parser.add_argument('--proj_dim', default=4096, type=int,
                        help='projection dimension')
      
    parser.add_argument('--var_option', default='language', type=str,
                        help='variable patch embed option') # spectrum* / language                         

    parser.add_argument('--pos_option', default='lonlat', type=str,
                        help='coord system of geolocation') # lonlat* / cartesian
    parser.add_argument('--time_option', default='absolute', type=str,
                        help='time format absolute or day of the year') # absolute* / dayofyear
    parser.add_argument('--scale_option', default='rawarea', type=str,
                        help='area of the original patch or augmented') # rawarea / augarea*

    parser.add_argument('--distill_size', default='base', type=str,
                        help='distill dinov2 backbone size') # base / large / giant / none

    # parser.add_argument('--continuous', default=None, type=str,
    #                     help='continuous pretrain checkpoint path')


    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    # If using CPU, ensure no CUDA operations
    if device.type == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Running on CPU - CUDA disabled")

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Only set cudnn benchmark for GPU training
    if device.type != 'cpu':
        cudnn.benchmark = True

    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    ## dataset preparation
    print("Loading data")
    # stats
    # TODO
    # stats
    S1_MEAN = [-11.44, -19.97]
    S1_STD = [6.44, 9.50]

    S2C_MEAN = [715.75, 790.08, 989.12, 907.87, 1255.85, 1981.60, 2243.83, 2318.84, 2376.60, 2368.32, 1625.88, 1063.44]
    S2C_STD = [664.54, 684.60, 672.75, 727.27, 686.61, 1031.59, 1210.08, 1276.39, 1332.25, 1390.04, 948.73, 743.08]

    # S3_OLCI: 原始22通道，去掉 index 0,9,21（无效通道），保留19个有效通道
    S3_OLCI_SCALE = [0.0133873, 0.0121481, 0.0115198, 0.0100953, 0.0123538,0.00879161, 0.00876539, 0.0095103, 0.00675523, 
    0.0071996, 0.00749684, 0.0086512, 0.00526779, 0.00530267, 0.00493004, 0.00549962,  0.00502847, 0.00326378,  0.00324118,]
    S3_OLCI_MEAN = [0] * 19
    S3_OLCI_STD = [1] * 19

    DEM_MEAN = [250.86]
    DEM_STD = [299.92]

    # ziyou_guang: 4通道 (B, G, R, NIR)
    ziyou_guang_MEAN = [67.31, 67.06, 59.80, 95.08]
    ziyou_guang_STD = [44.84, 46.54, 48.73, 62.64]

    # RGB: 从 ziyou_guang 提取 (R, G, B) 顺序
    RGB_MEAN = [59.80, 67.06, 67.31]   # R, G, B
    RGB_STD = [48.73, 46.54, 44.84]

    ziyou_sar_1_MEAN = [60.32]
    ziyou_sar_1_STD = [49.66]
    ziyou_sar_2_MEAN = [61.99]
    ziyou_sar_2_STD = [50.71]

    # data augmentation
    class MultiplyScale(AugmentationBase2D):
        def __init__(self, multipliers):
            super().__init__(p=1.0)  # always gets applied
            C = multipliers.shape[0]
            self.multipliers = multipliers.view(1, C, 1, 1)  # Shape for broadcasting
    
        def apply_transform(self, input, params, flags, transform=None):
            # Apply the multiplication
            out =  input * self.multipliers.to(input.device)
            return out
        
    class RemoveOutlier(AugmentationBase2D):
        def __init__(self):
            super().__init__(p=1.0)  # always gets applied
    
        def apply_transform(self, input, params, flags, transform=None):
            # nan to 0
            input[torch.isnan(input)] = 0
            # inf to 0
            input[torch.isinf(input)] = 0
            return input
#TODO
    transform_s1 = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((112, 112), scale=(0.1, 0.4)),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.Normalize(mean=torch.tensor(S1_MEAN), std=torch.tensor(S1_STD)),
        RemoveOutlier()
    )
    transform_s2 = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((112, 112), scale=(0.1, 0.4)),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.Normalize(mean=torch.tensor(S2C_MEAN), std=torch.tensor(S2C_STD)),
        RemoveOutlier()
    )
    transform_rgb = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((560, 560), scale=(0.1, 0.4)),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.Normalize(mean=torch.tensor(RGB_MEAN), std=torch.tensor(RGB_STD)),
        RemoveOutlier()
    )
    transform_s3 = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((96, 96), scale=(0.2, 1.0)),
        kornia.augmentation.RandomHorizontalFlip(),
        MultiplyScale(torch.tensor(S3_OLCI_SCALE)),
        kornia.augmentation.Normalize(mean=torch.tensor(S3_OLCI_MEAN), std=torch.tensor(S3_OLCI_STD)),
        RemoveOutlier()
    )
    transform_s5p = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((28, 28), scale=(0.2, 1.0)),
        kornia.augmentation.RandomHorizontalFlip(),
        RemoveOutlier()
    )
    transform_dem = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((960, 960), scale=(0.2, 1.0)),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.Normalize(mean=torch.tensor(DEM_MEAN), std=torch.tensor(DEM_STD)),
        RemoveOutlier()
    )
    transform_guang = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((560, 560), scale=(0.1, 0.4)),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.Normalize(mean=torch.tensor(ziyou_guang_MEAN), std=torch.tensor(ziyou_guang_STD)),
        RemoveOutlier()
    )
    transform_sar1 = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((560, 560), scale=(0.1, 0.4)),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.Normalize(mean=torch.tensor(ziyou_sar_1_MEAN), std=torch.tensor(ziyou_sar_1_STD)),
        RemoveOutlier()
    )
    transform_sar2 = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((560, 560), scale=(0.1, 0.4)),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.Normalize(mean=torch.tensor(ziyou_sar_2_MEAN), std=torch.tensor(ziyou_sar_2_STD)),
        RemoveOutlier()
    )    


    train_transforms = {
        "s1_grd": transform_s1,
        "s2_toa": transform_s2,
        "s3_olci": transform_s3,
        "s5p_co": transform_s5p,
        "s5p_no2": transform_s5p,
        "s5p_o3": transform_s5p,
        "s5p_so2": transform_s5p,
        "dem": transform_dem,
        "guang":transform_guang,
        "sar1":transform_sar1,
        "sar2":transform_sar2,
        "rgb": transform_rgb,
    }

    # webdataset
    reference_date = date(1970, 1, 1)
    reorganize_meta = create_meta_organizer(args,reference_date)
    if args.data_mode == 'webdataset':
        world_size = misc.get_world_size()
        rank = misc.get_rank()

        # 先展开 brace pattern 成列表，再按 rank 分片
        # args.trainshards 可能是列表（来自 glob）或字符串（brace pattern）
        if isinstance(args.trainshards, list):
            shards_list = args.trainshards
        else:
            shards_list = list(wds.shardlists.expand_urls(args.trainshards))

        # 分布式训练：预先按 rank 切分 shards
        if world_size > 1:
            my_shards = shards_list[rank::world_size]
        else:
            my_shards = shards_list

        # 空的 nodesplitter（因为已经预切分好了）
        def identity_nodesplitter(urls):
            return urls

        dataset_train = (
            wds.WebDataset(
                my_shards,
                resampled=False,
                shardshuffle=False,
                empty_check=False,
                nodesplitter=identity_nodesplitter,
            )
            .shuffle(args.shuffle)
            .decode()
            .select(has_all_modalities)
            .map(sample_one_local_patch)
            .map(sample_one_time_stamp)
            .map(reorganize_meta)
            .to_tuple("s1_grd.pth", "s2_toa.pth", "s3_olci.pth", "s5p_co.pth", "s5p_no2.pth", "s5p_o3.pth", "s5p_so2.pth", "dem.pth", "ziyou_sar_1.pth","ziyou_sar_2.pth","ziyou_guang.pth","json" )
        )

        # 分布式时用 num_workers=0 避免多进程冲突
        num_workers = 0 if world_size > 1 else args.num_workers
        data_loader_train = wds.WebLoader(dataset_train, batch_size=None, num_workers=num_workers)

        data_loader_train = data_loader_train.shuffle(args.shuffle).batched(args.batch_size)

        # # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
        number_of_batches = args.dataset_size // (args.batch_size * args.world_size)
        #print(args.dataset_size, args.batch_size, args.world_size, number_of_batches)
        data_loader_train = data_loader_train.slice(number_of_batches)
        data_loader_train = data_loader_train.with_length(number_of_batches)
        data_loader_train = data_loader_train.with_epoch(number_of_batches)


    ## model
    # TODO
    img_size={'s1_grd':112,'s2_toa':112,'s3_olci':96,'s5p_co':28,'s5p_no2':28,'s5p_o3':28,'s5p_so2':28,'dem':960,'guang':560,'sar1':560,'sar2':560}
    patch_size = {'s1_grd':8,'s2_toa':8,'s3_olci':8,'s5p_co':4,'s5p_no2':4,'s5p_o3':4,'s5p_so2':4,'dem':48,'guang':16,'sar1':16,'sar2':16}
    in_chans = {'s1_grd':2,'s2_toa':12,'s3_olci':19,'s5p_co':1,'s5p_no2':1,'s5p_o3':1,'s5p_so2':1,'dem':1,'guang':4,'sar1':1,'sar2':1}

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=112, patch_size=8, in_chans=None,
        var_option=args.var_option,
        pos_option=args.pos_option, time_option=args.time_option, scale_option=args.scale_option,
        distill_size=args.distill_size
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    

    # following timm: set wd as 0 for bias and norm layers
    #param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay) # timm 0.3.2
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.data_mode != 'webdataset':
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args, train_transforms=train_transforms
        )
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
        # if args.output_dir: #and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        # wandb
        # if misc.is_main_process():
        #     wandb.log(log_stats)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def has_all_modalities(sample):
    required_keys = [
        "s1_grd.pth", 
        "s2_toa.pth", 
        "s3_olci.pth", 
        "s5p_co.pth", 
        "s5p_no2.pth",
        "s5p_o3.pth",
        "s5p_so2.pth",
        "dem.pth",
        "ziyou_guang.pth",
        "ziyou_sar_1.pth",
        "ziyou_sar_2.pth",
        "json"
    ]
    return all(key in sample for key in required_keys)

# TODO
def sample_one_local_patch(sample):
    s1 = sample["s1_grd.pth"]
    s2 = sample["s2_toa.pth"]
    guang = sample["ziyou_guang.pth"]
    sar1 = sample["ziyou_sar_1.pth"]
    sar2 = sample["ziyou_sar_2.pth"]
    meta_s1 = sample["json"]["s1_grd"]
    meta_s2 = sample["json"]["s2_toa"]
    meta_guang = sample["json"]["ziyou_guang"]
    meta_sar1  = sample["json"]["ziyou_sar_1"]
    meta_sar2  = sample["json"]["ziyou_sar_2"]

    # 各模态独立随机选空间块（第一维）
    idx_s1 = random.randint(0, s1.shape[0]-1)
    idx_s2 = random.randint(0, s2.shape[0]-1)
    idx_guang = random.randint(0, guang.shape[0]-1)
    idx_sar1 = random.randint(0, sar1.shape[0]-1)
    idx_sar2 = random.randint(0, sar2.shape[0]-1)

    s1_new = s1[idx_s1]   # [T, C, H, W]
    s2_new = s2[idx_s2]
    guang_new = guang[idx_guang]
    sar1_new = sar1[idx_sar1]
    sar2_new = sar2[idx_sar2]
    meta_s1_new = meta_s1[idx_s1]
    meta_s2_new = meta_s2[idx_s2]
    meta_guang_new = meta_guang[idx_guang]
    meta_sar1_new = meta_sar1[idx_sar1]
    meta_sar2_new = meta_sar2[idx_sar2]

    sample["s1_grd.pth"] = s1_new
    sample["s2_toa.pth"] = s2_new
    sample["ziyou_guang.pth"] = guang_new
    sample["ziyou_sar_1.pth"] = sar1_new
    sample["ziyou_sar_2.pth"] = sar2_new
    sample["json"]["s1_grd"] = meta_s1_new
    sample["json"]["s2_toa"] = meta_s2_new
    sample["json"]["ziyou_guang"] = meta_guang_new
    sample["json"]["ziyou_sar_1"] = meta_sar1_new
    sample["json"]["ziyou_sar_2"] = meta_sar2_new
    return sample

def sample_one_time_stamp(sample):
    for key in sample:
        if key.endswith('.pth') and key != 'dem.pth':
            data = sample[key]
            idx = random.randint(0, data.shape[0]-1)
            sample[key] = data[idx] # CxHxW

            if 's3_olci' in key: # 取前21通道，去掉第1(index 0)和第10(index 9)，保留19通道
                data = sample[key][:21]
                keep_indices = list(range(1, 9)) + list(range(10, 21))
                sample[key] = data[keep_indices]

            json_key = key.replace('.pth', '')
            meta = sample["json"][json_key]
            sample["json"][json_key] = meta[idx]
    sample['dem.pth'] = sample['dem.pth'].unsqueeze(0)
    sample["json"]["dem"] = sample["json"]["dem"][0]
    return sample

def create_meta_organizer(args,reference_date):
    def reorganize_meta(sample): # assume one local patch and one time stamp
        meta = sample["json"]
        #meta_new = {}
        meta_new = torch.zeros(len(meta), 3, dtype=torch.float32)

        count = 0
        for key in meta.keys():
            entry = torch.zeros(3,dtype=torch.float32) # lon, lat, time
            meta_str = meta[key][0]
            # print(f"key={key}, type(meta[key])={type(meta[key])}, meta[key]={meta[key]}")  # 调试
            if 'dem' in key:
                coords_str = meta_str.split('_')
                lon = float(coords_str[1])
                lat = float(coords_str[2])
                entry[0] = lon
                entry[1] = lat
                if args.time_option=='absolute':
                    date_obj = date(2015, 1, 1)
                    delta = (date_obj - reference_date).days
                elif args.time_option=='dayofyear':
                    delta = torch.nan
                entry[2] = delta
            else:
                parts = meta_str.split('/')
                coords_str = parts[-2].split('_')
                lon = float(coords_str[1])
                lat = float(coords_str[2])
                entry[0] = lon
                entry[1] = lat
                date_str = parts[-1]
                date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                if args.time_option=='absolute':
                    delta = (date_obj - reference_date).days
                elif args.time_option=='dayofyear':
                    start_date = date(int(date_str[:4]),1,1)
                    delta = (date_obj - start_date).days
                entry[2] = delta
            
            meta_new[count] = entry
            count += 1
        sample["json"] = meta_new
        
        return sample
    return reorganize_meta


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # wandb init for the main device
    # if misc.is_main_process():
    #     wandb.init(
    #         entity='xxx', # wandb username
    #         project='copernicus-fm', 
    #         config=args.__dict__,
    #     )

    main(args)
