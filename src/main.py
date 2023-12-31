# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/main.py

from argparse import ArgumentParser
from warnings import simplefilter
import json
import os
import random
import sys
import tempfile

from torch.multiprocessing import Process
import torch
import torch.multiprocessing as mp

import config
import loader
import utils.log as log
import utils.misc as misc

RUN_NAME_FORMAT = ("{data_name}-" "{framework}-" "{phase}-" "{timestamp}")


def load_configs_initialize_training():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--entity", type=str, default=None, help="entity for wandb logging")
    parser.add_argument("--project", type=str, default=None, help="project name for wandb logging")

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/CIFAR10/ContraGAN.yaml")
    parser.add_argument("-data", "--data_dir", type=str, default=None)
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument("-ckpt", "--ckpt_dir", type=str, default=None)
    parser.add_argument("-best", "--load_best", action="store_true", help="load the best performed checkpoint")

    parser.add_argument("--seed", type=int, default=-1, help="seed for generating random numbers")
    parser.add_argument("-DDP", "--distributed_data_parallel", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl", help="cuda backend for DDP training \in ['nccl', 'gloo']")
    parser.add_argument("-tn", "--total_nodes", default=1, type=int, help="total number of nodes for training")
    parser.add_argument("-cn", "--current_node", default=0, type=int, help="rank of the current node")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("-sync_bn", "--synchronized_bn", action="store_true", help="turn on synchronized batchnorm")
    parser.add_argument("-mpc", "--mixed_precision", action="store_true", help="turn on mixed precision training")

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-empty_cache", "--empty_cache", action="store_true", help="empty cuda caches after training step of generator and discriminator, \
                        slightly reduces memory usage but slows training speed. (not recommended for normal use)")

    parser.add_argument("--print_every", type=int, default=100, help="logging interval")
    parser.add_argument("-every", "--save_every", type=int, default=2000, help="save interval")

    parser.add_argument("--dset_used", type=float, default=1.0, help="size of the training dataset, if less than 1 then it will be the fraction of the dataset used, \
                        if greater than 1 then it will be the number of elements to be used.")
    parser.add_argument("--oversample", action="store_true", help="oversample dataset.")

    args = parser.parse_args()
    run_cfgs = vars(args)

    if not args.train and \
            "none" in args.eval_metrics and \
            not args.save_real_images and \
            not args.save_fake_images and \
            not args.vis_fake_images and \
            not args.k_nearest_neighbor and \
            not args.interpolation and \
            not args.frequency_analysis and \
            not args.tsne_analysis and \
            not args.intra_class_fid and \
            not args.GAN_train and \
            not args.GAN_test and \
            not args.semantic_factorization:
        parser.print_help(sys.stderr)
        sys.exit(1)

    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()

    cfgs = config.Configurations(args.cfg_file)
    cfgs.update_cfgs(run_cfgs, super="RUN")
    cfgs.OPTIMIZATION.world_size = gpus_per_node * cfgs.RUN.total_nodes
    cfgs.check_compatability()

    run_name = log.make_run_name(RUN_NAME_FORMAT,
                                 data_name=cfgs.DATA.name,
                                 framework=cfgs.RUN.cfg_file.split("/")[-1][:-5],
                                 phase="train")

    misc.prepare_folder(names=cfgs.MISC.base_folders, save_dir=cfgs.RUN.save_dir)
    misc.download_data_if_possible(data_name=cfgs.DATA.name, data_dir=cfgs.RUN.data_dir)

    if cfgs.RUN.seed == -1:
        cfgs.RUN.seed = random.randint(1, 4096)
        cfgs.RUN.fix_seed = False
    else:
        cfgs.RUN.fix_seed = True

    if cfgs.OPTIMIZATION.world_size == 1:
        print("You have chosen a specific GPU. This will completely disable data parallelism.")
    return cfgs, gpus_per_node, run_name, rank

if __name__ == "__main__":
    cfgs, gpus_per_node, run_name, rank = load_configs_initialize_training()

    if cfgs.RUN.distributed_data_parallel and cfgs.OPTIMIZATION.world_size > 1:
        mp.set_start_method("spawn", force=True)
        print("Train the models through DistributedDataParallel (DDP) mode.")
        try:
            torch.multiprocessing.spawn(fn=loader.load_worker,
                                        args=(cfgs,
                                              gpus_per_node,
                                              run_name),
                                        nprocs=gpus_per_node)
        except KeyboardInterrupt:
            misc.cleanup()
    else:
        loader.load_worker(local_rank=rank,
                           cfgs=cfgs,
                           gpus_per_node=gpus_per_node,
                           run_name=run_name)