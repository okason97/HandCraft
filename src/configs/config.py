# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/config.py

from itertools import chain
import json
import os
import random
import sys
import yaml

import torch
import torch.nn as nn

import utils.misc as misc
import utils.losses as losses
import utils.ops as ops
import utils.optimizers as optimizers
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

class make_empty_object(object):
    pass


class Configurations(object):
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.load_base_cfgs()
        self._overwrite_cfgs(self.cfg_file)
        self.define_modules()

    def load_base_cfgs(self):
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = misc.make_empty_object()

        # dataset name \in ["MY_DATASET"]
        self.DATA.name = "LSFB"
        # input size for training
        self.DATA.input_size = [15, 32, 3]
        # number of classes in training dataset
        self.DATA.num_classes = 10
        # minimum number of samples per class in training dataset
        self.DATA.min_samples = 10
        # maximum number of frames
        self.DATA.max_len = 15
        self.DATA.target_len = 8
        self.DATA.pad_frames = True
        self.DATA.pad_mode = 'wrap'
        # oversample dataset to current maximum number of samples per class
        self.DATA.oversample = False
        # data poses and keypoints used
        self.DATA.poses = []
        self.DATA.synth_poses = [["pose","all"],["right_hand","all"],["left_hand","all"], ["face","all"]]
        # count of the subset of keypoints chosen
        self.DATA.num_keypoints = 91
        # data augmentation
        self.DATA.flip_p = 0.0
        self.DATA.scale = 0.0
        self.DATA.random_crop = False
        self.DATA.drop_frame = 0.0
        self.DATA.drop_keypoint = 0.0
        self.DATA.block_size = 5
        self.DATA.flip_p = 0.0
        self.DATA.scale = 0.0
        self.DATA.rot = 0.0
        # encode the data by applying a transformation ["DCT","DWT",None]
        self.DATA.transform = None
        self.DATA.batch_size = 128


        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = misc.make_empty_object()

        # type of backbone architectures of the generator and discriminator \in
        # ["conv1d"]
        self.MODEL.backbone = "conv1d"
        # whether to apply spectral normalization 
        self.MODEL.apply_sn = False
        # type of activation function \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
        self.MODEL.act_fn = "ReLU"
        # feature normalization method \in ["batchnorm", "layernorm", "slayernorm", "tlayernorm", None]
        self.MODEL.feature_norm = "batchnorm"
        # whether to apply transformer layers
        self.MODEL.apply_attn = True
        # use exponential moving average
        self.MODEL.apply_ema = False
        # ema parameters
        self.MODEL.ema_beta = 0.9999
        self.MODEL.ema_update_after_step = 10
        self.MODEL.ema_update_every = 1
        self.MODEL.ema_power = 0.9
        # transformer layer number of heads
        self.MODEL.nheads = 4
        # embeding size
        self.MODEL.embed_size = 64
        # bias for linear layers (MAMBA)
        self.MODEL.bias = True
        # class embeding size
        self.MODEL.class_emb_size = 32
        # class dropout probability for classifier-free guidance
        self.MODEL.class_dropout_prob = 0.0
        # base channel for the classifier architecture
        self.MODEL.conv_dim = 64
        # convolutional bias for the classifier architecture (MAMBA)
        self.MODEL.conv_bias = True
        # hidden dimension for the classifier architecture
        self.MODEL.hidden_dim = 64
        # hidden dimension for the classifier architecture
        self.MODEL.mlp_dim = 64
        # kernel size for the convolutions
        self.MODEL.k_size = 17
        # stride size for the convolutions
        self.MODEL.stride = 1
        # expand ratio of channels for conv1d block
        self.MODEL.expand_ratio = 2
        # classifier's depth
        self.MODEL.depth = 4
        # dropout ratio of the model layers
        self.MODEL.dropout = 0.8
        # drop path ratio of the model convolutional blocks
        self.MODEL.drop_path = 0.2
        # apply dropout when the training reaches the indicated epoch https://arxiv.org/abs/2303.01500
        self.MODEL.late_dropout = None
        self.MODEL.late_drop_path = None
        self.MODEL.late_dropout_step = None
        self.MODEL.late_drop_path_step = None
        # weight initialization method \in ["ortho", "N02", "glorot", "xavier"]
        self.MODEL.init = "ortho"
        # use temporal dimension for the input fully conected layer
        self.MODEL.temporal_fc_in = False
        # use temporal dimension for the output fully conected layer
        self.MODEL.temporal_fc_out = False
        self.MODEL.use_spatial_fc = True
        # use depthwise convolution
        self.MODEL.apply_dw_conv = False
        # generator noise value
        self.MODEL.noise_scale = 0.0
        # representation size for ViT output layers
        self.MODEL.representation_size = None
        # MAMBA
        # latent state dim
        self.MODEL.d_state = 16 
        # rank of Δ
        self.MODEL.dt_rank = 'auto'

        # -----------------------------------------------------------------------------
        # loss settings
        # -----------------------------------------------------------------------------
        self.LOSS = misc.make_empty_object()

        # type of loss \in ["CCE", "motion"]
        self.LOSS.loss_type = "CCE"
        # start iteration for EMALosses in src/utils/EMALosses
        self.LOSS.lecam_ema_start_iter = "N/A"
        # decay rate for the EMALosses
        self.LOSS.lecam_ema_decay = "N/A"
        # use relative motion for motion loss
        self.LOSS.relative_motion = True

        # -----------------------------------------------------------------------------
        # optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZATION = misc.make_empty_object()

        # type of the optimizer for training \in ["SGD", "RMSprop", "Adam", "RAdam"]
        self.OPTIMIZATION.type_ = "RAdam"
        # lr scheduler \in ["OneCycle"]
        self.OPTIMIZATION.lrscheduler = None
        self.OPTIMIZATION.max_lr = 0.1
        self.OPTIMIZATION.pct_start = 0.3
        # number of batch size for training,
        self.OPTIMIZATION.batch_size = 128
        # learning rate
        self.OPTIMIZATION.lr = 0.0002
        # weight decay strength
        self.OPTIMIZATION.weight_decay = 0.0
        # use lookahead
        self.OPTIMIZATION.lookahead = True
        # momentum value for SGD and RMSprop optimizers
        self.OPTIMIZATION.momentum = "N/A"
        # nesterov value for SGD optimizer
        self.OPTIMIZATION.nesterov = False
        # alpha value for RMSprop optimizer
        self.OPTIMIZATION.alpha = "N/A"
        # beta values for RAdam optimizer
        self.OPTIMIZATION.beta1 = 0.9
        self.OPTIMIZATION.beta2 = 0.999
        # the total number of steps for training
        self.OPTIMIZATION.total_steps = 10
        self.OPTIMIZATION.synth_total_steps = 1

        # -----------------------------------------------------------------------------
        # preprocessing settings
        # -----------------------------------------------------------------------------
        self.PRE = misc.make_empty_object()

        # whether to apply random flip preprocessing before training
        self.PRE.apply_rflip = True

        # -----------------------------------------------------------------------------
        # augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG = misc.make_empty_object()

        # -----------------------------------------------------------------------------
        # run settings
        # -----------------------------------------------------------------------------
        self.RUN = misc.make_empty_object()
        self.RUN.mixed_precision = False
        # classifier-free guidance scale (1 = regular conditional generation)
        self.RUN.guidance_scale = 1

        # -----------------------------------------------------------------------------
        # run settings
        # -----------------------------------------------------------------------------
        self.MISC = misc.make_empty_object()

        self.MISC.base_folders = ["checkpoints", "figures", "logs", "moments", "samples", "values"]

        # -----------------------------------------------------------------------------
        # Module settings
        # -----------------------------------------------------------------------------
        self.MODULES = misc.make_empty_object()

        self.super_cfgs = {
            "DATA": self.DATA,
            "MODEL": self.MODEL,
            "LOSS": self.LOSS,
            "OPTIMIZATION": self.OPTIMIZATION,
            "PRE": self.PRE,
            "AUG": self.AUG,
            "RUN": self.RUN,
        }

    def update_cfgs(self, cfgs, super="RUN"):
        for attr, value in cfgs.items():
            setattr(self.super_cfgs[super], attr, value)

    def _overwrite_cfgs(self, cfg_file):
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
            for super_cfg_name, attr_value in yaml_cfg.items():
                for attr, value in attr_value.items():
                    if hasattr(self.super_cfgs[super_cfg_name], attr):
                        setattr(self.super_cfgs[super_cfg_name], attr, value)
                    else:
                        raise AttributeError("There does not exist '{cls}.{attr}' attribute in the config.py.". \
                                             format(cls=super_cfg_name, attr=attr))

    def define_losses(self):
        losses_dic = {
            "CCE": nn.CrossEntropyLoss(),
            "motion": losses.MotionLoss(relative_motion=self.LOSS.relative_motion),
        }

        self.LOSS.loss = losses_dic[self.LOSS.loss_type]

    def define_modules(self):
        if not self.MODEL.use_spatial_fc:
            self.MODULES.glinear = ops.Temporal_FC
        else:
            self.MODULES.glinear = ops.Spatial_FC

        if self.MODEL.apply_sn:
            self.MODULES.linear = ops.snlinear
            if self.MODEL.apply_dw_conv:
                self.MODULES.conv1d = ops.sndwconv1d
            else:
                self.MODULES.conv1d = ops.snconv1d
        else:
            if self.MODEL.apply_dw_conv:
                self.MODULES.conv1d = ops.dwconv1d
            else:
                self.MODULES.conv1d = ops.conv1d
            self.MODULES.linear = ops.linear

        if self.MODEL.act_fn == "ReLU":
            self.MODULES.act_fn = nn.ReLU(inplace=True)
        elif self.MODEL.act_fn == "Leaky_ReLU":
            self.MODULES.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.MODEL.act_fn == "ELU":
            self.MODULES.act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.MODEL.act_fn == "GELU":
            self.MODULES.act_fn = nn.GELU()
        elif self.MODEL.act_fn == "Auto":
            pass
        else:
            raise NotImplementedError

        if self.MODEL.apply_attn:
            self.MODULES.transformer_layer = ops.transformer_layer

        if self.MODEL.feature_norm == "batchnorm":
            self.MODULES.feature_norm = ops.batchnorm
        elif self.MODEL.feature_norm == "layernorm":
            self.MODULES.feature_norm = ops.layernorm
        elif self.MODEL.feature_norm == "slayernorm":
            self.MODULES.feature_norm = ops.SLayerNorm
        elif self.MODEL.feature_norm == "tlayernorm":
            self.MODULES.feature_norm = ops.TLayerNorm
        else:
            self.MODULES.feature_norm = nn.Identity

        self.MODULES.dropout = ops.dropout
        self.MODULES.drop_path = ops.drop_path

        self.MODULES.eca = ops.eca

        self.MODULES.pooling = nn.AvgPool1d

        return self.MODULES

    def define_optimizer(self, model, epochs, len_dataloader):
        params = []
        for _, param in model.named_parameters():
            params.append(param)

        if self.OPTIMIZATION.type_ == "SGD":
            self.OPTIMIZATION.optimizer = torch.optim.SGD(params=params,
                                                            lr=self.OPTIMIZATION.lr,
                                                            weight_decay=self.OPTIMIZATION.weight_decay,
                                                            momentum=self.OPTIMIZATION.momentum,
                                                            nesterov=self.OPTIMIZATION.nesterov)
        elif self.OPTIMIZATION.type_ == "RMSprop":
            self.OPTIMIZATION.optimizer = torch.optim.RMSprop(params=params,
                                                                lr=self.OPTIMIZATION.lr,
                                                                weight_decay=self.OPTIMIZATION.weight_decay,
                                                                momentum=self.OPTIMIZATION.momentum,
                                                                alpha=self.OPTIMIZATION.alpha)
        elif self.OPTIMIZATION.type_ == "Adam":
            betas = [self.OPTIMIZATION.beta1, self.OPTIMIZATION.beta2]
            eps_ = 1e-6

            self.OPTIMIZATION.optimizer = torch.optim.Adam(params=params,
                                                           lr=self.OPTIMIZATION.lr,
                                                           betas=betas,
                                                           weight_decay=self.OPTIMIZATION.weight_decay,
                                                           eps=eps_)
        elif self.OPTIMIZATION.type_ == "RAdam":
            betas = [self.OPTIMIZATION.beta1, self.OPTIMIZATION.beta2]
            eps_ = 1e-6

            self.OPTIMIZATION.optimizer = torch.optim.RAdam(params=params,
                                                           lr=self.OPTIMIZATION.lr,
                                                           betas=betas,
                                                           weight_decay=self.OPTIMIZATION.weight_decay,
                                                           eps=eps_)
        else:
            raise NotImplementedError
        
        if self.OPTIMIZATION.lrscheduler == "OneCycle":
            self.OPTIMIZATION.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.OPTIMIZATION.optimizer, max_lr=self.OPTIMIZATION.max_lr, pct_start=self.OPTIMIZATION.pct_start, three_phase=False, steps_per_epoch=len_dataloader, epochs=epochs)
        else:
            self.OPTIMIZATION.scheduler = None

        if self.OPTIMIZATION.lookahead:
            self.OPTIMIZATION.optimizer = optimizers.Lookahead(optimizer=self.OPTIMIZATION.optimizer,k=5,alpha=0.5)

    """
    def define_augments(self, device):
        self.AUG.series_augment = misc.identity
        ada_augpipe = {
            'blit':   dict(xflip=1, rotate90=1, xint=1),
            'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise':  dict(noise=1),
            'cutout': dict(cutout=1),
            'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
            'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }
        if self.AUG.apply_diffaug:
            assert self.AUG.diffaug_type != "W/O", "Please select diffentiable augmentation type!"
            if self.AUG.diffaug_type == "cr":
                self.AUG.series_augment = cr.apply_cr_aug
            elif self.AUG.diffaug_type == "diffaug":
                self.AUG.series_augment = diffaug.apply_diffaug
            elif self.AUG.diffaug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.series_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug).train().to(device).requires_grad_(False)
            elif self.AUG.diffaug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]:
                self.AUG.series_augment = ada_aug.AdaAugment(**ada_augpipe[self.AUG.diffaug_type]).train().to(device).requires_grad_(False)
                self.AUG.series_augment.p = 1.0
            else:
                raise NotImplementedError

        if self.AUG.apply_ada:
            assert self.AUG.ada_aug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn",
                                             "bgcfnc"], "Please select ada supported augmentations"
            self.AUG.series_augment = ada_aug.AdaAugment(**ada_augpipe[self.AUG.ada_aug_type]).train().to(device).requires_grad_(False)

        if self.LOSS.apply_cr:
            assert self.AUG.cr_aug_type != "W/O", "Please select augmentation type for cr!"
            if self.AUG.cr_aug_type == "cr":
                self.AUG.parallel_augment = cr.apply_cr_aug
            elif self.AUG.cr_aug_type == "diffaug":
                self.AUG.parallel_augment = diffaug.apply_diffaug
            elif self.AUG.cr_aug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.parallel_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug).train().to(device).requires_grad_(False)
            elif self.AUG.cr_aug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]:
                self.AUG.parallel_augment = ada_aug.AdaAugment(**ada_augpipe[self.AUG.cr_aug_type]).train().to(device).requires_grad_(False)
                self.AUG.parallel_augment.p = 1.0
            else:
                raise NotImplementedError

        if self.LOSS.apply_bcr:
            assert self.AUG.bcr_aug_type != "W/O", "Please select augmentation type for bcr!"
            if self.AUG.bcr_aug_type == "bcr":
                self.AUG.parallel_augment = cr.apply_cr_aug
            elif self.AUG.bcr_aug_type == "diffaug":
                self.AUG.parallel_augment = diffaug.apply_diffaug
            elif self.AUG.bcr_aug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.parallel_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug).train().to(device).requires_grad_(False)
            elif self.AUG.bcr_aug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]:
                self.AUG.parallel_augment = ada_aug.AdaAugment(
                    **ada_augpipe[self.AUG.bcr_aug_type]).train().to(device).requires_grad_(False)
                self.AUG.parallel_augment.p = 1.0
            else:
                raise NotImplementedError
    """

    def check_compatability(self):

        if not self.RUN.train:
            assert self.RUN.ckpt_dir is not None, "Specify -ckpt CHECKPOINT_FOLDER to evaluate without training."

        if self.RUN.twin_generator:
            assert self.RUN.r_ckpt_dir is not None, "Specify -r_ckpt CHECKPOINT_FOLDER to evaluate with two generators."

        if self.RUN.distributed_data_parallel:
            print("Turning on DDP might cause inexact evaluation results. \
                \nPlease use a single GPU or DataParallel for the exact evluation.")

        if self.OPTIMIZATION.world_size == 1:
            assert not self.RUN.distributed_data_parallel, "Cannot perform distributed training with a single gpu."

        assert self.OPTIMIZATION.batch_size % self.OPTIMIZATION.world_size == 0, \
            "Batch_size should be divided by the number of gpus."

        assert self.RUN.save_every % self.RUN.print_every == 0, \
            "RUN.save_every should be divided by RUN.print_every for wandb logging."
        
        assert self.RUN.ss_num <= self.OPTIMIZATION.batch_size, "Number of saved samples should be smaller than the batch size."