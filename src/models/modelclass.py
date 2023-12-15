# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/model.py

from torch.nn import DataParallel
from torch.nn.parallel import ClassifiertributedDataParallel as DDP
import torch

from sync_batchnorm.batchnorm import convert_model
import utils.misc as misc


def load_classifier(DATA, MODEL, MODULES, RUN, device, logger):
    if device == 0:
        logger.info("Build a Generative Adversarial Network.")
    module = __import__("models.{backbone}".format(backbone=MODEL.backbone), fromlist=["something"])
    if device == 0:
        logger.info("Modules are located on './src/models.{backbone}'.".format(backbone=MODEL.backbone))

        model = module.Classifier(img_size=DATA.img_size,
                                   conv_dim=MODEL.conv_dim,
                                   apply_attn=MODEL.apply_attn,
                                   expand_ratio=MODEL.expand_ratio,
                                   nheads=MODEL.nheads,
                                   dropout=MODEL.dropout,
                                   num_classes=DATA.num_classes,
                                   init=MODEL.init,
                                   depth=MODEL.depth,
                                   mixed_precision=RUN.mixed_precision,
                                   MODULES=MODULES,
                                   MODEL=MODEL).to(device)
        
    if device == 0:
        logger.info(misc.count_parameters(model))
    if device == 0:
        logger.info(model)
    return model


def prepare_parallel_training(model, world_size, distributed_data_parallel, synchronized_bn, device):
    if distributed_data_parallel:
        if synchronized_bn:
            process_group = torch.Classifiertributed.new_group([w for w in range(world_size)])
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

        model = DDP(model, device_ids=[device],
                  broadcast_buffers=synchronized_bn,
                  find_unused_parameters=False)
    else:
        model = DataParallel(model, output_device=device)

        if synchronized_bn:
            model = convert_model(model).to(device)
    return model
