# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/model.py

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from sync_batchnorm.batchnorm import convert_model
import utils.misc as misc
from ema_pytorch import EMA

def load_model(DATA, MODEL, MODULES, RUN, device, logger):
    if device == 0:
        logger.info("Build the model.")
    module = __import__("models.{backbone}".format(backbone=MODEL.backbone), fromlist=["something"])
    if device == 0:
        logger.info("Modules are located on './src/models.{backbone}'.".format(backbone=MODEL.backbone))

        model = module.Model(DATA=DATA,
                                RUN=RUN,
                                MODULES=MODULES,
                                MODEL=MODEL)
        if MODEL.apply_ema:
            model = EMA(model,
                        beta = MODEL.ema_beta,              # exponential moving average factor
                        update_after_step = MODEL.ema_update_after_step,    # only after this number of .update() calls will it start updating
                        update_every = MODEL.ema_update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
                        power=MODEL.ema_power)              # exponential factor of EMA warmup. Default: 2/3.
            
    if device == 0:
        logger.info(misc.count_parameters(model))
    if device == 0:
        logger.info(model)
    return model.to(device)


def prepare_parallel_training(model, world_size, distributed_data_parallel, synchronized_bn, device):
    if distributed_data_parallel:
        if synchronized_bn:
            process_group = torch.distributed.new_group([w for w in range(world_size)])
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

        model = DDP(model, device_ids=[device],
                  broadcast_buffers=synchronized_bn,
                  find_unused_parameters=False)
    else:
        model = DataParallel(model, output_device=device)

        if synchronized_bn:
            model = convert_model(model).to(device)
    return model
