# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/ckpt.py

from os.path import join
import os
import glob

import torch
import numpy as np

import utils.log as log

def make_ckpt_dir(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir


def load_ckpt(model, optimizer, ckpt_path, load_model=False, load_opt=False, load_misc=False):
    import utils.misc as misc
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if load_model:
        model.load_state_dict(ckpt["state_dict"], strict=False)

    if load_opt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if load_misc:
        seed = ckpt["seed"]
        run_name = ckpt["run_name"]
        step = ckpt["step"]
        best_step = ckpt["best_step"]
        best_loss = ckpt["best_loss"]
        best_t1acc = ckpt["best_t1acc"]
        best_t10acc = ckpt["best_t10acc"]

        try:
            epoch = ckpt["epoch"]
        except:
            epoch = 0
        try:
            topk = ckpt["topk"]
        except:
            topk = "initialize"
        return seed, run_name, step, epoch, topk, best_step, best_loss, best_t1acc, best_t10acc


def load_model_ckpts(ckpt_dir, load_best, model, optimizer, run_name,
                         is_train, RUN, logger, global_rank, device, cfg_file, backbone):
    import utils.misc as misc
    when = "best" if load_best is True else "current"
    ckpt_path = glob.glob(join(ckpt_dir, "model={model}-{when}-weights-step*.pth".format(model=backbone,when=when)))[0]
    prev_run_name = torch.load(ckpt_path, map_location=lambda storage, loc: storage)["run_name"]

    seed, prev_run_name, step, epoch, topk, best_step, best_loss, best_t1acc, best_t10acc =\
        load_ckpt(model=model,
                  optimizer=optimizer,
                  ckpt_path=ckpt_path,
                  load_model=True,
                  load_opt=False if not is_train else True,
                  load_misc=True)

    if not is_train:
        prev_run_name = cfg_file[cfg_file.rindex("/")+1:cfg_file.index(".yaml")]+prev_run_name[prev_run_name.index("-train"):]

    if is_train and RUN.seed != seed:
        RUN.seed = seed + global_rank
        misc.fix_seed(RUN.seed)

    if device == 0:
        logger = log.make_logger(RUN.save_dir, prev_run_name, None)

        logger.info("Checkpoint is {}".format(ckpt_path))

    return prev_run_name, step, epoch, topk, best_step, best_loss, best_t1acc, best_t10acc, logger


def load_best_model(ckpt_dir, model, backbone):
    import utils.misc as misc
    model = misc.peel_model(model)
    ckpt_path = glob.glob(join(ckpt_dir, "model={model}-best-weights-step*.pth".format(model=backbone)))[0]

    _, _, _, _, _, _, best_step, _, _ = load_ckpt(model=model,
                                                  optimizer=None,
                                                  ckpt_path=ckpt_path,
                                                  load_model=True,
                                                  load_opt=False,
                                                  load_misc=True)

    return best_step


def load_prev_dict(directory, file_name):
    return np.load(join(directory, file_name), allow_pickle=True).item()


def check_is_pre_trained_model(ckpt_dir, GAN_train, GAN_test):
    assert GAN_train*GAN_test == 0, "cannot conduct GAN_train and GAN_test togather."
    if GAN_train:
        mode = "fake_trained"
    else:
        mode = "real_trained"

    ckpt_list = glob.glob(join(ckpt_dir, "model=C-{mode}-best-weights.pth".format(mode=mode)))
    if len(ckpt_list) == 0:
        is_pre_train_model = False
    else:
        is_pre_train_model = True
    return is_pre_train_model, mode


def load_GAN_train_test_model(model, mode, optimizer, RUN):
    ckpt_path = join(RUN.ckpt_dir, "model=C-{mode}-best-weights.pth".format(mode=mode))
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epoch_trained = ckpt["epoch"]
    best_top1 = ckpt["best_top1"]
    best_top10 = ckpt["best_top10"]
    best_epoch = ckpt["best_epoch"]
    return epoch_trained, best_top1, best_top10, best_epoch
