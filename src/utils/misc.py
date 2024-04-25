# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/misc.py

from os.path import exists, join, isfile
from datetime import datetime
import random
import os
import sys
import glob

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

import utils.ckpt as ckpt


class make_empty_object(object):
    pass


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accm_values_convert_dict(list_dict, value_dict, step, interval):
    for name, value_list in list_dict.items():
        if step is None:
            value_list += [value_dict[name]]
        else:
            try:
                value_list[step // interval - 1] = value_dict[name]
            except IndexError:
                try:
                    value_list += [value_dict[name]]
                except:
                    raise KeyError
        list_dict[name] = value_list
    return list_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(100 - wrong_k.mul_(100.0 / batch_size))
    return res


def prepare_folder(names, save_dir):
    for name in names:
        folder_path = join(save_dir, name)
        if not exists(folder_path):
            os.makedirs(folder_path)


def download_data_if_possible(data_name, data_dir):
    return True

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def setup(rank, world_size, backend="nccl"):
    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)
    else:
        # initialize the process group
        dist.init_process_group(backend,
                                init_method="tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]),
                                rank=rank,
                                world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def count_parameters(module):
    return "Number of parameters: {num}".format(num=sum([p.data.nelement() for p in module.parameters()]))


def toggle_grad(model, grad):
    model = peel_model(model)

    for _, param in model.named_parameters():
        param.requires_grad = grad


def load_log_dicts(directory, file_name, ph):
    try:
        log_dict = ckpt.load_prev_dict(directory=directory, file_name=file_name)
    except:
        log_dict = ph
    return log_dict


def make_model_require_grad(model):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module

    for name, param in model.named_parameters():
        param.requires_grad = True


def identity(x):
    return x


def set_bn_trainable(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()


def set_deterministic_op_trainable(m):
    if isinstance(m, torch.nn.modules.conv.Conv2d):
        m.train()
    if isinstance(m, torch.nn.modules.conv.ConvTranspose2d):
        m.train()
    if isinstance(m, torch.nn.modules.linear.Linear):
        m.train()
    if isinstance(m, torch.nn.modules.Embedding):
        m.train()


def reset_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.reset_running_stats()


def elapsed_time(start_time):
    now = datetime.now()
    elapsed = now - start_time
    return str(elapsed).split(".")[0]  # remove milliseconds


def reshape_weight_to_matrix(weight):
    weight_mat = weight
    dim = 0
    if dim != 0:
        weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def calculate_all_sn(model, prefix):
    sigmas = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            operations = model
            if "weight_orig" in name:
                splited_name = name.split(".")
                for name_element in splited_name[:-1]:
                    operations = getattr(operations, name_element)
                weight_orig = reshape_weight_to_matrix(operations.weight_orig)
                weight_u = operations.weight_u
                weight_v = operations.weight_v
                sigmas[prefix + "_" + name] = torch.dot(weight_u, torch.mv(weight_orig, weight_v)).item()
    return sigmas


def peel_model(model):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    return model


def save_model(model, when, step, ckpt_dir, states):
    model_tpl = "model={model}-{when}-weights-step={step}.pth"
    model_ckpt_list = glob.glob(join(ckpt_dir, model_tpl.format(model=model, when=when, step="*")))
    if len(model_ckpt_list) > 0:
        find_and_remove(model_ckpt_list[0])

    torch.save(states, join(ckpt_dir, model_tpl.format(model=model, when=when, step=step)))


def save_model_c(states, mode, RUN):
    ckpt_path = join(RUN.ckpt_dir, "model=C-{mode}-best-weights.pth".format(mode=mode))
    torch.save(states, ckpt_path)


def find_string(list_, string):
    for i, s in enumerate(list_):
        if string == s:
            return i


def find_and_remove(path):
    if isfile(path):
        os.remove(path)


def orthogonalize_model(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)


def interpolate(x0, x1, num_midpoints):
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device="cuda").to(x0.dtype)
    return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))


def save_dict_npy(directory, name, dictionary):
    if not exists(directory):
        os.makedirs(directory)

    save_path = join(directory, name + ".npy")
    np.save(save_path, dictionary)


def compute_gradient(fx, logits, label, num_classes):
    probs = torch.nn.Softmax(dim=1)(logits.detach().cpu())
    gt_prob = F.one_hot(label, num_classes)
    oneMp = gt_prob - probs
    preds = (probs*gt_prob).sum(-1)
    grad = torch.mean(fx.unsqueeze(1) * oneMp.unsqueeze(2), dim=0)
    return fx.norm(dim=1), preds, torch.norm(grad, dim=1)


def load_parameters(src, dst, strict=True):
    mismatch_names = []
    for dst_key, dst_value in dst.items():
        if dst_key in src:
            if dst_value.shape == src[dst_key].shape:
                dst[dst_key].copy_(src[dst_key])
            else:
                mismatch_names.append(dst_key)
                err = "source tensor {key}({src}) does not match with destination tensor {key}({dst}).".\
                    format(key=dst_key, src=src[dst_key].shape, dst=dst_value.shape)
                assert not strict, err
        else:
            mismatch_names.append(dst_key)
            assert not strict, "dst_key is not in src_dict."
    return mismatch_names


def enable_allreduce(dict_):
    loss = 0
    for key, value in dict_.items():
        if value is not None and key != "label":
            loss += value.mean()*0
    return loss

def sigmoid(x):
    return 1/(1+exp(-x))

def mixup_data(x_a, x_b, alpha=5, beta=5):
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1 
    mixed_x = lam * x_a + (1 - lam) * x_b
    return mixed_x, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def dataset_with_indices(cls):
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m