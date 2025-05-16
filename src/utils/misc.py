# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/misc.py

from os.path import dirname, exists, join, isfile
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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from torch.utils.data import Sampler
import shutil
from torch import linalg as LA
import pywt

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
    return 1/(1+math.exp(-x))

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

def i_normalize(data, pose):
    MEAN = {
        'pose': [0.0011, 0.1365, 0.0693],
        'right_hand': [-0.0325,  0.1191,  0.0000],
        'left_hand': [0.0251, 0.1365, 0.0000],
        'face': [-0.0024, -0.0086,  0.1384]
    }
    STD = {
        'pose': [0.0184, 0.1365, 0.0692],
        'right_hand': [0.0111, 0.0222, 42],
        'left_hand': [0.0105, 0.0198, 42],
        'face': [0.0046, 0.0086, 0.0040]
    }

    mean = np.array(MEAN[pose])
    std = np.array(STD[pose])

    mean = mean[None, None, :]
    std = std[None, None, :]

    # Normalize the data
    normalized_data = data * std + mean

    return normalized_data

# MediaPipe pose landmarks connection (in their order)

CONNECTIONS = {'pose': [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (11, 23), (12, 14), (14, 16), (15,19), (15,21), (16,22), (16,20), (20,18), (19,17), (16, 18), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32)],
               'right_hand': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)],
               'left_hand': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)],
               'face': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), 
                    (8, 9), (9, 10), (10, 11), (11, 8),
                    (12, 13), (13, 14), (14, 15),
                    (16, 17), (17, 18), (18, 19), (19, 16),
                    (20, 21), (21, 22), (22, 23)]}

CONNECTIONS_REDUCED = {'pose': [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5)],
               'right_hand': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)],
               'left_hand': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)],
               'face': [(0, 1), (1, 2), (2, 3), (3, 0), 
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (8, 9), (9, 10), (10, 11), (11, 8)]}

def plot_keypoints(xyz_keypoints, connections):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = xyz_keypoints[:, 0]
    ys = xyz_keypoints[:, 1]
    zs = xyz_keypoints[:, 2]

    ax.scatter(xs, ys, zs)

    for connection in connections:
        start = connection[0]
        end = connection[1]

        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set aspect ratio
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(-70, -90)

    return plt

def animate_keypoints(xyz_keypoints, pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xss = xyz_keypoints[:, :, 0]
    yss = xyz_keypoints[:, :, 1]
    zss = xyz_keypoints[:, :, 2]

    def animate(i):
        ax.cla()   
        xs = xss[i]
        ys = yss[i]
        zs = zss[i]
        ax.scatter(xs, ys, zs)

        for connection in CONNECTIONS_REDUCED[pose]:
            start = connection[0]
            end = connection[1]

            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'blue')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set aspect ratio
        max_range = np.array([xss[0].max()-xss[0].min(), yss[0].max()-yss[0].min(), zss[0].max()-zss[0].min()]).max() / 2.0
        mid_x = (xss[0].max()+xss[0].min()) * 0.5
        mid_y = (yss[0].max()+yss[0].min()) * 0.5
        mid_z = (zss[0].max()+zss[0].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(-70, -90)
    
    ani = animation.FuncAnimation(fig, animate, repeat=False,
                                        frames=len(xyz_keypoints) - 1, interval=50)
    return plt, ani

def animate_all_keypoints(pose_keypoints, rhand_keypoints, lhand_keypoints, face_keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.cla()

        xs_max, xs_min, ys_max, ys_min, zs_max, zs_min = np.empty(4), np.empty(4), np.empty(4), np.empty(4), np.empty(4), np.empty(4)
        xs_max[0], xs_min[0], ys_max[0], ys_min[0], zs_max[0], zs_min[0] = create_axis(ax, pose_keypoints, i, CONNECTIONS_REDUCED['pose'])
        xs_max[1], xs_min[1], ys_max[1], ys_min[1], zs_max[1], zs_min[1] = create_axis(ax, rhand_keypoints, i, CONNECTIONS_REDUCED['right_hand'])
        xs_max[2], xs_min[2], ys_max[2], ys_min[2], zs_max[2], zs_min[2] = create_axis(ax, lhand_keypoints, i, CONNECTIONS_REDUCED['left_hand'])
        xs_max[3], xs_min[3], ys_max[3], ys_min[3], zs_max[3], zs_min[3] = create_axis(ax, face_keypoints, i, CONNECTIONS_REDUCED['face'])

        xs_max = max(xs_max)
        ys_max = max(ys_max)
        zs_max = max(zs_max)
        xs_min = min(xs_min)
        ys_min = min(ys_min)
        zs_min = min(zs_min)

        # Set aspect ratio
        max_range = np.array([xs_max-xs_min, ys_max-ys_min, zs_max-zs_min]).max() / 2.0
        mid_x = (xs_max+xs_min) * 0.5
        mid_y = (ys_max+ys_min) * 0.5
        mid_z = (zs_max+zs_min) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.view_init(-90, -90)

    ani = animation.FuncAnimation(fig, animate, repeat=False,
                                        frames=len(pose_keypoints) - 1, interval=50)

    plt.close(fig)

    return ani

def create_axis(ax, xyz_keypoints, i, connections):
    xs = xyz_keypoints[i, :, 0]
    ys = xyz_keypoints[i, :, 1]
    zs = xyz_keypoints[i, :, 2]
    ax.scatter(xs, ys, zs, s=5)

    for connection in connections:
        start = connection[0]
        end = connection[1]

        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'blue')

    return xyz_keypoints[0, :, 0].max(), xyz_keypoints[0, :, 0].min(), xyz_keypoints[0, :, 1].max(), xyz_keypoints[0, :, 1].min(), xyz_keypoints[0, :, 2].max(), xyz_keypoints[0, :, 2].min()

def save_gif(ani, save_path, logger, logging=True):
    if logger is None:
        logging = False
    directory = dirname(save_path)

    if not exists(directory):
        os.makedirs(directory)

    writer = animation.PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
    ani.save(save_path, writer=writer, dpi=200)
    if logging:
        logger.info("Save poses to {}".format(save_path))

def classifier_free_guidance(pred, guidance_scale):
    # Compute scores from both models
    pred_uncond, pred_cond = pred.chunk(2)

    # Combine scores
    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    return pred

class SingleClassSamplerFabric():
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_indices = [[] for _ in range(len(self.dataset.classes))]
        for i, (value, target, label) in enumerate(dataset):
            self.class_indices[int(label.item())].append(i)

    def get_sampler(self, class_label):
        return SingleClassSampler(self.class_indices[class_label])

class SingleClassSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def prepare_save_folder(directory):
    if exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def mpjpe(motion_pred, motion_target):
    pjpe = LA.vector_norm(motion_pred - motion_target, 2, -1)
    return torch.mean(pjpe)

class DCTLayer(torch.nn.Module):
    def __init__(self, frames=32, device='cpu'):
        super(DCTLayer, self).__init__()
        self.frames = frames
        self.device = device
        
        self.dct_m, self.idct_m = self.get_dct_matrix(frames)

    def get_dct_matrix(self, N):
        dct_m = torch.eye(N)
        for k in torch.arange(N):
            for i in torch.arange(N):
                w = torch.sqrt(torch.tensor(2 / N))
                if k == 0:
                    w = torch.sqrt(torch.tensor(1 / N))
                dct_m[k, i] = w * torch.cos(torch.pi * (i + 1 / 2) * k / N)
        idct_m = torch.linalg.inv(dct_m)
        return dct_m.to(self.device), idct_m.to(self.device)

    def forward(self, x):
        # Apply DCT to each keypoint
        return torch.matmul(self.dct_m, x)  # (batch_size, 32, 180)

    def inverse(self, x):
        # Apply IDCT to each keypoint
        return torch.matmul(self.idct_m, x)  # (batch_size, 32, 180)

class Identity(torch.nn.Module):
    # placeholder transform
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def inverse(self, x):
        return x

def add_noise(x, noise_scale):
        return x + torch.randn_like(x) * noise_scale

def generate_poses(model, inputs, transform, target_len):
    offset = inputs[:, -1:]
    # apply Transform 
    inputs = transform(inputs)

    # get model output for the current batch
    outputs = model(inputs)

    # apply Inverse Transform 
    outputs = transform.inverse(outputs)

    # apply output over the last frame as offset
    outputs = outputs[:, :target_len] + offset
    
    return outputs

def cond_generate_poses(model, inputs, labels, transform, target_len):
    offset = inputs[:, -1:]
    # apply Transform 
    inputs = transform(inputs)

    # get model output for the current batch
    outputs = model(inputs, labels)

    # apply Inverse Transform 
    outputs = transform.inverse(outputs)

    # apply output over the last frame as offset
    outputs = outputs[:, :target_len] + offset
    
    return outputs

class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc
    """
    def __init__(self, data_source, batch_size, num_samples=None):
        self.data_source = data_source
        self.batch_size = batch_size
        if num_samples and num_samples < len(data_source):
            self.num_samples = num_samples
        else:
            self.num_samples = len(data_source)
        self.n_batches = self.num_samples // self.batch_size
        self.total_batches = len(data_source) // self.batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        batch_ids = np.random.permutation(self.total_batches)

        for id in batch_ids[:self.n_batches]:
            for index in range(id * self.batch_size, (id + 1) * self.batch_size):
                yield index

        if self.n_batches < self.total_batches:
            if self.n_batches < self.total_batches:
                for index in range(batch_ids[self.n_batches] * self.batch_size, 
                                   (batch_ids[self.n_batches] + 1) * self.batch_size):
                    yield index
            else:
                for index in range(self.n_batches * self.batch_size, self.num_samples):
                    yield index

def collate_fn_nested(batch):
    return [torch.nested.as_nested_tensor(samples) for samples in zip(*batch)]

def unpad(batch, masks):
    return torch.nested.nested_tensor([sample[mask.any(dim=1)] for mask, sample in zip(masks, batch)])

def keep_first_true(tensor, dim=-1):
    """
    Modifies a boolean tensor to keep only the first True value along specified dimension.
    
    Args:
        tensor (torch.Tensor): Input boolean tensor
        dim (int): Dimension along which to find first True value. Default: -1 (last dimension)
        
    Returns:
        torch.Tensor: Modified boolean tensor with same shape as input
    """
    # Get cumulative sum along dimension
    cumsum = torch.cumsum(tensor.long(), dim=dim)

    # First True will have value 1, subsequent Trues will have values > 1
    return cumsum == 1

def pad_index(tensor):
    # Find position of first True in each row
    first_ones = (tensor == True).float().argmax(dim=1)

    # The last 0 positions are right before the first 1s
    last_zero_positions = first_ones - 1

    # Create row indices
    row_indices = torch.arange(len(tensor))

    return row_indices, last_zero_positions