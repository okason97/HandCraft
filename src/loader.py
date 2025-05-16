# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/loader.py

from os.path import join
import json
import os

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import wandb

from data.data_util import Dataset_, train_val_dataset, OversamplingWrapper
from worker import WORKER
import utils.log as log
import utils.ckpt as ckpt
import utils.misc as misc
import utils.custom_ops as custom_ops
import models.model as model_generator


def load_worker(local_rank, cfgs, gpus_per_node, run_name):
    # -----------------------------------------------------------------------------
    # define default variables for loading ckpt or testing the trained model.
    # -----------------------------------------------------------------------------
    step, epoch, topk, best_step, best_loss, best_mpjpe, best_t1acc, best_t10acc, is_best = \
        0, 0, cfgs.OPTIMIZATION.batch_size, 0, 999, 999, 0, 0, False
    if cfgs.RUN.mode == "classification":
        loss_list_dict = {"train_loss": [], "train_top1": [], "train_top10": []}
        metric_dict_during_train = {"test_loss": [], "test_top1": [], "test_top10": []}
    else:
        metric_dict_during_train = {"test_loss": [], "test_mpjpe":[]}
        loss_list_dict = {"train_loss": []}

    # -----------------------------------------------------------------------------
    # determine cuda, cudnn, and backends settings.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.fix_seed:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False

    # -----------------------------------------------------------------------------
    # initialize all processes and fix seed of each process
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        global_rank = cfgs.RUN.current_node * (gpus_per_node) + local_rank
        print("Use GPU: {global_rank} for training.".format(global_rank=global_rank))
        misc.setup(global_rank, cfgs.OPTIMIZATION.world_size, cfgs.RUN.backend)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    misc.fix_seed(cfgs.RUN.seed + global_rank)

    # -----------------------------------------------------------------------------
    # Intialize python logger.
    # -----------------------------------------------------------------------------
    if local_rank == 0:
        logger = log.make_logger(cfgs.RUN.save_dir, run_name, None)
        if cfgs.RUN.ckpt_dir is not None:
            folder_hier = cfgs.RUN.ckpt_dir.split("/")
            if folder_hier[-1] == "":
                folder_hier.pop()
            logger.info("Run name : {run_name}".format(run_name=folder_hier.pop()))
        else:
            logger.info("Run name : {run_name}".format(run_name=run_name))
        for k, v in cfgs.super_cfgs.items():
            logger.info("cfgs." + k + " =")
            logger.info(json.dumps(vars(v), indent=2))
    else:
        logger = None

    # -----------------------------------------------------------------------------
    # load train and test datasets.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.train or cfgs.RUN.save_dataset:
        if local_rank == 0:
            logger.info("Load {name} train dataset.".format(name=cfgs.DATA.name))
        train_dataset = Dataset_(data_dir=cfgs.RUN.data_dir,
                                 train=True,
                                 load_data_in_memory=cfgs.RUN.load_data_in_memory,
                                 poses=cfgs.DATA.poses,
                                 max_len=cfgs.DATA.max_len,
                                 target_len=cfgs.DATA.target_len,
                                 pad_frames=cfgs.DATA.pad_frames,
                                 pad_mode=cfgs.DATA.pad_mode,
                                 min_samples=cfgs.DATA.min_samples,
                                 random_crop=cfgs.DATA.random_crop,
                                 drop_frame=cfgs.DATA.drop_frame,
                                 drop_keypoint=cfgs.DATA.drop_keypoint,
                                 block_size=cfgs.DATA.block_size,
                                 flip_p=cfgs.DATA.flip_p,
                                 scale=cfgs.DATA.scale,
                                 rot=cfgs.DATA.rot,
                                 mode=cfgs.RUN.mode)

        cfgs.DATA.num_classes = len(train_dataset.classes)

        train_dataset, valid_dataset = train_val_dataset(dataset = train_dataset, val_split=0.1, random_state = cfgs.RUN.seed, stratify=train_dataset.data['sign'].to_list())

        if cfgs.RUN.dset_used > 1:
            dset_used = int(cfgs.RUN.dset_used)
        elif cfgs.RUN.dset_used < 1:
            dset_used = 1 - cfgs.RUN.dset_used
        else:
            dset_used = cfgs.RUN.dset_used
        if dset_used != 1:
            train_dataset, _ = train_val_dataset(dataset = train_dataset, val_split=None, train_size=dset_used, random_state = cfgs.RUN.seed)

        if cfgs.DATA.oversample:
            train_dataset = OversamplingWrapper(train_dataset)

        if local_rank == 0:
            logger.info("Train dataset size: {dataset_size}".format(dataset_size=len(train_dataset)))
            logger.info("Valid dataset size: {dataset_size}".format(dataset_size=len(valid_dataset)))

        if cfgs.RUN.synth_dir:
            if local_rank == 0:
                logger.info("Load {name} train dataset.".format(name=cfgs.DATA.name))
            synth_dataset = Dataset_(data_dir=cfgs.RUN.synth_dir,
                                    train=True,
                                    load_data_in_memory=cfgs.RUN.load_data_in_memory,
                                    poses=cfgs.DATA.synth_poses,
                                    max_len=cfgs.DATA.max_len,
                                    target_len=cfgs.DATA.target_len,
                                    pad_frames=cfgs.DATA.pad_frames,
                                    pad_mode=cfgs.DATA.pad_mode,
                                    min_samples=cfgs.DATA.min_samples,
                                    random_crop=cfgs.DATA.random_crop,
                                    drop_frame=cfgs.DATA.drop_frame,
                                    drop_keypoint=cfgs.DATA.drop_keypoint,
                                    block_size=cfgs.DATA.block_size,
                                    flip_p=cfgs.DATA.flip_p,
                                    scale=cfgs.DATA.scale,
                                    rot=cfgs.DATA.rot,
                                    mode=cfgs.RUN.mode)
            if local_rank == 0:
                logger.info("Train dataset size: {dataset_size}".format(dataset_size=len(train_dataset)))
        else:
            synth_dataset = None
    else:
        train_dataset = None
        valid_dataset = None
        synth_dataset = None

    if local_rank == 0:
        logger.info("Load {name} test dataset.".format(name=cfgs.DATA.name))
    test_dataset = Dataset_(data_dir=cfgs.RUN.data_dir,
                            train=False,
                            load_data_in_memory=cfgs.RUN.load_data_in_memory,
                            filter_classes=train_dataset.classes,
                            map_classes=train_dataset.dataset.dataset.map_classes,
                            max_len=cfgs.DATA.max_len,
                            target_len=cfgs.DATA.target_len,
                            pad_frames=cfgs.DATA.pad_frames,
                            pad_mode=cfgs.DATA.pad_mode,
                            poses=cfgs.DATA.poses,
                            mode=cfgs.RUN.mode)
    if local_rank == 0:
        logger.info("Test dataset size: {dataset_size}".format(dataset_size=len(test_dataset)))

    # -----------------------------------------------------------------------------
    # define a distributed sampler for DDP train and test.
    # define dataloaders for train and test.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        cfgs.OPTIMIZATION.batch_size = cfgs.OPTIMIZATION.batch_size//cfgs.OPTIMIZATION.world_size

    if (cfgs.RUN.train or cfgs.RUN.save_dataset) and cfgs.RUN.distributed_data_parallel:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=cfgs.OPTIMIZATION.world_size,
                                           rank=local_rank,
                                           shuffle=True,
                                           drop_last=True)
        valid_sampler = DistributedSampler(valid_dataset,
                                           num_replicas=cfgs.OPTIMIZATION.world_size,
                                           rank=local_rank,
                                           shuffle=False,
                                           drop_last=True)
        topk = cfgs.OPTIMIZATION.batch_size
        if cfgs.RUN.synth_dir:
            synth_sampler = DistributedSampler(synth_dataset,
                                            num_replicas=cfgs.OPTIMIZATION.world_size,
                                            rank=local_rank,
                                            shuffle=True,
                                            drop_last=True)
        else:
            synth_sampler = None
    else:
        train_sampler = None
        valid_sampler = None
        if cfgs.RUN.synth_dir:
            synth_sampler = torch.utils.data.RandomSampler(data_source=synth_dataset, 
                                                           replacement=True, 
                                                           num_samples=len(train_dataset))
            #synth_sampler = misc.RandomBatchSampler(data_source=synth_dataset, 
            #                                        batch_size=cfgs.OPTIMIZATION.batch_size,
            #                                        num_samples=len(train_dataset))
        else:
            synth_sampler = None

    if cfgs.RUN.train or cfgs.RUN.save_dataset:
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=cfgs.OPTIMIZATION.batch_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=True,
                                      prefetch_factor=cfgs.RUN.prefetch_factor,
                                      num_workers=cfgs.RUN.num_workers,
                                      sampler=train_sampler,
                                      drop_last=True,
                                      #collate_fn=None if cfgs.DATA.pad_frames else misc.collate_fn_nested,
                                      persistent_workers=True)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                        batch_size=cfgs.OPTIMIZATION.batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        prefetch_factor=cfgs.RUN.prefetch_factor,
                                        num_workers=cfgs.RUN.num_workers,
                                        sampler=valid_sampler,
                                        #collate_fn=None if cfgs.DATA.pad_frames else misc.collate_fn_nested,
                                        drop_last=True)
        if cfgs.RUN.synth_dir:
            synth_dataloader = DataLoader(dataset=synth_dataset,
                                        batch_size=cfgs.OPTIMIZATION.batch_size,
                                        shuffle=(synth_sampler is None),
                                        pin_memory=True,
                                        prefetch_factor=cfgs.RUN.prefetch_factor,
                                        num_workers=cfgs.RUN.num_workers,
                                        sampler=synth_sampler,
                                        drop_last=True,
                                        #collate_fn=None if cfgs.DATA.pad_frames else misc.collate_fn_nested,
                                        persistent_workers=True)
        else:
            synth_dataloader = None
    else:
        train_dataloader = None
        valid_dataloader = None
        synth_dataloader = None

    if cfgs.RUN.distributed_data_parallel:
        test_sampler = DistributedSampler(test_dataset,
                                            num_replicas=cfgs.OPTIMIZATION.world_size,
                                            rank=local_rank,
                                            shuffle=False,
                                            drop_last=True)
    else:
        test_sampler = None

    test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=cfgs.OPTIMIZATION.batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    prefetch_factor=cfgs.RUN.prefetch_factor,
                                    num_workers=cfgs.RUN.num_workers,
                                    sampler=test_sampler,
                                    #collate_fn=None if cfgs.DATA.pad_frames else misc.collate_fn_nested,
                                    drop_last=True)

    # -----------------------------------------------------------------------------
    # load the model
    # -----------------------------------------------------------------------------
    model = model_generator.load_model(DATA=cfgs.DATA,
                                    MODEL=cfgs.MODEL,
                                    MODULES=cfgs.MODULES,
                                    RUN=cfgs.RUN,
                                    device=local_rank,
                                    logger=logger)
    if cfgs.RUN.twin_generator:
        r_model = model_generator.load_model(DATA=cfgs.DATA,
                                MODEL=cfgs.MODEL,
                                MODULES=cfgs.MODULES,
                                RUN=cfgs.RUN,
                                device=local_rank,
                                logger=logger)
    else:
        r_model = None


    if local_rank != 0:
        custom_ops.verbosity = "none"

    # -----------------------------------------------------------------------------
    # define optimizer
    # -----------------------------------------------------------------------------
    if cfgs.RUN.mode == "classification" and cfgs.RUN.synth_dir:
        cfgs.define_optimizer(model, cfgs.OPTIMIZATION.synth_total_steps, len(train_dataloader))
    else:
        cfgs.define_optimizer(model, cfgs.OPTIMIZATION.total_steps, len(train_dataloader))

    # -----------------------------------------------------------------------------
    # load the model from a checkpoint if possible
    # -----------------------------------------------------------------------------
    if cfgs.RUN.ckpt_dir is not None:
        if local_rank == 0:
            logger.handlers[0].close()
            os.remove(join(cfgs.RUN.save_dir, "logs", run_name + ".log"))
        run_name, step, epoch, topk, best_step, best_loss, best_mpjpe, best_t1acc, best_t10acc, logger =\
            ckpt.load_model_ckpts(ckpt_dir=cfgs.RUN.ckpt_dir,
                                      load_best=cfgs.RUN.load_best,
                                      model=model,
                                      optimizer=cfgs.OPTIMIZATION.optimizer,
                                      run_name=run_name,
                                      is_train=cfgs.RUN.train,
                                      RUN=cfgs.RUN,
                                      logger=logger,
                                      global_rank=global_rank,
                                      device=local_rank,
                                      cfg_file=cfgs.RUN.cfg_file,
                                      backbone=cfgs.MODEL.backbone)

        if topk == "initialize":
            topk == cfgs.OPTIMIZATION.batch_size

    if cfgs.RUN.ckpt_dir is None:
        if local_rank == 0:
            cfgs.RUN.ckpt_dir = ckpt.make_ckpt_dir(join(cfgs.RUN.save_dir, "checkpoints", run_name))
        dict_dir = join(cfgs.RUN.save_dir, "statistics", run_name)
        loss_list_dict = misc.load_log_dicts(directory=dict_dir, file_name="losses.npy", ph=loss_list_dict)
        metric_dict_during_train = misc.load_log_dicts(directory=dict_dir, file_name="metrics.npy", ph=metric_dict_during_train)

    if cfgs.RUN.r_ckpt_dir is not None:
        ckpt.load_model_ckpts(ckpt_dir=cfgs.RUN.r_ckpt_dir,
                                load_best=cfgs.RUN.load_best,
                                model=r_model,
                                run_name=run_name,
                                is_train=cfgs.RUN.train,
                                RUN=cfgs.RUN,
                                optimizer=None,
                                logger=logger,
                                global_rank=global_rank,
                                device=local_rank,
                                cfg_file=cfgs.RUN.cfg_file,
                                backbone=cfgs.MODEL.backbone)

    # -----------------------------------------------------------------------------
    # prepare parallel training
    # -----------------------------------------------------------------------------
    if cfgs.OPTIMIZATION.world_size > 1:
        model = model_generator.prepare_parallel_training(model=model,
                                        world_size=cfgs.OPTIMIZATION.world_size,
                                        distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                        synchronized_bn=cfgs.RUN.synchronized_bn,
                                        device=local_rank)
        if cfgs.RUN.twin_generator:
            r_model = model_generator.prepare_parallel_training(model=r_model,
                                            world_size=cfgs.OPTIMIZATION.world_size,
                                            distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                            synchronized_bn=cfgs.RUN.synchronized_bn,
                                            device=local_rank)

    # -----------------------------------------------------------------------------
    # initialize WORKER for training and testing
    # -----------------------------------------------------------------------------
    worker = WORKER(
        cfgs=cfgs,
        run_name=run_name,
        model=model,
        r_model=r_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        synth_dataloader=synth_dataloader,
        global_rank=global_rank,
        local_rank=local_rank,
        logger=logger,
        best_step=best_step,
        best_loss=best_loss,
        best_mpjpe=best_mpjpe,
        best_t1acc=best_t1acc,
        best_t10acc=best_t10acc,
        loss_list_dict=loss_list_dict,
        metric_dict_during_train=metric_dict_during_train,
    )

    # -----------------------------------------------------------------------------
    # pretrain model with the synthetic dataset
    # -----------------------------------------------------------------------------
    if cfgs.RUN.mode == "classification" and cfgs.RUN.synth_dir:
        if cfgs.RUN.train:
            if global_rank == 0:
                logger.info("Pretraining model!")

            worker.prepare_synthtrain_iter(epoch_counter=epoch)
            while step < cfgs.OPTIMIZATION.synth_total_steps:
                top1, top10, loss = worker.train_step(step)

                if global_rank == 0 and (step + 1) % cfgs.RUN.print_every == 0:

                    worker.log_train_statistics(current_step=step,
                                                loss=loss,
                                                top1=top1,
                                                top10=top10)

                step += 1
            step = 0

            worker.evaluate(step=step, writing=True, training=True)

            if global_rank == 0:
                logger.info("End of pretraining!")

    # -----------------------------------------------------------------------------
    # reset optimizer if synthetic pretraining is used
    # -----------------------------------------------------------------------------
    if cfgs.RUN.mode == "classification" and cfgs.RUN.synth_dir:
        cfgs.define_optimizer(model, cfgs.OPTIMIZATION.total_steps, len(train_dataloader))
        worker.reset_optimizer(cfgs.OPTIMIZATION)

    # -----------------------------------------------------------------------------
    # train model
    # -----------------------------------------------------------------------------
    if cfgs.RUN.train:
        if global_rank == 0:
            logger.info("Start training!")

            if "prediction" in cfgs.RUN.mode:
                worker.visualize_real_poses()

        worker.prepare_train_iter(epoch_counter=epoch)
        while step < cfgs.OPTIMIZATION.total_steps:
            top1, top10, loss = worker.train_step(step)

            if global_rank == 0 and (step + 1) % cfgs.RUN.print_every == 0:

                worker.log_train_statistics(current_step=step,
                                            loss=loss,
                                            top1=top1,
                                            top10=top10)

            if step % cfgs.RUN.save_every == 0:
                if global_rank == 0 and "prediction" in cfgs.RUN.mode:
                    worker.visualize_fake_poses(step=step)

                # validate model for monitoring purpose
                is_best = worker.evaluate(step=step, writing=True, training=True)

                # save model in "./checkpoints/RUN_NAME/*"
                if global_rank == 0:
                    worker.save(step=step, is_best=is_best)

                # stop processes until all processes arrive
                if cfgs.RUN.distributed_data_parallel:
                    dist.barrier(worker.group)

            step += 1

        if global_rank == 0:
            logger.info("End of training!")

    # -----------------------------------------------------------------------------
    # evaluate the best model on the testing dataset
    # -----------------------------------------------------------------------------
    print("")
    worker.epoch_counter = epoch

    if global_rank == 0:
        best_step = ckpt.load_best_model(ckpt_dir=cfgs.RUN.ckpt_dir,
                                         model=model,
                                         backbone=cfgs.MODEL.backbone)
        print(""), logger.info("-" * 80)

    if cfgs.RUN.test:
        worker.evaluate(step=best_step, writing=False, training=False)

    if cfgs.RUN.save_samples:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.visualize_samples_poses()

    if cfgs.RUN.save_dataset:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        single_class_sampler_fabric = misc.SingleClassSamplerFabric(train_dataset)
        worker.prepare_generation()
        for sign in train_dataset.classes:
            class_n = train_dataset.dataset.dataset.map_classes[sign]
            print(f"Generating class {class_n}")
            single_class_sampler = single_class_sampler_fabric.get_sampler(class_n)
            class_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=cfgs.OPTIMIZATION.batch_size,
                                pin_memory=True,
                                prefetch_factor=cfgs.RUN.prefetch_factor,
                                num_workers=cfgs.RUN.num_workers,
                                sampler=single_class_sampler,
                                drop_last=True,
                                #collate_fn=None if cfgs.DATA.pad_frames else misc.collate_fn_nested,
                                persistent_workers=True)
            worker.save_dataset(class_dataloader, sign)
        worker.save_metadata()

    if global_rank == 0:
        wandb.finish()