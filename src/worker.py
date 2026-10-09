# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/worker.py

import json
import math
import os
import random
import shutil
from datetime import datetime
from os.path import join

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import wandb

import utils.misc as misc


def numpy_rng_state():
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return name, torch.from_numpy(keys.astype(np.int64)), int(pos), int(has_gauss), float(cached_gaussian)


class WORKER(object):
    def __init__(
        self,
        cfgs,
        run_name,
        model,
        r_model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        synth_dataloader,
        global_rank,
        local_rank,
        logger,
        best_step,
        best_loss,
        best_mpjpe,
        best_t1acc,
        best_t10acc,
        loss_list_dict,
        metric_dict_during_train,
    ):
        self.cfgs = cfgs
        self.run_name = run_name
        self.model = model
        self.r_model = r_model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.synth_dataloader = synth_dataloader
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.logger = logger
        self.best_step = best_step
        self.best_loss = best_loss
        self.best_mpjpe = best_mpjpe
        self.best_t1acc = best_t1acc
        self.best_t10acc = best_t10acc
        self.loss_list_dict = loss_list_dict
        self.metric_dict_during_train = metric_dict_during_train
        self.metric_dict_during_final_eval = {}
        self.wandb_step = None
        self.gen_curr_id = 0
        self.base_save_dir = join(cfgs.RUN.save_dir, "generated_datasets/{run_name}".format(run_name=self.run_name))
        self.metadata = {'id': [], 'sign': [], 'signer': [], 'start': [], 'end': []}

        if cfgs.RUN.mode == "classification":
            self.train_step = self.classification_train_step
            self.evaluate_step = self.classification_evaluate_step
        elif cfgs.RUN.mode == "prediction":
            self.train_step = self.prediction_train_step
            self.evaluate_step = self.prediction_evaluate_step
        elif cfgs.RUN.mode == "cond_prediction":
            self.train_step = self.cond_prediction_train_step
            self.evaluate_step = self.cond_prediction_evaluate_step
        else:
            raise NotImplementedError

        if cfgs.DATA.transform == "DCT":
            self.transform = misc.DCTLayer(frames=cfgs.DATA.input_size[0], device=self.local_rank)
        else:
            self.transform = misc.Identity()

        # self.cfgs.define_augments(local_rank)
        self.cfgs.define_losses()
        self.DATA = cfgs.DATA
        self.MODEL = cfgs.MODEL
        self.OPTIMIZATION = cfgs.OPTIMIZATION
        self.PRE = cfgs.PRE
        self.AUG = cfgs.AUG
        self.RUN = cfgs.RUN
        # "pretrain" during the pretraining on synthetic data, "train" otherwise (set by the loader)
        self.phase = "train"
        # epochs of synthetic pretraining before the training on the real data: its wandb steps come after them
        self.wandb_offset = self.OPTIMIZATION.synth_total_steps if (self.RUN.mode == "classification" and self.RUN.synth_dir) else 0
        self.MISC = cfgs.MISC
        self.DDP = self.RUN.distributed_data_parallel

        self.loss = cfgs.LOSS.loss

        self.len_body_kp = 33 if self.DATA.poses[0][1] == 'all' else len(self.DATA.poses[0][1])

        if self.DDP:
            self.group = dist.new_group([n for n in range(self.OPTIMIZATION.world_size)])

        if self.RUN.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')

        if self.global_rank == 0:
            wandb.init(
                project=self.RUN.project,
                entity=self.RUN.entity,
                name=self.run_name,
                dir=self.RUN.save_dir,
                # a resumed run continues the wandb run of the checkpoint
                id=self.RUN.wandb_id,
                resume="allow" if self.RUN.wandb_id else None,
                config=misc.cfgs_to_dict(cfgs),
            )

        self.start_time = datetime.now()

    def prepare_train_iter(self, epoch_counter):
        self.epoch_counter = epoch_counter
        if self.DDP:
            self.train_dataloader.sampler.set_epoch(self.epoch_counter)
        self.train_iter = self.train_dataloader

    def set_epoch(self, dataloader, epoch):
        # the samplers of the training and validation loaders draw the order and the augmentation from the epoch
        if not self.DDP and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

    def resume_state(self):
        """
        State needed to continue the training exactly where it stopped, saved with every checkpoint.
        """
        return {
            "scheduler": self.OPTIMIZATION.scheduler.state_dict() if self.OPTIMIZATION.scheduler else None,
            "scaler": self.scaler.state_dict() if self.RUN.mixed_precision else None,
            "rng": {
                "python": random.getstate(),
                # stored as a tuple of plain values and a tensor, so the checkpoint loads with weights_only=True
                "numpy": numpy_rng_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "wandb_id": wandb.run.id if wandb.run is not None else None,
        }

    def load_resume_state(self, state):
        if self.RUN.mixed_precision and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        rng = state.get("rng")
        if rng is not None:
            random.setstate(rng["python"])
            name, keys, pos, has_gauss, cached_gaussian = rng["numpy"]
            np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached_gaussian))
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])

    def prepare_synthtrain_iter(self, epoch_counter):
        self.epoch_counter = epoch_counter
        if self.DDP:
            self.synth_dataloader.sampler.set_epoch(self.epoch_counter)
        self.train_iter = self.synth_dataloader

    def optimizer_step(self):
        if self.RUN.mixed_precision:
            self.scaler.step(self.OPTIMIZATION.optimizer)
            self.scaler.update()
        else:
            self.OPTIMIZATION.optimizer.step()
        if self.OPTIMIZATION.scheduler:
            self.OPTIMIZATION.scheduler.step()

    def reset_optimizer(self, OPTIMIZATION):
        self.OPTIMIZATION = OPTIMIZATION
        if self.RUN.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')

    def reset_best(self):
        """
        Forget the best validation result, so that the best checkpoint is chosen among the epochs of the training on
        the real data and not compared with the model at the end of the synthetic pretraining.
        """
        self.best_loss, self.best_mpjpe, self.best_t1acc, self.best_t10acc, self.best_step = None, None, 0, 0, 0

    # -----------------------------------------------------------------------------
    # train model
    # -----------------------------------------------------------------------------
    def classification_train_step(self, step):
        # make the model be trainable before starting training
        self.model.train()
        # toggle gradients
        misc.toggle_grad(model=self.model, grad=True)
        # create accumulators
        valid_top1_acc, valid_top10_acc = misc.AverageMeter(), misc.AverageMeter()
        valid_loss = misc.AverageMeter()
        # sample real values and labels, then train for an epoch
        for values, labels in self.train_iter:
            self.OPTIMIZATION.optimizer.zero_grad()
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                values = values.to(self.local_rank, non_blocking=True).float()
                labels = labels.to(self.local_rank, non_blocking=True)

                if self.DATA.pad_mode == 'pad':
                    masks = torch.isnan(values)
                    values = torch.nan_to_num(values)
                else:
                    masks = None

                values = self.transform(values)

                # get model output for the current batch
                outputs = self.model(values, masks)

            loss = self.loss(outputs, labels)

            # accumulate gradients of the model
            if self.RUN.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # update the model using the pre-defined optimizer
            self.optimizer_step()

            # calculate topk
            valid_acc1, valid_acc10 = misc.accuracy(outputs.data, labels, topk=(1, 10))

            # accumulate topk
            valid_top1_acc.update(valid_acc1.item(), values.size(0))
            valid_top10_acc.update(valid_acc10.item(), values.size(0))

            # accumulate loss
            valid_loss.update(loss.item(), values.size(0))

        # apply late dropout when reaching the indicated step
        self.apply_l_drop(step)

        if self.RUN.empty_cache:
            torch.cuda.empty_cache()

        top1 = valid_top1_acc.avg
        top10 = valid_top10_acc.avg
        # del values
        # del labels
        # del masks
        # del outputs
        return top1, top10, valid_loss.avg

    def prediction_train_step(self, step):
        # make the model be trainable before starting training
        self.model.train()
        # toggle gradients
        misc.toggle_grad(model=self.model, grad=True)
        # create accumulators
        valid_loss = misc.AverageMeter()
        # sample real values and targets, then train for an epoch
        for values, targets in self.train_iter:
            self.OPTIMIZATION.optimizer.zero_grad()
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                values = values.to(self.local_rank, non_blocking=True).float()
                targets = targets.to(self.local_rank, non_blocking=True).float()

                if self.RUN.reverse:
                    r_values = torch.flip(values, [1])
                    r_targets = torch.flip(targets, [1])
                    values = r_targets
                    targets = r_values

                outputs = misc.generate_poses(self.model, values, self.transform, self.DATA.target_len)

            # calculate Loss
            loss = self.loss(outputs, targets)

            # accumulate gradients of the model
            if self.RUN.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # update the model using the pre-defined optimizer
            self.optimizer_step()

            # accumulate loss
            valid_loss.update(loss.item(), values.size(0))

        # apply late dropout when reaching the indicated step
        self.apply_l_drop(step)

        if self.RUN.empty_cache:
            torch.cuda.empty_cache()

        return None, None, valid_loss.avg

    def cond_prediction_train_step(self, step):
        # make the model be trainable before starting training
        self.model.train()
        # toggle gradients
        misc.toggle_grad(model=self.model, grad=True)
        # create accumulators
        valid_loss = misc.AverageMeter()
        # sample real values, targets and labels, then train for an epoch
        for values, targets, labels in self.train_iter:
            self.OPTIMIZATION.optimizer.zero_grad()
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                # load values, targets and labels onto the GPU memory
                values = values.to(self.local_rank, non_blocking=True).float()
                targets = targets.to(self.local_rank, non_blocking=True).float()
                labels = labels.to(self.local_rank, non_blocking=True)

                if self.RUN.reverse:
                    r_values = torch.flip(values, [1])
                    r_targets = torch.flip(targets, [1])
                    values = r_targets
                    targets = r_values

                outputs = misc.cond_generate_poses(self.model, values, labels, self.transform, self.DATA.target_len)

            # calculate Loss
            loss = self.loss(outputs, targets)

            # accumulate gradients of the model
            if self.RUN.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # update the model using the pre-defined optimizer
            self.optimizer_step()

            # accumulate loss
            valid_loss.update(loss.item(), values.size(0))

        # apply late dropout when reaching the indicated step
        self.apply_l_drop(step)

        if self.RUN.empty_cache:
            torch.cuda.empty_cache()

        return None, None, valid_loss.avg

    def apply_l_drop(self, step):
        if self.MODEL.late_dropout:
            if self.MODEL.late_dropout_step <= step:
                if self.MODEL.apply_ema:
                    self.model.model.update_dropout(self.MODEL.late_dropout)
                    self.model.ema_model.update_dropout(self.MODEL.late_dropout)
                else:
                    self.model.update_dropout(self.MODEL.late_dropout)
        if self.MODEL.late_drop_path:
            if self.MODEL.late_drop_path_step <= step:
                if self.MODEL.apply_ema:
                    self.model.model.update_drop_path(self.MODEL.late_drop_path)
                    self.model.ema_model.update_drop_path(self.MODEL.late_drop_path)
                else:
                    self.model.update_drop_path(self.MODEL.late_drop_path)

    # -----------------------------------------------------------------------------
    # log training statistics
    # -----------------------------------------------------------------------------
    def wandb_step_of(self, step):
        """
        The wandb step of epoch `step` (0-based) of the current phase: the epochs of the synthetic pretraining come
        first, then those of the training on the real data.
        """
        return step + 1 + (self.wandb_offset if self.phase == "train" else 0)

    def log_train_statistics(self, current_step, loss, top1=None, top10=None):
        pretrain = self.phase == "pretrain"
        epoch = current_step + 1
        total = self.OPTIMIZATION.synth_total_steps if pretrain else self.OPTIMIZATION.total_steps
        lr = self.OPTIMIZATION.optimizer.param_groups[0]["lr"]
        self.wandb_step = self.wandb_step_of(current_step)

        message = "{name} {epoch:>4}/{total} | {elapsed} | lr {lr:.3g} | train loss {loss:.4f}".format(
            name="Pretrain epoch" if pretrain else "Epoch", epoch=epoch, total=total, elapsed=misc.elapsed_time(self.start_time), lr=lr, loss=loss
        )
        if self.RUN.mode == "classification":
            message += " | top1 {top1:.2f} | top10 {top10:.2f}".format(top1=top1, top10=top10)
        self.logger.info(message)

        # the keys of the saved statistics
        dict = {"train_loss": loss}
        if self.RUN.mode == "classification":
            dict.update({"train_top1": top1, "train_top10": top10})

        # on wandb the pretraining on synthetic data has its own keys
        prefix = "pretrain_" if pretrain else ""
        wandb.log({**{prefix + name: value for name, value in dict.items()}, prefix + "epoch": epoch, "lr": lr}, step=self.wandb_step)

        if not pretrain:
            save_dict = misc.accm_values_convert_dict(list_dict=self.loss_list_dict, value_dict=dict, step=epoch, interval=self.RUN.print_every)
            misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name, "train"), name="losses", dictionary=save_dict)

    # -----------------------------------------------------------------------------
    # evaluate model.
    # -----------------------------------------------------------------------------
    def evaluate(self, step, writing=True, training=False):
        """
        Validation of epoch `step` (0-based) when `training`, or the evaluation on the test split of the checkpoint
        of epoch `step`. The best checkpoint is the one with the lowest validation loss.
        """
        pretrain = training and self.phase == "pretrain"
        is_best = False

        dataloader = self.valid_dataloader if training else self.test_dataloader
        self.set_epoch(dataloader, step)
        top1_acc, top10_acc, mpjpe, loss = self.evaluate_step(dataloader)

        # the keys of the saved statistics; validation used the "test_" names, so they are kept for old runs
        metric_dict = {
            "test_loss": loss,
            "test_mpjpe": mpjpe,
            "test_top1": top1_acc,
            "test_top10": top10_acc,
        }
        # on wandb: "valid_*" for validation, "pretrain_valid_*" for the validation after the synthetic pretraining
        # and "test_*" for the final evaluation on the test split
        prefix = "pretrain_valid" if pretrain else "valid" if training else "test"
        wandb_dict = {"{}_{}".format(prefix, name[len("test_") :]): value for name, value in metric_dict.items() if value is not None}

        if self.global_rank == 0:
            if self.best_loss is None or loss <= self.best_loss:
                self.best_loss, self.best_mpjpe, self.best_t1acc, self.best_t10acc, self.best_step, is_best = loss, mpjpe, top1_acc, top10_acc, step, True

            if self.RUN.mode == "classification":
                metrics = "loss {loss:.4f} | top1 {top1:.2f} | top10 {top10:.2f}".format(loss=loss, top1=top1_acc, top10=top10_acc)
                best = "loss {loss:.4f}, top1 {top1:.2f}".format(loss=self.best_loss, top1=self.best_t1acc)
            else:
                metrics = "loss {loss:.4f} | mpjpe {mpjpe:.4f}".format(loss=loss, mpjpe=mpjpe)
                best = "loss {loss:.4f}, mpjpe {mpjpe:.4f}".format(loss=self.best_loss, mpjpe=self.best_mpjpe)

            if pretrain:
                self.logger.info(
                    "Pretrain epoch {epoch:>4}/{total} | valid {metrics}".format(epoch=step + 1, total=self.OPTIMIZATION.synth_total_steps, metrics=metrics)
                )
            elif training:
                # "best" is the epoch with the lowest validation loss so far; its checkpoint is the one tested at the end
                self.logger.info(
                    "Epoch {epoch:>4}/{total} | valid {metrics} | best: epoch {best_epoch} ({best}){new}".format(
                        epoch=step + 1,
                        total=self.OPTIMIZATION.total_steps,
                        metrics=metrics,
                        best_epoch=self.best_step + 1,
                        best=best,
                        new=" *" if is_best else "",
                    )
                )
            else:
                self.logger.info(
                    "Test of the checkpoint of epoch {epoch} on {n} clips | test {metrics}".format(epoch=step + 1, n=len(dataloader.dataset), metrics=metrics)
                )

            if writing:
                # validation goes to the wandb step of its epoch, the test to the last step of the run
                wandb.log(wandb_dict, step=self.wandb_step_of(step) if training else self.wandb_step)

            if training and not pretrain:
                # the index of this validation among those of the run: every save_every epochs, and the last epoch
                index = math.ceil((step + 1) / self.RUN.save_every) * self.RUN.save_every
                save_dict = misc.accm_values_convert_dict(
                    list_dict=self.metric_dict_during_train, value_dict=metric_dict, step=index, interval=self.RUN.save_every
                )
                misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name, "valid"), name="test_stats", dictionary=save_dict)
            elif not training:
                save_dict = misc.accm_values_convert_dict(list_dict=self.metric_dict_during_final_eval, value_dict=metric_dict, step=None, interval=None)
                misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name, "test"), name="test_stats", dictionary=save_dict)

        self.model.train()
        return is_best

    # -----------------------------------------------------------------------------
    # save the trained model.
    # -----------------------------------------------------------------------------
    def save(self, step, is_best):
        when = "best" if is_best is True else "current"
        self.model.eval()
        model = misc.peel_model(self.model)

        states = {
            "state_dict": model.state_dict(),
            "optimizer": self.OPTIMIZATION.optimizer.state_dict(),
            "seed": self.RUN.seed,
            "run_name": self.run_name,
            "step": step,
            "epoch": self.epoch_counter,
            "best_step": self.best_step,
            "best_loss": self.best_loss,
            "best_mpjpe": self.best_mpjpe,
            "best_t1acc": self.best_t1acc,
            "best_t10acc": self.best_t10acc,
            "best_loss_ckpt": self.RUN.ckpt_dir,
            "resume": self.resume_state(),
        }

        misc.save_model(model=self.MODEL.backbone, when=when, step=step, ckpt_dir=self.RUN.ckpt_dir, states=states)

        if when == "best":
            misc.save_model(model=self.MODEL.backbone, when="current", step=step, ckpt_dir=self.RUN.ckpt_dir, states=states)

        if self.global_rank == 0 and self.logger:
            self.logger.info("Save model to {}".format(self.RUN.ckpt_dir))

        self.model.train()

    # -----------------------------------------------------------------------------
    # evaluate model on a given dataset (valid or test)
    # -----------------------------------------------------------------------------
    @torch.no_grad()
    def classification_evaluate_step(self, dataloader):
        self.model.eval()
        top1_acc, top10_acc, loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        for values, labels in dataloader:
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                # load values and labels onto the GPU memory
                values = values.to(self.local_rank).float()
                labels = labels.to(self.local_rank)

                if self.DATA.pad_mode == 'pad':
                    masks = torch.isnan(values)
                    values = torch.nan_to_num(values)
                else:
                    masks = None

                values = self.transform(values)

                # get model output for the current batch
                outputs = self.model(values, masks)

            # calculate Cross Entropy Loss
            batch_loss = self.loss(outputs, labels)

            # calculate topk
            acc1, acc10 = misc.accuracy(outputs.data, labels, topk=(1, 10))

            # accumulate topk
            top1_acc.update(acc1.item(), values.size(0))
            top10_acc.update(acc10.item(), values.size(0))

            # accumulate loss
            loss.update(batch_loss.item(), values.size(0))

        top1 = top1_acc.avg
        top10 = top10_acc.avg

        return top1, top10, None, loss.avg

    @torch.no_grad()
    def prediction_evaluate_step(self, dataloader):
        self.model.eval()
        loss, mpjpe = misc.AverageMeter(), misc.AverageMeter()
        for values, targets in dataloader:
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                # load values and targets onto the GPU memory
                values = values.to(self.local_rank).float()
                targets = targets.to(self.local_rank).float()

                if self.RUN.reverse:
                    r_values = torch.flip(values, [1])
                    r_targets = torch.flip(targets, [1])
                    values = r_targets
                    targets = r_values

                outputs = misc.generate_poses(self.model, values, self.transform, self.DATA.target_len)

            # calculate Loss
            batch_loss = self.loss(outputs, targets)

            # accumulate loss
            loss.update(batch_loss.item(), values.size(0))

            # calculate Mean Per Joint Position Error (MPJPE)
            e = misc.mpjpe(outputs, targets)

            # accumulate MPJPE
            mpjpe.update(e.item(), values.size(0))

        return None, None, mpjpe.avg, loss.avg

    @torch.no_grad()
    def cond_prediction_evaluate_step(self, dataloader):
        self.model.eval()
        loss, mpjpe = misc.AverageMeter(), misc.AverageMeter()
        for values, targets, labels in dataloader:
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                # load values and labels onto the GPU memory
                values = values.to(self.local_rank).float()
                targets = targets.to(self.local_rank).float()
                labels = labels.to(self.local_rank)

                if self.RUN.reverse:
                    r_values = torch.flip(values, [1])
                    r_targets = torch.flip(targets, [1])
                    values = r_targets
                    targets = r_values

                outputs = misc.cond_generate_poses(self.model, values, labels, self.transform, self.DATA.target_len)

            # calculate Cross Entropy Loss
            batch_loss = self.loss(outputs, targets)

            # accumulate loss
            loss.update(batch_loss.item(), values.size(0))

            # calculate Mean Per Joint Position Error (MPJPE)
            e = misc.mpjpe(outputs, targets)

            # accumulate MPJPE
            mpjpe.update(e.item(), values.size(0))

        return None, None, mpjpe.avg, loss.avg

    # -----------------------------------------------------------------------------
    # visualize fake poses for monitoring purpose.
    # -----------------------------------------------------------------------------
    @torch.no_grad()
    def visualize_fake_poses(self, step):
        if self.global_rank == 0:
            self.logger.info("Visualize fake poses.")

        # generate fake
        self.model.eval()
        if self.RUN.mode == "prediction":
            values, targets = next(iter(self.test_dataloader))
        elif self.RUN.mode == "cond_prediction":
            values, targets, labels = next(iter(self.test_dataloader))
        else:
            raise NotImplementedError
        with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
            # load values and labels onto the GPU memory
            values = values.to(self.local_rank).float()
            targets = targets.to(self.local_rank).float()
            if self.RUN.mode == "cond_prediction":
                labels = labels.to(self.local_rank)

            if self.RUN.reverse:
                r_values = torch.flip(values, [1])
                r_targets = torch.flip(targets, [1])
                values = r_targets
                targets = r_values

            # get model output for the current batch
            if self.RUN.mode == "prediction":
                outputs = misc.generate_poses(self.model, values, self.transform, self.DATA.target_len)
            elif self.RUN.mode == "cond_prediction":
                outputs = misc.cond_generate_poses(self.model, values, labels, self.transform, self.DATA.target_len)

            outputs = torch.cat((values, outputs), dim=-2)

            if self.RUN.reverse:
                outputs = torch.flip(outputs, [1])

        self.animate_save_poses(outputs[0], step, 'generated')

        if step == 0:
            self.animate_save_poses(values[0], step, 'value')
            self.animate_save_poses(targets[0], step, 'target')

    # -----------------------------------------------------------------------------
    # visualize samples of fake and real poses.
    # -----------------------------------------------------------------------------
    @torch.no_grad()
    def visualize_samples_poses(self, step):
        if self.global_rank == 0:
            self.logger.info("Visualize fake poses.")

        # generate fake
        self.model.eval()
        if self.RUN.mode == "prediction":
            values, targets = next(iter(self.test_dataloader))
        elif self.RUN.mode == "cond_prediction":
            values, targets, labels = next(iter(self.test_dataloader))
        else:
            raise NotImplementedError
        with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
            # load values and labels onto the GPU memory
            values = values.to(self.local_rank).float()
            targets = targets.to(self.local_rank).float()
            if self.RUN.mode == "cond_prediction":
                labels = labels.to(self.local_rank)

            if self.RUN.reverse:
                r_values = torch.flip(values, [1])
                r_targets = torch.flip(targets, [1])
                values = r_targets
                targets = r_values

            # get model output for the current batch
            if self.RUN.mode == "prediction":
                outputs = misc.generate_poses(self.model, values, self.transform, self.DATA.target_len)
            elif self.RUN.mode == "cond_prediction":
                outputs = misc.cond_generate_poses(self.model, values, labels, self.transform, self.DATA.target_len)

            if self.RUN.twin_generator:
                r_values = torch.flip(targets.to(self.local_rank).float(), [1])

                if self.RUN.mode == "prediction":
                    r_outputs = misc.generate_poses(self.model, r_values, self.transform, self.DATA.target_len)
                elif self.RUN.mode == "cond_prediction":
                    r_outputs = misc.cond_generate_poses(self.model, r_values, labels, self.transform, self.DATA.target_len)

                # replace first half with generated frames
                output_values = torch.flip(r_outputs, [1])
            else:
                output_values = values

            outputs = torch.cat((output_values, outputs), dim=-2)

            if self.RUN.reverse:
                outputs = torch.flip(outputs, [1])

        for i in range(self.RUN.ss_num):
            self.animate_save_poses(outputs[i], step, 'generated-sample-{}'.format(i))

        if step == 0:
            for i in range(self.RUN.ss_num):
                self.animate_save_poses(values[i], step, 'value-sample-{}'.format(i))
                self.animate_save_poses(targets[i], step, 'target-sample-{}'.format(i))
                self.animate_save_poses(torch.cat((values[i], targets[i]), dim=-2), step, 'original-sample-{}'.format(i))

    def split_keypoints(self, raw_keypoints):
        input = torch.unflatten(raw_keypoints, 1, (self.DATA.num_keypoints, 3)).detach().cpu()
        body_kp = misc.i_normalize(input[:, : self.len_body_kp, :], 'pose')
        rhand_kp = misc.i_normalize(input[:, self.len_body_kp : (self.len_body_kp + 21), :], 'right_hand')
        lhand_kp = misc.i_normalize(input[:, (self.len_body_kp + 21) : (self.len_body_kp + 21 + 21), :], 'left_hand')
        face_kp = misc.i_normalize(input[:, (self.len_body_kp + 21 + 21) :, :], 'face')
        return body_kp, rhand_kp, lhand_kp, face_kp

    def animate_save_poses(self, keypoints, step, mode):
        body_kp, rhand_kp, lhand_kp, face_kp = self.split_keypoints(keypoints)
        ani = misc.animate_all_keypoints(body_kp, rhand_kp, lhand_kp, face_kp)

        save_path = join(self.RUN.save_dir, "figures/{run_name}/{mode}_keypoints_{step}.gif".format(run_name=self.run_name, mode=mode, step=step))

        misc.save_gif(ani=ani, save_path=save_path, logger=self.logger, logging=self.global_rank == 0 and self.logger)

        if self.wandb_step:
            wandb.log({"{mode}_poses".format(mode=mode): wandb.Video(save_path, format='gif')}, step=self.wandb_step)
        else:
            wandb.log({"{mode}_poses".format(mode=mode): wandb.Video(save_path, format='gif')})

    # -----------------------------------------------------------------------------
    # visualize real poses for monitoring purpose.
    # -----------------------------------------------------------------------------
    def visualize_real_poses(self):
        if self.global_rank == 0:
            self.logger.info("Visualize real poses.")

        self.animate_save_poses(torch.cat((self.test_dataloader.dataset[0][0], self.test_dataloader.dataset[0][1]), dim=-2), '', 'real')

    def prepare_generation(self):
        self.gen_curr_id = 0
        poses = ['pose', 'right_hand', 'left_hand', 'face']
        for pose in poses:
            directory = join(self.base_save_dir, "poses/{pose}/".format(pose=pose))
            misc.prepare_save_folder(directory)

    @torch.no_grad()
    def save_dataset(self, dataloader, signs):
        """
        Generate a clip from every clip of `dataloader`, whose batches each hold clips of one class
        (misc.ClassBatchSampler), and save it. `signs` maps a class index to its sign.
        """
        if self.global_rank == 0:
            self.logger.info("Generate {n} poses per class for {c} classes.".format(n=self.RUN.sd_num * self.OPTIMIZATION.batch_size, c=len(signs)))

        self.model.eval()
        ids, current = [], None

        for values, targets, labels in dataloader:
            label = int(labels[0])
            if label != current:
                if current is not None:
                    self.update_metadata(ids, signs[current])
                    ids = []
                current = label
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                # load values and labels onto the GPU memory
                values = values.to(self.local_rank).float()
                labels = labels.to(self.local_rank)

                outputs = misc.cond_generate_poses(self.model, values, labels, self.transform, self.DATA.target_len)

                if self.RUN.twin_generator:
                    r_values = torch.flip(targets.to(self.local_rank).float(), [1])

                    r_outputs = misc.cond_generate_poses(self.model, r_values, labels, self.transform, self.DATA.target_len)

                    # replace first half with generated frames
                    values = torch.flip(r_outputs, [1])

                for value, output in zip(values, outputs, strict=False):
                    self.save_keypoints(value, output)
                    ids.append(str(self.gen_curr_id))
                    self.gen_curr_id += 1

        if current is not None:
            self.update_metadata(ids, signs[current])

        if self.global_rank == 0:
            self.logger.info("Dataset saved: {n} poses.".format(n=self.gen_curr_id))

    def update_metadata(self, ids, sign):
        sti = pl.read_csv(os.path.join(self.RUN.data_dir, 'metadata', 'sign_to_index.csv'))
        self.metadata['id'] = self.metadata['id'] + ids
        self.metadata['sign'] = self.metadata['sign'] + [sti.filter(sti['class'] == sign)['sign'].item() for _ in ids]
        self.metadata['signer'] = self.metadata['signer'] + ['Bender' for _ in ids]
        self.metadata['start'] = self.metadata['start'] + [0 for _ in ids]
        self.metadata['end'] = self.metadata['end'] + [self.DATA.input_size[0] for _ in ids]

    def save_keypoints(self, value_kp, output_kp):
        value_body_kp, value_rhand_kp, value_lhand_kp, value_face_kp = self.split_keypoints(value_kp)
        output_body_kp, output_rhand_kp, output_lhand_kp, output_face_kp = self.split_keypoints(output_kp)

        with open(join(self.base_save_dir, 'poses/pose/{i}.npy'.format(i=self.gen_curr_id)), 'wb') as f:
            np.save(f, torch.cat((value_body_kp, output_body_kp), dim=0))
        with open(join(self.base_save_dir, 'poses/right_hand/{i}.npy'.format(i=self.gen_curr_id)), 'wb') as f:
            np.save(f, torch.cat((value_rhand_kp, output_rhand_kp), dim=0))
        with open(join(self.base_save_dir, 'poses/left_hand/{i}.npy'.format(i=self.gen_curr_id)), 'wb') as f:
            np.save(f, torch.cat((value_lhand_kp, output_lhand_kp), dim=0))
        with open(join(self.base_save_dir, 'poses/face/{i}.npy'.format(i=self.gen_curr_id)), 'wb') as f:
            np.save(f, torch.cat((value_face_kp, output_face_kp), dim=0))

    def save_metadata(self):
        # copy sign_to_index from original dataset
        src = os.path.join(self.RUN.data_dir, 'metadata', 'sign_to_index.csv')
        dst_dir = os.path.join(self.base_save_dir, 'metadata')
        dst = os.path.join(dst_dir, 'sign_to_index.csv')

        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, dst)

        # save metadata
        df = pl.DataFrame(self.metadata)
        df.write_csv(os.path.join(self.base_save_dir, 'instances.csv'), separator=",")

        # save ids to root_ids .json
        splits_dir = os.path.join(dst_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        with open(os.path.join(splits_dir, 'train.json'), "w") as file:
            json.dump(self.metadata['id'], file)
