# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/worker.py

from os.path import join

from datetime import datetime
import torch
import torch.distributed as dist

import utils.misc as misc
import wandb

SAVE_FORMAT = "step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth"

LOG_FORMAT = ("Step: {step:>6} "
              "Progress: {progress:<.1%} "
              "Elapsed: {elapsed} "
              "Loss: {loss:<.4} "
              "Top1: {top1:<.4} "
              "Top5: {top5:<.4} ")


class WORKER(object):
    def __init__(self, cfgs, run_name, model, train_dataloader, valid_dataloader, test_dataloader, 
                 global_rank, local_rank, logger, best_step, best_loss, 
                 loss_list_dict, metric_dict_during_train):
        self.cfgs = cfgs
        self.run_name = run_name
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.logger = logger
        self.best_step = best_step
        self.best_loss = best_loss
        self.loss_list_dict = loss_list_dict
        self.metric_dict_during_train = metric_dict_during_train
        self.metric_dict_during_final_eval = {}

        #self.cfgs.define_augments(local_rank)
        self.cfgs.define_losses()
        self.DATA = cfgs.DATA
        self.MODEL = cfgs.MODEL
        self.LOSS = cfgs.LOSS
        self.OPTIMIZATION = cfgs.OPTIMIZATION
        self.PRE = cfgs.PRE
        self.AUG = cfgs.AUG
        self.RUN = cfgs.RUN
        self.MISC = cfgs.MISC
        self.DDP = self.RUN.distributed_data_parallel

        self.ce_loss = torch.nn.CrossEntropyLoss()

        if self.DDP:
            self.group = dist.new_group([n for n in range(self.OPTIMIZATION.world_size)])

        if self.RUN.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.global_rank == 0:
            wandb.init(project=self.RUN.project,
                       entity=self.RUN.entity,
                       name=self.run_name,
                       dir=self.RUN.save_dir,
                       resume=self.best_step > 0)

        self.start_time = datetime.now()

    def prepare_train_iter(self, epoch_counter):
        self.epoch_counter = epoch_counter
        if self.DDP:
            self.train_dataloader.sampler.set_epoch(self.epoch_counter)
        self.train_iter = iter(self.train_dataloader)

    # -----------------------------------------------------------------------------
    # train model
    # -----------------------------------------------------------------------------
    def train_step(self):
        # make the model be trainable before starting training
        self.model.train()
        # toggle gradients
        misc.toggle_grad(model=self.model, grad=True)
        # create accumulators
        valid_top1_acc, valid_top5_acc, valid_loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        # sample real values and labels, then train for an epoch
        for values, labels in self.train_dataloader:
            self.OPTIMIZATION.optimizer.zero_grad()
            with torch.cuda.amp.autocast() if self.RUN.mixed_precision else misc.dummy_context_mgr() as mpc:
                # load values and labels onto the GPU memory
                values = values.to(self.local_rank, non_blocking=True)
                labels = labels.to(self.local_rank, non_blocking=True)

                # get model output for the current batch
                outputs = self.model(values)

            # calculate Cross Entropy Loss
            loss = self.ce_loss(outputs, labels)

            # accumulate gradients of the model
            if self.RUN.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # update the model using the pre-defined optimizer
            if self.RUN.mixed_precision:
                self.scaler.step(self.OPTIMIZATION.optimizer)
                self.scaler.update()
            else:
                self.OPTIMIZATION.optimizer.step()

            # calculate topk
            valid_acc1, valid_acc5 = misc.accuracy(outputs.data, labels, topk=(1, 5))

            # accumulate loss and topk
            valid_loss.update(loss.item(), values.size(0))
            valid_top1_acc.update(valid_acc1.item(), values.size(0))
            valid_top5_acc.update(valid_acc5.item(), values.size(0))

        if self.local_rank == 0:
            self.logger.info("Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t"
                             "Top 5-acc {top5.val:.4f} ({top5.avg:.4f})".format(top1=valid_top1_acc, top5=valid_top5_acc))

        if self.RUN.empty_cache:
            torch.cuda.empty_cache()
        return valid_top1_acc.avg, valid_top5_acc.avg, valid_loss.avg


    # -----------------------------------------------------------------------------
    # log training statistics
    # -----------------------------------------------------------------------------
    def log_train_statistics(self, current_step, loss, top1, top5):
        self.wandb_step = current_step + 1

        log_message = LOG_FORMAT.format(
            step=current_step + 1,
            progress=(current_step + 1) / self.OPTIMIZATION.total_steps,
            elapsed=misc.elapsed_time(self.start_time),
            loss=loss.item(),
            top1=top1.item(),
            top5=top5.item()
        )
        self.logger.info(log_message)

        # save loss values in wandb event file and .npz format
        dict = {
            "loss": loss.item(),
            "top1": top1.item(),
            "top5": top5.item(),
        }

        wandb.log(dict, step=self.wandb_step)

        save_dict = misc.accm_values_convert_dict(list_dict=self.loss_list_dict,
                                            value_dict=dict,
                                            step=current_step + 1,
                                            interval=self.RUN.print_every)

        misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name, "train"),
                           name="losses",
                           dictionary=save_dict)

    # -----------------------------------------------------------------------------
    # evaluate model.
    # -----------------------------------------------------------------------------
    def evaluate(self, step, writing=True, training=False):
        if self.global_rank == 0:
            self.logger.info("Start {mode} ({step} Step): {run_name}".format(mode='Validation' if training else 'Testing',step=step, run_name=self.run_name))

        is_best = False
        metric_dict = {}

        top1_acc, top5_acc, loss = self.evaluate_step(self.valid_dataloader if training else self.test_dataloader)

        if self.global_rank == 0:
            self.logger.info("Top 1-acc {top1:.4f}\t"
                             "Top 5-acc {top5:.4f}"
                             "Loss {loss}".format(top1=top1_acc, top5=top5_acc, loss=loss))
            if self.best_loss is None or loss <= self.best_loss:
                self.best_loss, self.best_step, is_best = loss, step, True
            if writing:
                wandb.log({"Loss": loss}, step=self.wandb_step)
                metric_dict.update({"Loss": loss})
            if training:
                self.logger.info("Best Loss (Step: {step}): {loss}".format(
                    step=self.best_step, loss=self.best_loss))

        if self.global_rank == 0:
            if training:
                save_dict = misc.accm_values_convert_dict(list_dict=self.metric_dict_during_train,
                                                            value_dict=metric_dict,
                                                            step=step,
                                                            interval=self.RUN.save_every)
            else:
                save_dict = misc.accm_values_convert_dict(list_dict=self.metric_dict_during_final_eval,
                                                            value_dict=metric_dict,
                                                            step=None,
                                                            interval=None)

            misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name, "valid" if training else "test"),
                                name="losses",
                                dictionary=save_dict)

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
            "best_loss_ckpt": self.RUN.ckpt_dir
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
    def evaluate_step(self, dataloader):
        self.model.eval()
        top1_acc, top5_acc, loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        for values, labels in dataloader:
            # load values and labels onto the GPU memory
            values = values.to(self.local_rank)
            labels = labels.to(self.local_rank)

            # get model output for the current batch
            outputs = self.model(values)

            # calculate Cross Entropy Loss
            loss = self.ce_loss(outputs, labels)

            # calculate topk
            acc1, acc5 = misc.accuracy(outputs.data, labels, topk=(1, 5))

            # accumulate loss and topk
            loss.update(loss.item(), values.size(0))
            top1_acc.update(acc1.item(), values.size(0))
            top5_acc.update(acc5.item(), values.size(0))

        return top1_acc.avg, top5_acc.avg, loss.avg
