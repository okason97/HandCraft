# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/worker.py

from os.path import join

from datetime import datetime
import torch
import torch.distributed as dist
import numpy as np
import itertools

import utils.misc as misc
import wandb


class WORKER(object):
    def __init__(self, cfgs, run_name, model, train_dataloader, valid_dataloader, test_dataloader, 
                 global_rank, local_rank, logger, best_step, best_loss, best_mpjpe, best_t1acc, best_t10acc,
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
        self.best_mpjpe = best_mpjpe
        self.best_t1acc = best_t1acc
        self.best_t10acc = best_t10acc
        self.loss_list_dict = loss_list_dict
        self.metric_dict_during_train = metric_dict_during_train
        self.metric_dict_during_final_eval = {}
        self.wandb_step = None

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

        if "prediction" in cfgs.RUN.mode:
            self.dct_m, self.idct_m = misc.get_dct_matrix(cfgs.DATA.input_size[0])
            self.dct_m = torch.tensor(self.dct_m, dtype=torch.half).unsqueeze(0).to(self.local_rank)
            self.idct_m = torch.tensor(self.idct_m, dtype=torch.half).unsqueeze(0).to(self.local_rank)

        #self.cfgs.define_augments(local_rank)
        self.cfgs.define_losses()
        self.DATA = cfgs.DATA
        self.MODEL = cfgs.MODEL
        self.OPTIMIZATION = cfgs.OPTIMIZATION
        self.PRE = cfgs.PRE
        self.AUG = cfgs.AUG
        self.RUN = cfgs.RUN
        self.MISC = cfgs.MISC
        self.DDP = self.RUN.distributed_data_parallel

        self.loss = cfgs.LOSS.loss

        self.len_body_kp = 33 if self.DATA.poses[0][1] == 'all' else len(self.DATA.poses[0][1])

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

    def optimizer_step(self):
        if self.RUN.mixed_precision:
            self.scaler.step(self.OPTIMIZATION.optimizer)
            self.scaler.update()
        else:
            self.OPTIMIZATION.optimizer.step()
        if self.OPTIMIZATION.scheduler:
            self.OPTIMIZATION.scheduler.step()

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
        for values, labels in self.train_dataloader:
            self.OPTIMIZATION.optimizer.zero_grad()
            with torch.cuda.amp.autocast() if self.RUN.mixed_precision else torch.autocast("cuda") as mpc:
                values = values.to(self.local_rank, non_blocking=True)
                labels = labels.to(self.local_rank, non_blocking=True)

                # get model output for the current batch
                outputs = self.model(values)

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

        if self.local_rank == 0:
            self.logger.info("Train Top 1-acc {top1.avg:.4f}\t"
                            "Train Top 10-acc {top10.avg:.4f}\t"
                            "Train Loss {loss.avg:.4f}".format(top1=valid_top1_acc, top10=valid_top10_acc, loss=valid_loss))

        # apply late dropout when reaching the indicated step
        self.apply_l_drop(step)

        if self.RUN.empty_cache:
            torch.cuda.empty_cache()

        top1 = valid_top1_acc.avg
        top10 = valid_top10_acc.avg
        return top1, top10, valid_loss.avg
    
    def prediction_train_step(self, step):
        # make the model be trainable before starting training
        self.model.train()
        # toggle gradients
        misc.toggle_grad(model=self.model, grad=True)
        # create accumulators
        valid_loss = misc.AverageMeter()
        # sample real values and targets, then train for an epoch
        for values, targets in self.train_dataloader:
            self.OPTIMIZATION.optimizer.zero_grad()
            with torch.cuda.amp.autocast() if self.RUN.mixed_precision else torch.autocast("cuda") as mpc:
                values = values.to(self.local_rank, non_blocking=True)
                targets = targets.to(self.local_rank, non_blocking=True)

                offset = values[:, -1:]
                # apply Discrete Cosine Transform 
                values = torch.matmul(self.dct_m, values)

                # get model output for the current batch
                outputs = self.model(values)

                # apply Inverse Discrete Cosine Transform 
                values = torch.matmul(self.idct_m, outputs)

                # apply output over the last frame as offset
                outputs = outputs[:, :self.DATA.target_len] + offset                

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

        if self.local_rank == 0:
            self.logger.info("Train Loss {loss.avg:.4f}".format(loss=valid_loss))

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
        for values, targets, labels in self.train_dataloader:
            self.OPTIMIZATION.optimizer.zero_grad()
            with torch.cuda.amp.autocast() if self.RUN.mixed_precision else torch.autocast("cuda") as mpc:
                # load values, targets and labels onto the GPU memory
                values = values.to(self.local_rank, non_blocking=True)
                targets = targets.to(self.local_rank, non_blocking=True)
                labels = labels.to(self.local_rank, non_blocking=True)

                offset = values[:, -1:]
                # apply Discrete Cosine Transform 
                values = torch.matmul(self.dct_m, values)

                # get model output for the current batch
                outputs = self.model(values, labels)

                # apply Inverse Discrete Cosine Transform 
                values = torch.matmul(self.idct_m, outputs)

                # apply output over the last frame as offset
                outputs = outputs[:, :self.DATA.target_len] + offset                

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

        if self.local_rank == 0:
            self.logger.info("Train Loss {loss.avg:.4f}".format(loss=valid_loss))

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
    def log_train_statistics(self, current_step, loss, top1=None, top10=None):
        self.wandb_step = current_step + 1

        if self.RUN.mode == "classification":
            LOG_FORMAT = ("Step: {step:>6} "
                        "Progress: {progress:<.1%} "
                        "Elapsed: {elapsed} "
                        "Loss: {loss:<.4} "
                        "Top1: {top1:<.4} "
                        "Top10: {top10:<.4} ")
            log_message = LOG_FORMAT.format(
                step=current_step + 1,
                progress=(current_step + 1) / self.OPTIMIZATION.total_steps,
                elapsed=misc.elapsed_time(self.start_time),
                loss=loss,
                top1=top1,
                top10=top10
            )
        else:
            LOG_FORMAT = ("Step: {step:>6} "
                        "Progress: {progress:<.1%} "
                        "Elapsed: {elapsed} "
                        "Loss: {loss:<.4} ")
            log_message = LOG_FORMAT.format(
                step=current_step + 1,
                progress=(current_step + 1) / self.OPTIMIZATION.total_steps,
                elapsed=misc.elapsed_time(self.start_time),
                loss=loss,
            )
        self.logger.info(log_message)

        # save loss values in wandb event file and .npz format
        if self.RUN.mode == "classification":
            dict = {
                "train_loss": loss,
                "train_top1": top1,
                "train_top10": top10,
            }
        else:
            dict = {
                "train_loss": loss,
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

        top1_acc, top10_acc, mpjpe, loss = self.evaluate_step(self.valid_dataloader if training else self.test_dataloader)

        metric_dict = {
            "test_loss": loss,
            "test_mpjpe": mpjpe,
            "test_top1": top1_acc,
            "test_top10": top10_acc,
        }

        if self.global_rank == 0:
            if self.RUN.mode == "classification":
                self.logger.info("Test Top 1-acc {top1:.4f}\t"
                                "Test Top 10-acc {top10:.4f}\t"
                                "Test Loss {loss}".format(top1=top1_acc, top10=top10_acc, loss=loss))
            else:
                self.logger.info("Test Loss {loss}\t"
                                 "Test Loss {mpjpe}".format(loss=loss, mpjpe=mpjpe))
            if self.best_loss is None or loss <= self.best_loss:
                self.best_loss, self.best_mpjpe, self.best_t1acc, self.best_t10acc, self.best_step, is_best = loss, mpjpe, top1_acc, top10_acc, step, True
            if writing:
                wandb.log(metric_dict, step=self.wandb_step)
            if training:
                if self.RUN.mode == "classification":
                    self.logger.info("Best Top 1-acc {top1:.4f}\t"
                                    "Best Top 10-acc {top10:.4f}\t"
                                    "Best Loss (Step: {step}): {loss}".format(
                        step=self.best_step, loss=self.best_loss, top1=top1_acc, top10=top10_acc))
                else:
                    self.logger.info("Best MPJPE {mpjpe}\t"
                                     "Best Loss (Step: {step}): {loss}\t".format(
                        step=self.best_step, mpjpe=self.best_mpjpe, loss=self.best_loss))

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
                                name="test_stats",
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
            "best_mpjpe": self.best_mpjpe,
            "best_t1acc": self.best_t1acc,
            "best_t10acc": self.best_t10acc,
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
    def classification_evaluate_step(self, dataloader):
        self.model.eval()
        top1_acc, top10_acc, loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        for values, labels in dataloader:
            with torch.autocast("cuda") as mpc:
                # load values and labels onto the GPU memory
                values = values.to(self.local_rank)
                labels = labels.to(self.local_rank)

                # get model output for the current batch
                outputs = self.model(values)

            # calculate Cross Entropy Loss
            l = self.loss(outputs, labels)

            # calculate topk
            acc1, acc10 = misc.accuracy(outputs.data, labels, topk=(1, 10))

            # accumulate topk
            top1_acc.update(acc1.item(), values.size(0))
            top10_acc.update(acc10.item(), values.size(0))

            # accumulate loss
            loss.update(l.item(), values.size(0))

        top1 = top1_acc.avg
        top10 = top10_acc.avg

        return top1, top10, None, loss.avg

    def prediction_evaluate_step(self, dataloader):
        self.model.eval()
        loss, mpjpe = misc.AverageMeter(), misc.AverageMeter()
        for values, targets in dataloader:
            with torch.autocast("cuda") as mpc:
                # load values and targets onto the GPU memory
                values = values.to(self.local_rank)
                targets = targets.to(self.local_rank)

                offset = values[:, -1:]
                # apply Discrete Cosine Transform 
                values = torch.matmul(self.dct_m, values)

                # get model output for the current batch
                outputs = self.model(values)

                # apply Inverse Discrete Cosine Transform 
                values = torch.matmul(self.idct_m, outputs)

                # apply output over the last frame as offset
                outputs = outputs[:, :self.DATA.target_len] + offset                

            # calculate Loss
            l = self.loss(outputs, targets)

            # accumulate loss
            loss.update(l.item(), values.size(0))

            # calculate Mean Per Joint Position Error (MPJPE)
            e = misc.mpjpe(outputs, targets)

            # accumulate MPJPE
            mpjpe.update(e.item(), values.size(0))

        return None, None, mpjpe.avg, loss.avg

    def cond_prediction_evaluate_step(self, dataloader):
        self.model.eval()
        loss, mpjpe = misc.AverageMeter(), misc.AverageMeter()
        for values, targets, labels in dataloader:
            with torch.autocast("cuda") as mpc:
                # load values and labels onto the GPU memory
                values = values.to(self.local_rank)
                targets = targets.to(self.local_rank)
                labels = labels.to(self.local_rank)

                offset = values[:, -1:]
                # apply Discrete Cosine Transform 
                values = torch.matmul(self.dct_m, values)

                # get model output for the current batch
                outputs = self.model(values, labels)

                # apply Inverse Discrete Cosine Transform 
                values = torch.matmul(self.idct_m, outputs)

                # apply output over the last frame as offset
                outputs = outputs[:, :self.DATA.target_len] + offset

            # calculate Cross Entropy Loss
            l = self.loss(outputs, targets)

            # accumulate loss
            loss.update(l.item(), values.size(0))

            # calculate Mean Per Joint Position Error (MPJPE)
            e = misc.mpjpe(outputs, targets)

            # accumulate MPJPE
            mpjpe.update(e.item(), values.size(0))

        return None, None, mpjpe.avg, loss.avg

    # -----------------------------------------------------------------------------
    # visualize fake poses for monitoring purpose.
    # -----------------------------------------------------------------------------
    def visualize_fake_poses(self, step):
        if self.global_rank == 0:
            self.logger.info("Visualize fake poses.")

        #generate fake
        self.model.eval()
        if self.RUN.mode == "prediction":
            values, targets = next(iter(self.test_dataloader))
        elif self.RUN.mode == "cond_prediction":
            values, targets, labels = next(iter(self.test_dataloader))
        else:
            raise NotImplementedError
        with torch.autocast("cuda") as mpc:
            # load values and labels onto the GPU memory
            values = values.to(self.local_rank)
            if self.RUN.mode == "cond_prediction":
                labels = labels.to(self.local_rank)

            offset = values[:, -1:]
            # apply Discrete Cosine Transform 
            values = torch.matmul(self.dct_m, values)

            # get model output for the current batch
            if self.RUN.mode == "prediction":
                outputs = self.model(values)
            elif self.RUN.mode == "cond_prediction":
                outputs = self.model(values, labels)

            # apply Inverse Discrete Cosine Transform 
            values = torch.matmul(self.idct_m, outputs)

            # apply output over the last frame as offset
            outputs = outputs[:, :self.DATA.target_len] + offset

        self.animate_save_poses(outputs[0], step, 'generated')

        if step == 0:
            self.animate_save_poses(values[0], step, 'value')

            self.animate_save_poses(targets[0], step, 'target')

    def split_keypoints(self, raw_keypoints):
        input = torch.unflatten(raw_keypoints, 1, (self.DATA.num_keypoints, 3)).detach().cpu()
        body_kp = misc.i_normalize(input[:,:self.len_body_kp,:], 'pose')
        rhand_kp = misc.i_normalize(input[:,self.len_body_kp:(self.len_body_kp+21),:], 'right_hand')
        lhand_kp = misc.i_normalize(input[:,(self.len_body_kp+21):(self.len_body_kp+21+21),:], 'left_hand')
        face_kp = misc.i_normalize(input[:,(self.len_body_kp+21+21):,:], 'face')
        return body_kp, rhand_kp, lhand_kp, face_kp

    def animate_save_poses(self, keypoints, step, mode):
        body_kp, rhand_kp, lhand_kp, face_kp = self.split_keypoints(keypoints)
        ani = misc.animate_all_keypoints(body_kp, rhand_kp, lhand_kp, face_kp)

        save_path = join(self.RUN.save_dir,
                        "figures/{run_name}/{mode}_keypoints_{step}.gif".format(run_name=self.run_name, mode=mode, step=step))

        misc.save_gif(ani=ani,
                      save_path=save_path,
                      logger=self.logger,
                      logging=self.global_rank == 0 and self.logger)

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

        self.animate_save_poses(self.test_dataloader.dataset[0][0], '', 'real')

    def save_dataset(self, class_dataloader):
        if self.global_rank == 0:
            self.logger.info("Save {s_dataset_len} generated poses.".format(
                s_dataset_len=self.RUN.sd_num))

        poses = ['pose', 'right_hand', 'left_hand', 'face']
        base_dir = join(self.RUN.save_dir, "generated_datasets/{run_name}".format(run_name=self.run_name))
        for pose in poses:
            directory = join(base_dir, "poses/{pose}/".format(pose=pose))
            misc.prepare_save_folder(directory)

        #generate fake
        self.model.eval()
        data_iter = itertools.cycle(class_dataloader)

        for i in range(self.RUN.sd_num):
            values, targets, labels = next(data_iter)
            with torch.autocast("cuda") as mpc:
                # load values and labels onto the GPU memory
                values = values.to(self.local_rank)
                labels = labels.to(self.local_rank)

                offset = values[:, -1:]
                # apply Discrete Cosine Transform 
                values = torch.matmul(self.dct_m, values)

                # get model output for the current batch
                outputs = self.model(values, labels)

                # apply Inverse Discrete Cosine Transform 
                values = torch.matmul(self.idct_m, outputs)

                # apply output over the last frame as offset
                outputs = outputs[:, :self.DATA.target_len] + offset

                self.save_keypoints(base_dir, outputs[0], i)

        if self.global_rank == 0:
            self.logger.info("Dataset saved.")

def save_keypoints(self, base_dir, keypoints, i):
    body_kp, rhand_kp, lhand_kp, face_kp = self.split_keypoints(keypoints)

    with open(join(base_dir, 'poses/pose/{i}.npy'.format(i=i)), 'wb') as f:
        np.save(f, body_kp)
    with open(join(base_dir, 'poses/right_hand/{i}.npy'.format(i=i)), 'wb') as f:
        np.save(f, rhand_kp)
    with open(join(base_dir, 'poses/left_hand/{i}.npy'.format(i=i)), 'wb') as f:
        np.save(f, lhand_kp)
    with open(join(base_dir, 'poses/face/{i}.npy'.format(i=i)), 'wb') as f:
        np.save(f, face_kp)
