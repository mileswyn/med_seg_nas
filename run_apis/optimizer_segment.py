import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dropped_model_segment import Dropped_Network

from tools import utils
from tools.metric_seg import SegmentationMetric
from tools.loss_seg1 import softmax_helper


# ###promise12 experiment
# class Optimizer(object):
#
#     def __init__(self, model, criterion, config):
#         self.config = config
#         self.weight_sample_num = self.config.search_params.weight_sample_num
#         self.criterion = criterion
#         self.Dropped_Network = lambda model: Dropped_Network(
#             model, softmax_temp=config.search_params.softmax_temp)
#         # self.weight_metric_train = SegmentationMetric(2)
#         # self.weight_train_loss_meter = utils.AverageMeter()
#         # self.arch_train_loss_meter = utils.AverageMeter()
#         # self.arch_metric_train = SegmentationMetric(2)
#         arch_params_id = list(map(id, model.module.arch_parameters))
#         weight_params = filter(lambda p: id(p) not in arch_params_id, model.parameters())
#
#         self.weight_optimizer = torch.optim.SGD(
#             weight_params,
#             config.optim.weight.init_lr,
#             momentum=config.optim.weight.momentum,
#             weight_decay=config.optim.weight.weight_decay)
#
#         self.arch_optimizer = torch.optim.Adam(
#             [{'params': model.module.arch_alpha_params, 'lr': config.optim.arch.alpha_lr},
#              {'params': model.module.arch_beta_params, 'lr': config.optim.arch.beta_lr}],
#             betas=(0.5, 0.999),
#             weight_decay=config.optim.arch.weight_decay)
#
#     def arch_step(self, input_valid, target_valid, model, search_stage, arch_loss, arch_metric):
#         head_sampled_w_old, alpha_head_index = \
#             model.module.sample_branch('head', 2, search_stage=search_stage)
#         stack_sampled_w_old, alpha_stack_index = \
#             model.module.sample_branch('stack', 2, search_stage=search_stage)
#         self.arch_optimizer.zero_grad()
#
#         dropped_model = nn.DataParallel(self.Dropped_Network(model))
#         logits, sub_obj = dropped_model(input_valid)
#         sub_obj = torch.mean(sub_obj)
#         loss = self.criterion(logits, target_valid)
#         if self.config.optim.if_sub_obj:
#             loss_sub_obj = torch.log(sub_obj) / torch.log(torch.tensor(self.config.optim.sub_obj.log_base))
#             sub_loss_factor = self.config.optim.sub_obj.sub_loss_factor
#             loss += loss_sub_obj * sub_loss_factor
#         # arch_loss.update(loss.item())
#         # arch_metric.update(target_valid, logits)
#         loss.backward()
#         self.arch_optimizer.step()
#         # pixAcc_train, mIoU_train, Dice_train = self.arch_metric_train.get()
#
#         self.rescale_arch_params(head_sampled_w_old,
#                                  stack_sampled_w_old,
#                                  alpha_head_index,
#                                  alpha_stack_index,
#                                  model)
#         return logits.detach(), loss.item(), sub_obj.item()  # , self.arch_train_loss_meter.mloss, self.arch_metric_train  # , (self.arch_train_loss_meter.mloss, pixAcc_train, mIoU_train, Dice_train)
#
#     def weight_step(self, *args, **kwargs):
#         return self.weight_step_(*args, **kwargs)
#
#     def weight_step_(self, input_train, target_train, model, search_stage, weight_loss, weight_metric):
#         _, _ = model.module.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
#         _, _ = model.module.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)
#
#         self.weight_optimizer.zero_grad()
#         dropped_model = nn.DataParallel(self.Dropped_Network(model))
#         logits, sub_obj = dropped_model(input_train)
#         sub_obj = torch.mean(sub_obj)
#         loss = self.criterion(logits, target_train)
#         # weight_loss.update(loss.item())
#         # weight_metric.update(target_train, logits)
#         loss.backward()
#         self.weight_optimizer.step()
#         # pixAcc_train, mIoU_train, Dice_train = self.weight_metric_train.get()
#
#         return logits.detach(), loss.item(), sub_obj.item()  # , self.weight_train_loss_meter.mloss, self.weight_metric_train  # , (self.weight_train_loss_meter.mloss, pixAcc_train, mIoU_train, Dice_train)
#
#     def valid_step(self, input_valid, target_valid, model, val_loss, val_metric):
#         _, _ = model.module.sample_branch('head', 1, training=False)
#         _, _ = model.module.sample_branch('stack', 1, training=False)
#
#         # self.model.eval()
#         dropped_model = nn.DataParallel(self.Dropped_Network(model))
#         logits, sub_obj = dropped_model(input_valid)
#         sub_obj = torch.mean(sub_obj)
#         loss = self.criterion(logits, target_valid)
#         val_loss.update(loss.item())
#         val_metric.update(target_valid, logits)
#
#         return logits, loss.item(), sub_obj.item()
#
#     def rescale_arch_params(self, alpha_head_weights_drop,
#                             alpha_stack_weights_drop,
#                             alpha_head_index,
#                             alpha_stack_index,
#                             model):
#
#         def comp_rescale_value(old_weights, new_weights, index):
#             old_exp_sum = old_weights.exp().sum()
#             new_drop_arch_params = torch.gather(new_weights, dim=-1, index=index)
#             new_exp_sum = new_drop_arch_params.exp().sum()
#             rescale_value = torch.log(old_exp_sum / new_exp_sum)
#             rescale_mat = torch.zeros_like(new_weights).scatter_(0, index, rescale_value)
#             return rescale_value, rescale_mat
#
#         def rescale_params(old_weights, new_weights, indices):
#             for i, (old_weights_block, indices_block) in enumerate(zip(old_weights, indices)):
#                 for j, (old_weights_branch, indices_branch) in enumerate(
#                         zip(old_weights_block, indices_block)):
#                     rescale_value, rescale_mat = comp_rescale_value(old_weights_branch,
#                                                                         new_weights[i][j],
#                                                                         indices_branch)
#                     new_weights[i][j].data.add_(rescale_mat)
#
#         # rescale the arch params for head layers
#         rescale_params(alpha_head_weights_drop, model.module.alpha_head_weights, alpha_head_index)
#         # rescale the arch params for stack layers
#         rescale_params(alpha_stack_weights_drop, model.module.alpha_stack_weights, alpha_stack_index)
#
#     def set_param_grad_state(self, stage):
#         def set_grad_state(params, state):
#             for group in params:
#                 for param in group['params']:
#                     param.requires_grad_(state)
#
#         if stage == 'Arch':
#             state_list = [True, False]  # [arch, weight]
#         elif stage == 'Weights':
#             state_list = [False, True]
#         else:
#             state_list = [False, False]
#         set_grad_state(self.arch_optimizer.param_groups, state_list[0])
#         set_grad_state(self.weight_optimizer.param_groups, state_list[1])
# ###promise12 experiment

###sliver07 experiment

###sliver07 experiment
class Optimizer(object):
    def KL_loss(self, outputs, targets):
        temperature = 1
        log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
        log_softmax_targets = F.log_softmax(targets / temperature, dim=1)
        softmax_targets = F.softmax(targets / temperature, dim=1)
        softmax_outputs = F.softmax(outputs / temperature, dim=1)
        # pdb.set_trace()
        return ((log_softmax_targets - log_softmax_outputs) * softmax_targets).sum(dim=1).mean()

    def CrossEntropy(self, outputs, targets):
        temperature = 1
        log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
        log_softmax_targets = F.log_softmax(targets / temperature, dim=1)
        softmax_targets = F.softmax(targets / temperature, dim=1)
        softmax_outputs = F.softmax(outputs / temperature, dim=1)
        # pdb.set_trace()
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

    def InfoEnergy(self, outputs):
        temperature = 1
        log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
        softmax_outputs = F.softmax(outputs / temperature, dim=1)
        return -(log_softmax_outputs * softmax_outputs).sum(dim=1).mean()

    def __init__(self, model, criterion, config):
        self.config = config
        self.weight_sample_num = self.config.search_params.weight_sample_num
        self.criterion = criterion
        self.Dropped_Network = lambda model: Dropped_Network(
            model, softmax_temp=config.search_params.softmax_temp)
        # self.weight_metric_train = SegmentationMetric(2)
        # self.weight_train_loss_meter = utils.AverageMeter()
        # self.arch_train_loss_meter = utils.AverageMeter()
        # self.arch_metric_train = SegmentationMetric(2)
        arch_params_id = list(map(id, model.module.arch_parameters))
        weight_params = filter(lambda p: id(p) not in arch_params_id, model.parameters())

        self.weight_optimizer = torch.optim.SGD(
            weight_params,
            config.optim.weight.init_lr,
            momentum=config.optim.weight.momentum,
            weight_decay=config.optim.weight.weight_decay, nesterov=True)

        self.arch_optimizer = torch.optim.Adam(
            [{'params': model.module.arch_alpha_params, 'lr': config.optim.arch.alpha_lr},
             {'params': model.module.arch_beta_params, 'lr': config.optim.arch.beta_lr}],
            betas=(0.5, 0.999),
            weight_decay=config.optim.arch.weight_decay)

    def arch_step(self, input_valid, target_valid, model, search_stage, arch_loss, arch_metric):
        head_sampled_w_old, alpha_head_index = \
            model.module.sample_branch('head', 2, search_stage=search_stage)
        stack_sampled_w_old, alpha_stack_index = \
            model.module.sample_branch('stack', 2, search_stage=search_stage)
        self.arch_optimizer.zero_grad()

        dropped_model = nn.DataParallel(self.Dropped_Network(model)).cuda()
        # logits_main, logits_vice, sub_obj, deeps_up2, deeps_entropy_sup = dropped_model(input_valid)
        # logits_main, sub_obj = dropped_model(input_valid) #1.13
        logits_main, sub_obj = dropped_model(input_valid)
        logits_main = softmax_helper(logits_main)
        # logits_main, sub_obj = dropped_model(input_valid)
        sub_obj = torch.mean(sub_obj)
        # loss1 = self.criterion(logits_main, target_valid)#1.13
        loss1 = self.criterion((logits_main,),  target_valid)
        # loss = torch.stack([self.criterion(logits_main, target_valid)] + [self.criterion(deep, target_valid) for deep in
        #                                                                   deep_dice_2final])
        # loss = torch.mean(loss)
        # loss2 = self.CrossEntropy(deeps_up2[0], deeps_entropy_sup[0])
        # loss3 = self.CrossEntropy(deeps_up2[1], deeps_entropy_sup[1])
        # loss4 = self.CrossEntropy(deeps_up2[2], deeps_entropy_sup[2])
        # loss2_ie = self.InfoEnergy(logits_main)
        # loss3_ie = self.InfoEnergy(deeps_up2[1])
        # loss4_ie = self.InfoEnergy(deeps_up2[2])
        # loss_ = 0.5*loss2+0.5*loss3+0.5*loss4-0.5*loss2_ie-0.5*loss3_ie-0.5*loss4_ie
        # loss=loss1+loss_

        ###cancel in no_stack###
        # if self.config.optim.if_sub_obj:
        #     loss_sub_obj = torch.log(sub_obj) / torch.log(torch.tensor(self.config.optim.sub_obj.log_base))
        #     sub_loss_factor = self.config.optim.sub_obj.sub_loss_factor
        #     loss1 += loss_sub_obj * sub_loss_factor
        ###cancel in no_stack###

        # arch_loss.update(loss.item())
        # arch_metric.update(target_valid, logits)
        loss1.backward()
        self.arch_optimizer.step()
        # pixAcc_train, mIoU_train, Dice_train = self.arch_metric_train.get()

        self.rescale_arch_params(head_sampled_w_old,
                                 stack_sampled_w_old,
                                 alpha_head_index,
                                 alpha_stack_index,
                                 model)
        return logits_main.detach(), loss1.item(), sub_obj.item()  # , self.arch_train_loss_meter.mloss, self.arch_metric_train  # , (self.arch_train_loss_meter.mloss, pixAcc_train, mIoU_train, Dice_train)

    def weight_step(self, *args, **kwargs):
        return self.weight_step_(*args, **kwargs)

    def weight_step_(self, input_train, target_train, model, search_stage, weight_loss, weight_metric):
        _, _ = model.module.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
        _, _ = model.module.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)

        self.weight_optimizer.zero_grad()
        dropped_model = nn.DataParallel(self.Dropped_Network(model)).cuda()
        # logits_main, sub_obj = dropped_model(input_train)
        # logits_main, sub_obj = dropped_model(input_train)1.13
        logits_main, sub_obj = dropped_model(input_train)
        logits_main = softmax_helper(logits_main)
        # logits_main, logits_vice, sub_obj, deeps_up2, deeps_entropy_sup = dropped_model(input_train)
        sub_obj = torch.mean(sub_obj)
        loss1 = self.criterion((logits_main,), target_train)
        # loss = torch.stack([self.criterion(logits_main, target_train)]+[self.criterion(deep, target_train) for deep in deep_dice_2final])
        # loss = torch.mean(loss)
        # loss2 = self.CrossEntropy(deeps_up2[0], deeps_entropy_sup[0])
        # loss3 = self.CrossEntropy(deeps_up2[1], deeps_entropy_sup[1])
        # loss4 = self.CrossEntropy(deeps_up2[2], deeps_entropy_sup[2])
        # loss2_ie = self.InfoEnergy(logits_main)
        # loss3_ie = self.InfoEnergy(deeps_up2[1])
        # loss4_ie = self.InfoEnergy(deeps_up2[2])
        # loss_ = 0.5 * loss2 + 0.5 * loss3 + 0.5 * loss4 - 0.5 * loss2_ie - 0.5 * loss3_ie - 0.5 * loss4_ie
        # loss = loss1 + loss_
        # weight_loss.update(loss.item())
        # weight_metric.update(target_train, logits)
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(dropped_model.parameters(), 12)
        self.weight_optimizer.step()
        # pixAcc_train, mIoU_train, Dice_train = self.weight_metric_train.get()

        return logits_main.detach(), loss1.item(), sub_obj.item()  # , self.weight_train_loss_meter.mloss, self.weight_metric_train  # , (self.weight_train_loss_meter.mloss, pixAcc_train, mIoU_train, Dice_train)

    def valid_step(self, input_valid, target_valid, model, val_loss, val_metric):
        _, _ = model.module.sample_branch('head', 1, training=False)
        _, _ = model.module.sample_branch('stack', 1, training=False)

        # self.model.eval()
        dropped_model = nn.DataParallel(self.Dropped_Network(model)).cuda()
        # logits_main, logits_vice, sub_obj, deeps_up2, deeps_entropy_sup = dropped_model(input_valid)
        # logits_main, sub_obj = dropped_model(input_valid)
        logits_main, sub_obj = dropped_model(input_valid)
        logits_main = softmax_helper(logits_main)
        sub_obj = torch.mean(sub_obj)
        loss1 = self.criterion((logits_main,), target_valid)
        # loss = torch.stack([self.criterion(logits_main, target_valid)] + [self.criterion(deep, target_valid) for deep in
        #                                                                   deep_dice_2final])
        # loss = torch.mean(loss)
        # loss2 = self.CrossEntropy(deeps_up2[0], deeps_entropy_sup[0])
        # loss3 = self.CrossEntropy(deeps_up2[1], deeps_entropy_sup[1])
        # loss4 = self.CrossEntropy(deeps_up2[2], deeps_entropy_sup[2])
        # loss2_ie = self.InfoEnergy(logits_main)
        # loss3_ie = self.InfoEnergy(deeps_up2[1])
        # loss4_ie = self.InfoEnergy(deeps_up2[2])
        # loss_ = 0.5 * loss2 + 0.5 * loss3 + 0.5 * loss4 - 0.5 * loss2_ie - 0.5 * loss3_ie - 0.5 * loss4_ie
        # loss = loss1 + loss_
        val_loss.update(loss1.item())
        val_metric.update(target_valid, logits_main)

        return logits_main, loss1.item(), sub_obj.item()

    def rescale_arch_params(self, alpha_head_weights_drop,
                            alpha_stack_weights_drop,
                            alpha_head_index,
                            alpha_stack_index,
                            model):

        def comp_rescale_value(old_weights, new_weights, index):
            old_exp_sum = old_weights.exp().sum()
            new_drop_arch_params = torch.gather(new_weights, dim=-1, index=index)
            new_exp_sum = new_drop_arch_params.exp().sum()
            rescale_value = torch.log(old_exp_sum / new_exp_sum)
            rescale_mat = torch.zeros_like(new_weights).scatter_(0, index, rescale_value)
            return rescale_value, rescale_mat

        def rescale_params(old_weights, new_weights, indices):
            for i, (old_weights_block, indices_block) in enumerate(zip(old_weights, indices)):
                for j, (old_weights_branch, indices_branch) in enumerate(
                        zip(old_weights_block, indices_block)):
                    rescale_value, rescale_mat = comp_rescale_value(old_weights_branch,
                                                                        new_weights[i][j],
                                                                        indices_branch)
                    new_weights[i][j].data.add_(rescale_mat)

        # rescale the arch params for head layers
        rescale_params(alpha_head_weights_drop, model.module.alpha_head_weights, alpha_head_index)
        # rescale the arch params for stack layers
        rescale_params(alpha_stack_weights_drop, model.module.alpha_stack_weights, alpha_stack_index)

    def set_param_grad_state(self, stage):
        def set_grad_state(params, state):
            for group in params:
                for param in group['params']:
                    param.requires_grad_(state)

        if stage == 'Arch':
            state_list = [True, False]  # [arch, weight]
        elif stage == 'Weights':
            state_list = [False, True]
        else:
            state_list = [False, False]
        set_grad_state(self.arch_optimizer.param_groups, state_list[0])
        set_grad_state(self.weight_optimizer.param_groups, state_list[1])
###sliver07 experiment