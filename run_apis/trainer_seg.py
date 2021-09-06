import os
from os import listdir
from os.path import isfile, join, splitext
import logging
import time
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
from tqdm.std import trange

from dataset.prefetch_data import data_prefetcher
from tools import utils
from tools.metric_seg import SegmentationMetric
import pickle
from collections import OrderedDict
from sklearn.model_selection import KFold
from batchgenerators.dataloading import SlimDataLoaderBase
from dataset.abdominal_dataset import dataloader3D_abdominal
from data_augment.default_data_augmentation import get_moreDA_augmentation


data_aug_params = {'selected_data_channels': None, 'selected_seg_channels': [0], 
                   'do_elastic': False, 'elastic_deform_alpha': (0.0, 900.0), 
                   'elastic_deform_sigma': (9.0, 13.0), 'p_eldef': 0.2, 
                   'do_scaling': True, 'scale_range': (0.7, 1.4), 
                   'independent_scale_factor_for_each_axis': False, 
                   'p_independent_scale_per_axis': 1, 'p_scale': 0.2, 
                   'do_rotation': True, 'rotation_x': (-0.5235987755982988, 0.5235987755982988), 
                   'rotation_y': (-0.5235987755982988, 0.5235987755982988), 
                   'rotation_z': (-0.5235987755982988, 0.5235987755982988), 
                   'rotation_p_per_axis': 1, 'p_rot': 0.2, 'random_crop': False, 
                   'random_crop_dist_to_border': None, 'do_gamma': True, 
                   'gamma_retain_stats': True, 'gamma_range': (0.7, 1.5), 
                   'p_gamma': 0.3, 'do_mirror': True, 'mirror_axes': (0, 1, 2), 
                   'dummy_2D': False, 'mask_was_used_for_normalization': OrderedDict([(0, False)]), 
                   'border_mode_data': 'constant', 'all_segmentation_labels': None, 
                   'move_last_seg_chanel_to_data': False, 'cascade_do_cascade_augmentations': False, 
                   'cascade_random_binary_transform_p': 0.4, 
                   'cascade_random_binary_transform_p_per_label': 1, 
                   'cascade_random_binary_transform_size': (1, 8), 'cascade_remove_conn_comp_p': 0.2, 
                   'cascade_remove_conn_comp_max_size_percent_threshold': 0.15, 
                   'cascade_remove_conn_comp_fill_with_other_class_p': 0.0, 
                   'do_additive_brightness': False, 'additive_brightness_p_per_sample': 0.15, 
                   'additive_brightness_p_per_channel': 0.5, 'additive_brightness_mu': 0.0, 
                   'additive_brightness_sigma': 0.1, 'num_threads': 12, 'num_cached_per_thread': 2, 
                   'patch_size_for_spatialtransform': np.array([128, 128, 128])}

def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=True)
    return data

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers

def load_dataset(folder, num_cases_properties_loading_threshold=1000):
    # we don't load the actual data but instead return the filename to the np file.
    print('loading dataset')
    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset
# ###promise12 experiment
# class Trainer(object):
#     def __init__(self, model, train_data, val_data, optimizer=None, criterion=None,
#                  scheduler=None, config=None, report_freq=None):
#         self.model = model
#         self.train_data = train_data
#         self.val_data = val_data
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.scheduler = scheduler
#         self.config = config
#         self.report_freq = report_freq
#         self.patience = 0
#         self.save_best = True
#         self.metric_train = SegmentationMetric(2)
#         self.metric_val = SegmentationMetric(2)
#         self.train_loss_meter = utils.AverageMeter()
#         self.val_loss_meter = utils.AverageMeter()
#
#     def train(self, epoch):
#         # if there's some problems of 'self.model'
#         self.metric_train.reset()
#         self.model.train()
#         tbar = tqdm(self.train_data)
#         self.scheduler.step()
#         for step, (input, target) in enumerate(tbar):
#             # if step == 0:
#             #     logging.info('epoch %d lr %e', epoch, self.optimizer.param_groups[0]['lr'])
#             self.optimizer.zero_grad()
#             input = input.cuda()
#             target = target.cuda()
#             input = torch.unsqueeze(input, dim=1)
#             predicts = self.model(input)
#             train_loss = self.criterion(predicts, target)
#             self.train_loss_meter.update(train_loss.item())
#             self.metric_train.update(target, predicts)
#             train_loss.backward()
#             if self.config.optim.use_grad_clip:
#                 nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
#             self.optimizer.step()
#         pixAcc_train, mIoU_train, Dice_train = self.metric_train.get()
#         visual_predicts_train = torch.unsqueeze(torch.max(predicts, 1)[1][1], dim=0)
#         visual_target_train = torch.unsqueeze(target[1], dim=0)
#
#         return self.train_loss_meter.mloss, pixAcc_train, mIoU_train, Dice_train, visual_predicts_train, visual_target_train
#
#
#     def infer(self, epoch=0):
#         self.metric_val.reset()
#         self.model.eval()
#         tbar = tqdm(self.val_data)
#         with torch.no_grad():
#             for step, (input, target) in enumerate(tbar):
#                 input = input.cuda()
#                 target = target.cuda()
#                 input = torch.unsqueeze(input, dim=1)
#                 # target = torch.unsqueeze(target, dim=1)
#                 predicts = self.model(input)
#                 val_loss = self.criterion(predicts, target)
#                 self.val_loss_meter.update(val_loss.item())
#                 self.metric_val.update(target, predicts)
#         pixAcc_val, mIoU_val, Dice_val = self.metric_val.get()
#         cur_loss = self.val_loss_meter.mloss
#         visual_predicts_valid = torch.unsqueeze(torch.max(predicts, 1)[1][1], dim=0)
#         visual_target_valid = torch.unsqueeze(target[1], dim=0)
#
#         return self.model, cur_loss, pixAcc_val, mIoU_val, Dice_val, visual_predicts_valid, visual_target_valid
#
#
# class SearchTrainer(object):
#     def __init__(self, train_data, val_data, search_optim, criterion, scheduler, config, args, writer):
#         self.train_data = train_data
#         self.val_data = val_data
#         self.search_optim = search_optim
#         self.criterion = criterion
#         self.scheduler = scheduler
#         self.num_epochs = config.train_params.epochs
#         self.sub_obj_type = config.optim.sub_obj.type
#         self.args = args
#         self.patience = 0
#         self.save_best = True
#         self.train_loss_meter = utils.AverageMeter()
#         self.metric_train = SegmentationMetric(2)
#         # self.weight_metric_train = SegmentationMetric(2)
#         # self.arch_metric_train = SegmentationMetric(2)
#         # self.weight_train_loss_meter = utils.AverageMeter()
#         # self.arch_train_loss_meter = utils.AverageMeter()
#         self.val_loss_meter = utils.AverageMeter()
#         self.metric_val = SegmentationMetric(2)
#         self.writer = writer
#
#     def train(self, model, epoch, optim_obj='Weights', search_stage=0):
#         assert optim_obj in ['Weights', 'Arch']
#         objs = utils.AverageMeter()
#         # top1 = utils.AverageMeter()
#         # top5 = utils.AverageMeter()
#         sub_obj_avg = utils.AverageMeter()
#         # data_time = utils.AverageMeter()
#         # batch_time = utils.AverageMeter()
#         self.metric_train.reset()
#         model.train()
#         tbar = tqdm(self.train_data)
#         # start = time.time()
#         if optim_obj == 'Weights':
#             tbar = tqdm(self.train_data)
#         elif optim_obj == 'Arch':
#             tbar = tqdm(self.val_data)
#         step = 0
#         self.scheduler.step()
#         for stepnum, (input, target) in enumerate(tbar):
#             input, target = input.cuda(), target.cuda()
#             input = torch.unsqueeze(input, dim=1)
#             n = input.size(0)
#             if optim_obj == 'Weights':
#
#                 if step == 0:
#                     logging.info('epoch %d weight_lr %e', epoch,
#                                  self.search_optim.weight_optimizer.param_groups[0]['lr'])
#                 logits, loss, sub_obj = self.search_optim.weight_step(input, target, model, search_stage,
#                                                                     self.train_loss_meter, self.metric_train)
#
#             elif optim_obj == 'Arch':
#                 if step == 0:
#                     logging.info('epoch %d arch_lr %e', epoch, self.search_optim.arch_optimizer.param_groups[0]['lr'])
#                 logits, loss, sub_obj = self.search_optim.arch_step(input, target, model, search_stage,
#                                                                     self.train_loss_meter, self.metric_train)
#
#             self.train_loss_meter.update(loss)
#             self.metric_train.update(target, logits)
#             del input
#
#             objs.update(loss, n)
#             sub_obj_avg.update(sub_obj)
#             train_loss = self.train_loss_meter.avg
#             pixAcc_train, mIoU_train, Dice_train = self.metric_train.get()
#             # if step != 0 and step % self.args.report_freq == 0:
#             #     if optim_obj == 'Weights':
#             #         pixAcc_train, mIoU_train, Dice_train = self.metric_train.get()
#             #         logging.info(
#             #             'Train%s epoch %03d step %03d | weight_loss %.4f sub_obj: "%s %.2f" weight_pixAcc %.3f weight_mIoU %.5f weight_Dice %.5f |',
#             #             optim_obj, epoch, step, self.train_loss_meter.avg, self.sub_obj_type, sub_obj_avg.avg,
#             #             pixAcc_train, mIoU_train, Dice_train)
#             #     elif optim_obj == 'Arch':
#             #         pixAcc_train, mIoU_train, Dice_train = self.metric_train.get()
#             #         logging.info(
#             #             'Train%s epoch %03d step %03d | arch_loss %.4f sub_obj: "%s %.2f" arch_pixAcc %.3f arch_mIoU %.5f arch_Dice %.5f |',
#             #             optim_obj, epoch, step, self.train_loss_meter.avg, self.sub_obj_type, sub_obj_avg.avg,
#             #             pixAcc_train, mIoU_train, Dice_train)
#             step += 1
#
#         visual_predicts_train = torch.unsqueeze(torch.max(logits, 1)[1][1], dim=0)
#         visual_target_train = torch.unsqueeze(target[1], dim=0)
#         return objs.avg, train_loss, pixAcc_train, mIoU_train, Dice_train, sub_obj_avg.avg, visual_predicts_train, visual_target_train   # top1.avg, top5.avg, objs.avg, sub_obj_avg.avg, batch_time.avg
#
#     def infer(self, model, epoch):
#         objs = utils.AverageMeter()
#         sub_obj_avg = utils.AverageMeter()
#         self.metric_val.reset()
#         model.train()  # don't use running_mean and running_var during search
#         tbar = tqdm(self.val_data)
#         # start = time.time()
#         step = 0
#         for stepnum, (input, target) in enumerate(tbar):
#             step += 1
#             n = input.size(0)
#             input, target = input.cuda(), target.cuda()
#             input = torch.unsqueeze(input, dim=1)
#             logits, loss, sub_obj = self.search_optim.valid_step(input, target, model, self.val_loss_meter, self.metric_val)
#             objs.update(loss, n)
#             sub_obj_avg.update(sub_obj)
#             val_loss = self.val_loss_meter.avg
#
#             if step % self.args.report_freq == 0:
#                 pixAcc_val, mIoU_val, Dice_val = self.metric_val.get()
#                 logging.info(
#                     'Val epoch %03d step %03d | loss %.4f sub_obj "%s %.2f" pixAcc %.2f mIoU %.5f Dice %.5f|',
#                     epoch, step, val_loss, self.sub_obj_type, sub_obj_avg.avg, pixAcc_val, mIoU_val, Dice_val)
#         visual_predicts_valid = torch.unsqueeze(torch.max(logits, 1)[1][1], dim=0)
#         visual_target_valid = torch.unsqueeze(target[1], dim=0)
#         return objs.avg, val_loss, pixAcc_val, mIoU_val, Dice_val, sub_obj_avg.avg, visual_predicts_valid, visual_target_valid
# ###promise12 experiment


###sliver07 experiment

#sliver07 experiment
class Trainer(object):
    def __init__(self, model, train_data, val_data, optimizer=None, criterion=None,
                 scheduler=None, config=None, report_freq=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.report_freq = report_freq
        self.patience = 0
        self.save_best = True
        self.metric_train = SegmentationMetric(2)
        self.metric_val = SegmentationMetric(2)
        self.train_loss_meter = utils.AverageMeter()
        self.val_loss_meter = utils.AverageMeter()

    def train(self, epoch):
        # if there's some problems of 'self.model'
        self.metric_train.reset()
        self.model.train()
        tbar = tqdm(self.train_data)
        self.scheduler.step()
        for step, (input, target) in enumerate(tbar):
            # if step == 0:
            #     logging.info('epoch %d lr %e', epoch, self.optimizer.param_groups[0]['lr'])
            self.optimizer.zero_grad()
            input = input.cuda()
            target = target.cuda()
            input = torch.unsqueeze(input, dim=1)
            target = torch.unsqueeze(target, dim=1)
            predicts, predicts_vice, deeps, deeps_vice = self.model(input)
            loss = torch.stack([self.criterion(predicts, predicts_vice, target)] + [self.criterion(deep, deep_vice, target) for deep, deep_vice in zip(deeps, deeps_vice)])
            #print(f"main loss: {loss}")
            train_loss = torch.mean(loss)
            self.train_loss_meter.update(train_loss.item())
            self.metric_train.update(target, predicts)
            train_loss.backward()
            if self.config.optim.use_grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
            self.optimizer.step()
        pixAcc_train, mIoU_train, Dice_train = self.metric_train.get()
        print(predicts.shape)
        visual_predicts_train = torch.unsqueeze(torch.max(predicts, 1)[1][1], dim=0)
        visual_target_train = torch.unsqueeze(target[1], dim=0)

        return self.train_loss_meter.mloss, pixAcc_train, mIoU_train, Dice_train, visual_predicts_train, visual_target_train


    def infer(self, epoch=0):
        self.metric_val.reset()
        self.model.eval()
        tbar = tqdm(self.val_data)
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.cuda()
                target = target.cuda()
                input = torch.unsqueeze(input, dim=1)
                target = torch.unsqueeze(target, dim=1)
                predicts, predicts_vice, deeps, deeps_vice = self.model(input)
                val_loss = torch.stack(
                    [self.criterion(predicts, predicts_vice, target)] + [self.criterion(deep, deep_vice, target) for deep, deep_vice in zip(deeps, deeps_vice)])
                #print(f"valid loss: {val_loss}")
                val_loss = torch.mean(val_loss)
                self.val_loss_meter.update(val_loss.item())
                self.metric_val.update(target, predicts)
        pixAcc_val, mIoU_val, Dice_val = self.metric_val.get()
        cur_loss = self.val_loss_meter.mloss
        visual_predicts_valid = torch.unsqueeze(torch.max(predicts, 1)[1][1], dim=0)
        visual_target_valid = torch.unsqueeze(target[1], dim=0)

        return self.model, cur_loss, pixAcc_val, mIoU_val, Dice_val, visual_predicts_valid, visual_target_valid


class SearchTrainer(object):
    def __init__(self, search_optim, criterion, scheduler, config, args, writer, fold=0, batch_size=2):
        self.train_data, self.val_data = None, None
        self.tr_gen, self.val_gen = None, None
        self.search_optim = search_optim
        self.criterion = criterion
        self.scheduler = scheduler
        self.num_epochs = config.train_params.epochs
        self.sub_obj_type = config.optim.sub_obj.type
        self.args = args
        self.patience = 0
        self.save_best = True
        self.train_loss_meter = utils.AverageMeter()
        self.metric_train = SegmentationMetric(9)
        # self.weight_metric_train = SegmentationMetric(2)
        # self.arch_metric_train = SegmentationMetric(2)
        # self.weight_train_loss_meter = utils.AverageMeter()
        # self.arch_train_loss_meter = utils.AverageMeter()
        self.val_loss_meter = utils.AverageMeter()
        self.metric_val = SegmentationMetric(9)
        self.writer = writer
        ###loss###

        ###data###
        self.dataset_directory = '/hdd1/wyn/nnUNetFrame/nnUNet_preprocessed/Task102_TCIAorgan'
        self.folder_with_preprocessed_data = '/hdd1/wyn/nnUNetFrame/nnUNet_preprocessed/Task102_TCIAorgan/nnUNetData_plans_v2.1_stage0'
        self.dataset = None
        self.batch_size = batch_size
        self.fold = fold
        self.patch_size = [205, 205, 205]
        self.final_patch_size = [128,128,128]
        self.data_aug_params = data_aug_params
        self.num_batches_per_epoch = 60
        self.max_num_epochs = config.train_params.epochs
        self.initialize()
    
    def initialize(self):
        self.train_data, self.val_data = self.get_basic_generators()
        self.tr_gen, self.val_gen = get_moreDA_augmentation(self.train_data, 
                                                            self.val_data, 
                                                            self.data_aug_params['patch_size_for_spatialtransform'],
                                                            self.data_aug_params,
                                                            deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]],
                                                            pin_memory=True)
        # _ = self.tr_gen.next()
        # _ = self.val_gen.next()

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            print("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys

        splits = load_pickle(splits_file)
        tr_keys = splits[self.fold]['train']
        val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_tr = dataloader3D_abdominal(self.dataset_tr, self.batch_size, self.patch_size, self.final_patch_size)
        dl_val = dataloader3D_abdominal(self.dataset_val, self.batch_size, self.final_patch_size, self.final_patch_size)
        return dl_tr, dl_val

    def train(self, model, epoch, optim_obj='Weights', search_stage=0):
        assert optim_obj in ['Weights', 'Arch']
        objs = utils.AverageMeter()
        sub_obj_avg = utils.AverageMeter()
        # data_time = utils.AverageMeter()
        # batch_time = utils.AverageMeter()
        self.epoch = epoch
        self.train_loss_meter.reset()
        self.metric_train.reset()
        model.train()
        # tbar = tqdm(self.train_data)
        # start = time.time()
        if optim_obj == 'Weights':
            tbar = tqdm(self.train_data)
        elif optim_obj == 'Arch':
            tbar = tqdm(self.val_data)
        step = 0
        
        self.scheduler.step()
        with trange(self.num_batches_per_epoch) as tbar:
            for b in tbar:
        # for stepnum, (input, target) in enumerate(tbar):
                tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))
                data_dict = next(self.tr_gen)
                data, target = data_dict['data'], data_dict['target']
                data, target = maybe_to_torch(data), maybe_to_torch(target)
                data, target = to_cuda(data), to_cuda(target)

                # data = torch.unsqueeze(data, dim=1)
                # target = torch.unsqueeze(target, dim=1)
                n = data.size(0)
                if optim_obj == 'Weights':

                    if step == 0:
                        logging.info('epoch %d weight_lr %e', epoch,
                                    self.search_optim.weight_optimizer.param_groups[0]['lr'])
                    pred_train, loss, sub_obj = self.search_optim.weight_step(data, target, model, search_stage,
                                                                        self.train_loss_meter, self.metric_train)

                elif optim_obj == 'Arch':
                    if step == 0:
                        logging.info('epoch %d arch_lr %e', epoch, self.search_optim.arch_optimizer.param_groups[0]['lr'])
                    pred_train, loss, sub_obj = self.search_optim.arch_step(data, target, model, search_stage,
                                                                        self.train_loss_meter, self.metric_train)

                self.train_loss_meter.update(loss)
                self.metric_train.update(target, pred_train)
                del data

                objs.update(loss, n)
                sub_obj_avg.update(sub_obj)
                train_loss = self.train_loss_meter.mloss
                mIoU_train, Dice_train = self.metric_train.get()
                # if step != 0 and step % self.args.report_freq == 0:
                #     if optim_obj == 'Weights':
                #         pixAcc_train, mIoU_train, Dice_train = self.metric_train.get()
                #         logging.info(
                #             'Train%s epoch %03d step %03d | weight_loss %.4f sub_obj: "%s %.2f" weight_pixAcc %.3f weight_mIoU %.5f weight_Dice %.5f |',
                #             optim_obj, epoch, step, self.train_loss_meter.avg, self.sub_obj_type, sub_obj_avg.avg,
                #             pixAcc_train, mIoU_train, Dice_train)
                #     elif optim_obj == 'Arch':
                #         pixAcc_train, mIoU_train, Dice_train = self.metric_train.get()
                #         logging.info(
                #             'Train%s epoch %03d step %03d | arch_loss %.4f sub_obj: "%s %.2f" arch_pixAcc %.3f arch_mIoU %.5f arch_Dice %.5f |',
                #             optim_obj, epoch, step, self.train_loss_meter.avg, self.sub_obj_type, sub_obj_avg.avg,
                #             pixAcc_train, mIoU_train, Dice_train)
                step += 1

        visual_predicts_train = torch.unsqueeze(torch.max(pred_train.detach(), 1)[1][1][55], dim=0)
        visual_target_train = target[0][0,:,55].detach()
        return objs.avg, train_loss, mIoU_train, Dice_train, sub_obj_avg.avg, visual_predicts_train, visual_target_train   # top1.avg, top5.avg, objs.avg, sub_obj_avg.avg, batch_time.avg

    def infer(self, model, epoch):
        objs = utils.AverageMeter()
        sub_obj_avg = utils.AverageMeter()
        self.metric_val.reset()
        model.train()  # don't use running_mean and running_var during search
        tbar = tqdm(self.val_data)
        # start = time.time()
        step = 0
        with trange(self.num_batches_per_epoch) as tbar:
            for b in tbar:
        # for stepnum, (input, target) in enumerate(tbar):
            # step += 1
            
                tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))
                data_dict = next(self.val_gen)
                data, target = data_dict['data'], data_dict['target']
                data, target = maybe_to_torch(data), maybe_to_torch(target)
                data, target = to_cuda(data), to_cuda(target)
                n = data.size(0)
                # input, target = input.cuda(), target.cuda()
                # input = torch.unsqueeze(input, dim=1)
                # target = torch.unsqueeze(target, dim=1)
                pred_val, loss, sub_obj = self.search_optim.valid_step(data, target, model, self.val_loss_meter, self.metric_val)
                objs.update(loss, n)
                sub_obj_avg.update(sub_obj)
                val_loss = self.val_loss_meter.mloss

                if step % self.args.report_freq == 0:
                    mIoU_val, Dice_val = self.metric_val.get()
                    logging.info(
                        'Val epoch %03d step %03d | loss %.4f sub_obj "%s %.2f"  mIoU %.5f Dice %.5f|',
                        epoch, step, val_loss, self.sub_obj_type, sub_obj_avg.avg, mIoU_val, Dice_val)
        visual_predicts_valid = torch.unsqueeze(torch.max(pred_val.detach(), 1)[1][1][55], dim=0)
        visual_target_valid = target[0][0,:,55].detach()
        return objs.avg, val_loss, mIoU_val, Dice_val, sub_obj_avg.avg, visual_predicts_valid, visual_target_valid
###sliver07 experiment