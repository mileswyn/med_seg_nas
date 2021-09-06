import argparse
import ast
import importlib
import logging
import os
import pprint
import sys
import time
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

from configs.search_config_seg import search_cfg
from configs.promise12_train_cfg import cfg
# from dataset import imagenet_data
from dataset.Promise12_dataset import Promise12
from dataset.sliver07_dataset import Silver07
from models import model_derived_segment
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg
from tools.lr_scheduler import get_lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from tools.multadds_count import comp_multadds
from tools.loss_seg import SegmentationLosses, EDiceLoss

sys.path.append('.')
from run_apis.optimizer_segment import Optimizer
from run_apis.trainer_seg import SearchTrainer

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2,3'
    print('torch_version:', torch.__version__)
    parser = argparse.ArgumentParser("Search_Configs")
    parser.add_argument('--report_freq', type=float, default=3, help='report frequency')
    # parser.add_argument('--data_path', type=str, default='/hdd1/wyn/PROMISE2012/', help='location of the data corpus')
    parser.add_argument('--data_path', type=str, default='/hdd1/wyn/silver07/', help='location of the data corpus')
    # parser.add_argument('--save', type=str, default='/hdd1/wyn/DenseNAS/log/', help='experiment name')
    parser.add_argument('--save', type=str, default='/hdd1/wyn/DenseNAS/log/', help='experiment name')
    parser.add_argument('--tb_path', type=str, default='', help='tensorboard output path')
    parser.add_argument('--job_name', type=str, default='search_res_sliver07', help='job_name')
    # parser.add_argument('-c', '--config', metavar='C',
    #                     default='/hdd1/wyn/DenseNAS/configs/promise12_search_cfg_mbv2.yaml',
    #                     help='The Configuration file')
    parser.add_argument('-c', '--config', metavar='C',
                        default='/hdd1/wyn/DenseNAS/configs/sliver07_search_cfg_resnet.yaml',
                        help='The Configuration file')

    args = parser.parse_args()

    update_cfg_from_cfg(search_cfg, cfg)
    if args.config is not None:
        merge_cfg_from_file(args.config, cfg)
    config = cfg

    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S-") + args.job_name
        args.save = os.path.join(args.save, args.job_name)
        utils.create_exp_dir(args.save)
        # os.system('cp -r ./* '+args.save)
        args.save = os.path.join(args.save, 'output')
        utils.create_exp_dir(args.save)
    else:
        args.save = os.path.join(args.save, 'output')
        utils.create_exp_dir(args.save)

    if args.tb_path == '':
        args.tb_path = args.save

    log_format = '%(asctime)s %(message)s'
    date_format = '%m/%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt=date_format)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format, date_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('No gpu device available')
        sys.exit(1)
    cudnn.benchmark = False
    cudnn.enabled = False

    if config.train_params.use_seed:
        np.random.seed(config.train_params.seed)
        torch.manual_seed(config.train_params.seed)
        torch.cuda.manual_seed(config.train_params.seed)

    logging.info("args = %s", args)
    # logging.info('Training with config:')
    # logging.info(pprint.pformat(config))
    # job_name = os.popen('cd %s && pwd -P && cd -' % args.save).readline().split('/')[-2]
    job_name = 'DenseNAS'
    writer = SummaryWriter(args.save)

    # criterion = SegmentationLosses(name='cross_entropy_with_dice')
    # criterion = criterion.cuda()
    criterion = EDiceLoss().cuda()

    SearchSpace = importlib.import_module('models.promise_search_space_' + config.net_type).Network
    ArchGenerater = importlib.import_module('run_apis.seg_derive_arch_' + config.net_type, __package__).ArchGenerate
    derivedNetwork = getattr(model_derived_segment, '%s_Net' % config.net_type.upper())

    super_model = SearchSpace(config.optim.init_dim, config.data.dataset, config)
    arch_gener = ArchGenerater(super_model, config)
    der_Net = lambda net_config: derivedNetwork(net_config,
                                                config=config)
    super_model = nn.DataParallel(super_model)

    # whether to resume from a checkpoint
    if config.optim.if_resume:
        utils.load_model(super_model, config.optim.resume.load_path)
        start_epoch = config.optim.resume.load_epoch + 1
    else:
        start_epoch = 0

    super_model = super_model.cuda()

    if config.optim.sub_obj.type == 'flops':
        flops_list, total_flops = super_model.module.get_cost_list(
            config.data.input_size, cost_type='flops')
        super_model.module.sub_obj_list = flops_list
        logging.info("Super Network flops (M) list: \n")
        logging.info(str(flops_list))
        logging.info("Total flops: " + str(total_flops))
    elif config.optim.sub_obj.type == 'latency':
        with open(os.path.join('/hdd1/wyn/DenseNAS/latency_list/', config.optim.sub_obj.latency_list_path), 'r') as f:
            latency_list = eval(f.readline())
        super_model.module.sub_obj_list = latency_list
        logging.info("Super Network latency (ms) list: \n")
        logging.info(str(latency_list))
    else:
        raise NotImplementedError
    logging.info("Num Params = %.2fMB", utils.count_parameters_in_MB(super_model))

    #
    # if config.data.dataset == 'imagenet':
    #     imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'),
    #                                         testFolder=os.path.join(args.data_path, 'val'),
    #                                         num_workers=config.data.num_workers,
    #                                         type_of_data_augmentation=config.data.type_of_data_aug,
    #                                         data_config=config.data)
    #     train_queue, valid_queue = imagenet.getTrainTestLoader(config.data.batch_size,
    #                                                            train_shuffle=True,
    #                                                            val_shuffle=True)
    # else:
    #     raise NotImplementedError
    train_set = Silver07(args.data_path, mode='train')
    valid_set = Silver07(args.data_path, mode='val')
    print('config.data.batch_size:', config.data.batch_size)
    train_queue = torch.utils.data.DataLoader(train_set, batch_size=config.data.batch_size,
                                              shuffle=True,
                                              num_workers=config.data.num_workers, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(valid_set, batch_size=config.data.batch_size,
                                              shuffle=True,
                                              num_workers=config.data.num_workers, pin_memory=True)

    search_optim = Optimizer(super_model, criterion, config)

    # scheduler = get_lr_scheduler(config, search_optim.weight_optimizer, train_queue.dataset.__len__())
    # scheduler.last_step = start_epoch * (train_queue.dataset.__len__() // config.data.batch_size + 1)
    scheduler = ExponentialLR(search_optim.weight_optimizer, gamma=0.9996)
    search_trainer = SearchTrainer(train_queue, valid_queue, search_optim, criterion, scheduler, config, args, writer)

    betas, head_alphas, stack_alphas = super_model.module.display_arch_params()
    derived_archs = arch_gener.derive_archs(betas, head_alphas, stack_alphas, if_display=True)
    derived_model = der_Net('|'.join(map(str, derived_archs)))
    logging.info("Derived Model Mult-Adds = %.2fMB" % comp_multadds(derived_model,
                                                                    input_size=config.data.input_size))
    logging.info("Derived Model Num Params = %.2fMB", utils.count_parameters_in_MB(derived_model))

    rec_list = []

    start = time.time()
    patience, max_patience = 0, 40
    dur_time = 0
    save_best = True
    best_epoch, best_mIoU, best_loss, best_pixAcc, best_Dice = 0, 0, 1.0, 0, 0

    for epoch in range(start_epoch, config.train_params.epochs):
        search_trainer.metric_train.reset()
        search_trainer.metric_val.reset()
        # training part1: update the architecture parameters
        if epoch >= config.search_params.arch_update_epoch:  # epoch>=50
            search_stage = 1
            search_optim.set_param_grad_state('Arch')
            train_loss, train_loss2, train_pixAcc, train_mIoU, train_Dice, sub_obj, visual_predicts_train, visual_target_train = search_trainer.train(
                super_model, epoch, 'Arch', search_stage)
            logging.info('EPOCH%d Arch Train_metric  pixAcc %.2f mIoU %.5f dice %.5f loss %.4f %s %.2f',
                         epoch, train_pixAcc, train_mIoU, train_Dice, train_loss2, config.optim.sub_obj.type, sub_obj)
            writer.add_scalar('arch_train_Dice', train_Dice, epoch)
            writer.add_scalar('arch_train_loss', train_loss2, epoch)
        else:
            search_stage = 0

        # training part2: update the operator parameters
        search_optim.set_param_grad_state('Weights')
        lattest_lr = scheduler.get_lr()
        print('latest lr: ', lattest_lr)
        train_loss, train_loss2, train_pixAcc, train_mIoU, train_Dice, sub_obj_train, visual_predicts_train, visual_target_train = search_trainer.train(
            super_model, epoch, 'Weights', search_stage)
        logging.info('EPOCH%d Weights Train_metric  pixAcc %.2f mIoU %.5f dice %.5f loss %.4f sub_obj_train "%s %.2f"',
                     epoch, train_pixAcc, train_mIoU, train_Dice, train_loss2, config.optim.sub_obj.type, sub_obj_train)
        print('weight latest lr: ', lattest_lr)
        writer.add_scalar('weight_train_Dice', train_Dice, epoch)
        writer.add_scalar('weight_train_loss', train_loss2, epoch)
        writer.add_image('predict_image/train', visual_predicts_train, epoch)
        visual_target_train = torch.squeeze(visual_target_train, dim=1)
        writer.add_image('target_image/train', visual_target_train, epoch)

        # validation
        if epoch >= config.search_params.val_start_epoch:
            with torch.no_grad():
                val_loss, val_loss2, val_pixAcc, val_mIoU, val_Dice, sub_obj_val, visual_predicts_valid, visual_target_valid = search_trainer.infer(super_model, epoch)
            logging.info('EPOCH%d Valid_metric  val_pixAcc %.2f val_mIoU %.5f val_dice %.5f loss %.4f sub_obj_val "%s %.2f"',
                         epoch, val_pixAcc, val_mIoU, val_Dice, train_loss2, config.optim.sub_obj.type, sub_obj_val)
            writer.add_scalar('arch_val_Dice', val_Dice, epoch)
            writer.add_scalar('arch_whole_{}'.format(config.optim.sub_obj.type), sub_obj_val, epoch)
            writer.add_image('predict_image/valid', visual_predicts_valid, epoch)
            writer.add_image('target_image/valid', torch.squeeze(visual_target_valid, dim=1), epoch)

            if val_Dice > best_Dice and val_loss2 < best_loss:
                best_mIoU, best_loss, best_pixAcc, best_Dice = val_mIoU, val_loss2, val_pixAcc, val_Dice
                utils.save(super_model, os.path.join(args.save, 'weights_best.pt'))
            logging.info('BEST EPOCH %d  val_loss %.2f val_dice %.2f', best_epoch, best_loss, best_Dice)
        else:
            utils.save(super_model, os.path.join(args.save, 'weights_best.pt'))

        betas, head_alphas, stack_alphas = super_model.module.display_arch_params()
        derived_arch = arch_gener.derive_archs(betas, head_alphas, stack_alphas, if_display=True)
        derived_arch_str = '|\n'.join(map(str, derived_arch))
        derived_model = der_Net(derived_arch_str)
        derived_flops = comp_multadds(derived_model, input_size=config.data.input_size)
        derived_params = utils.count_parameters_in_MB(derived_model)
        logging.info("Derived Model Mult-Adds = %.2fMB" % derived_flops)
        logging.info("Derived Model Num Params = %.2fMB" % derived_params)
        writer.add_scalar('derived_flops', derived_flops, epoch)

        if (epoch + 1) == config.search_params.arch_update_epoch:
            utils.save(super_model, os.path.join(args.save, 'arch_update_epoch_weights_{}.pt'.format(epoch)))

        if epoch >= config.search_params.val_start_epoch:
            epoch_rec = {'dice': val_Dice,
                         'epoch': epoch,
                         'multadds': derived_flops,
                         'params': derived_params,
                         'arch': derived_arch_str}
            if_update = utils.record_topk(2, rec_list, epoch_rec, 'dice', 'arch')
            if if_update:
                with open(os.path.join(args.save, 'top_results'), 'w') as f:
                    f.write(str(rec_list) + '\n')
                    f.write(job_name)
                with open(os.path.join(args.save, 'excel_record'), 'w') as f:
                    for record in rec_list:
                        f.write(',,,{:.2f}MB,{:.2f}MB,,,,{},{}\n'.format(
                            record['multadds'], record['params'],
                            job_name, record['epoch']))
                        f.write(record['arch'] + '\n')

    logging.info('\nTop2 arch records for Excel: ')
    for record in rec_list:
        logging.info('\n,,,{:.2f}MB,{:.2f}MB,,,,{},{}'.format(
            record['multadds'], record['params'], job_name, record['epoch']))
        logging.info('\n' + record['arch'])
