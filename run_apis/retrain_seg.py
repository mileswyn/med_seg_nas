import argparse
import ast
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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
import torchsummary

sys.path.append('..')
# from configs.promise12_train_cfg import cfg as config
from configs.sliver07_train_cfg import cfg as config
from dataset import imagenet_data
from models import model_derived_segment as model_derived
from tools import utils
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds
from dataset.Promise12_dataset import Promise12
from dataset.sliver07_dataset import Silver07
from tools.loss_seg import SegmentationLosses, EDiceLoss


from run_apis.trainer_seg import Trainer

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    parser = argparse.ArgumentParser("Train_Params")
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--data_path', type=str, default='/hdd1/wyn/silver07/', help='location of the data corpus')
    parser.add_argument('--load_path', type=str, default='./model_path', help='model loading path')
    parser.add_argument('--save', type=str, default='/hdd1/wyn/DenseNAS/log', help='experiment name')
    #parser.add_argument('--save_pt', type=str, default=r'E:\CodeRepo\DenseNAS\pt', help='experiment name')
    parser.add_argument('--tb_path', type=str, default='', help='tensorboard output path')
    parser.add_argument('--meas_lat', type=ast.literal_eval, default='False',
                        help='whether to measure the latency of the model')
    parser.add_argument('--job_name', type=str, default='res_sliver07Train', help='job_name')
    args = parser.parse_args()

    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S-") + args.job_name
        args.save = os.path.join(args.save, args.job_name)
        utils.create_exp_dir(args.save)
        # os.system('cp -r ./* ' + args.save)
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

    writer = SummaryWriter(args.save)

    utils.set_logging(args.save)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = False
    cudnn.enabled = True

    if config.train_params.use_seed:
        utils.set_seed(config.train_params.seed)

    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(pprint.pformat(config))

    if os.path.isfile(os.path.join(args.load_path, 'net_config')):
        config.net_config, config.net_type = utils.load_net_config(
            os.path.join(args.load_path, 'net_config'))
    derivedNetwork = getattr(model_derived, '%s_Net' % config.net_type.upper())
    derivedNetwork_summary = getattr(model_derived, '%s_Net_summary' % config.net_type.upper())
    model = derivedNetwork(config.net_config, config=config)
    model_summary = derivedNetwork_summary(config.net_config, config=config)

    model.eval()
    model_summary.cuda()
    if hasattr(model, 'net_config'):
        logging.info("Network Structure: \n" + '|\n'.join(map(str, model.net_config)))
    if args.meas_lat:
        latency_cpu = utils.latency_measure(model, (1, 256, 256), 1, 2000, mode='cpu')
        logging.info('latency_cpu (batch 1): %.2fms' % latency_cpu)
        latency_gpu = utils.latency_measure(model, (1, 256, 256), 32, 5000, mode='gpu')
        logging.info('latency_gpu (batch 32): %.2fms' % latency_gpu)
    params = utils.count_parameters_in_MB(model)
    logging.info("Params = %.2fMB" % params)
    # mult_adds = comp_multadds(model, input_size=config.data.input_size)
    # logging.info("Mult-Adds = %.2fMB" % mult_adds)

    model = nn.DataParallel(model)

    # whether to resume from a checkpoint
    if config.optim.if_resume:
        utils.load_model(model, config.optim.resume.load_path)
        start_epoch = config.optim.resume.load_epoch + 1
    else:
        start_epoch = 0

    model = model.cuda()

    # criterion = SegmentationLosses(name='cross_entropy_with_dice').cuda()
    criterion = EDiceLoss().cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        config.optim.init_lr,
        weight_decay=config.optim.weight_decay
    )
    scheduler = ExponentialLR(optimizer, gamma=0.996)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5) # MAX_STEP=int(1e10)

    # imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'),
    #                                     testFolder=os.path.join(args.data_path, 'val'),
    #                                     num_workers=config.data.num_workers,
    #                                     type_of_data_augmentation=config.data.type_of_data_aug,
    #                                     data_config=config.data)
    train_set = Silver07(args.data_path, mode='train')
    valid_set = Silver07(args.data_path, mode='val')
    train_queue = torch.utils.data.DataLoader(train_set, batch_size=config.data.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.data.num_workers, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(valid_set, batch_size=config.data.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.data.num_workers, pin_memory=True)

    # if config.optim.use_multi_stage:
    #     (train_queue, week_train_queue), valid_queue = imagenet.getSetTrainTestLoader(config.data.batch_size)
    # else:
    #     train_queue, valid_queue = imagenet.getTrainTestLoader(config.data.batch_size)

    torchsummary.summary(model_summary, input_size=(1, 256, 256))

    # scheduler = get_lr_scheduler(config, optimizer, train_queue.dataset.__len__())
    # scheduler.last_step = start_epoch * (train_queue.dataset.__len__() // config.data.batch_size + 1) - 1


    trainer = Trainer(model, train_queue, valid_queue, optimizer, criterion, scheduler, config, args.report_freq)

    start = time.time()
    patience, max_patience = 0, 500
    dur_time = 0
    save_best = True
    best_mIoU, best_loss, best_pixAcc, best_Dice = 0, 1.0, 0, 0
    for epoch in range(start_epoch, config.train_params.epochs):

        train_data = train_queue

        train_loss_avg, pixAcc_train, mIoU_train, Dice_train, visual_predicts_train, visual_target_train = trainer.train(epoch)
        logging.info('train loss: %e | epoch[%d]/[%d]', train_loss_avg, epoch,
                     config.train_params.epochs)
        # tbar.set_description('Train loss: %.3f' % (self.train_loss_meter.avg))
        lattest_lr = scheduler.get_lr()
        logging.info('pixAcc: %.3f; mIoU: %.5f; ; Dice: %.5f' % (pixAcc_train, mIoU_train, Dice_train))
        print('latest lr: ', lattest_lr)
        writer.add_scalar('Train/Loss', train_loss_avg, epoch)
        writer.add_scalar('Train/Dice', Dice_train, epoch)
        writer.add_image('predict_image/train', visual_predicts_train, epoch)
        visual_target_train = torch.squeeze(visual_target_train, dim=1)
        writer.add_image('target_image/train', visual_target_train, epoch)


        model_cur, val_loss_avg, pixAcc_val, mIoU_val, Dice_val, visual_predicts_valid, visual_target_valid = trainer.infer(epoch)
        logging.info(
            'Val loss: %.6f; pixAcc: %.3f; mIoU: %.5f; Dice: %.5f' % (
                val_loss_avg, pixAcc_val, mIoU_val, Dice_val))
        writer.add_scalar('Val/pixAcc', pixAcc_val, epoch)
        writer.add_scalar('Val/mIoU', mIoU_val, epoch)
        writer.add_scalar('Val/loss', val_loss_avg, epoch)
        writer.add_scalar('Val/Dice', Dice_val, epoch)
        writer.add_image('predict_image/valid', visual_predicts_valid, epoch)
        visual_target_valid = torch.squeeze(visual_target_valid, dim=1)
        writer.add_image('target_image/valid', visual_target_valid, epoch)

        if best_loss > val_loss_avg or best_mIoU < mIoU_val:
            patience = 0
        else:
            patience += 1
        best_pixAcc = pixAcc_val if best_pixAcc < pixAcc_val else best_pixAcc
        best_loss = val_loss_avg if best_loss > val_loss_avg else best_loss
        if best_Dice < Dice_val:
            best_Dice = Dice_val
            best_mIoU = mIoU_val if best_mIoU < mIoU_val else best_mIoU
            save_best = True

        logging.info('current best loss {}, pixAcc {}, mIoU {}, Dice {}'.format(
            best_loss, best_pixAcc, best_mIoU, best_Dice
        ))
        if save_best:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'dur_time': dur_time + time.time() - start,
                'model_state': model_cur.state_dict(),
                'best_pixAcc': best_pixAcc,
                'best_mIoU': best_mIoU,
                'best_Dice': best_Dice,
                'best_loss': best_loss,
            }, True, args.save)
            logging.info('save checkpoint (epoch %d) in %s  dur_time: %s'
                             , epoch, args.save, utils.calc_time(time.time() - config.train_params.epochs))
            save_best = False
        if patience == max_patience or epoch == config.train_params.epochs - 1:
            print('Early stopping')
            break
        else:
            logging.info('current patience :{}'.format(patience))
        logging.info('cost time: {}'.format(utils.calc_time(time.time() - start)))
    writer.export_scalars_to_json(args.save + "/all_scalars.json")
    writer.close()


    # if hasattr(model.module, 'net_config'):
    #     logging.info("Network Structure: \n" + '|\n'.join(map(str, model.module.net_config)))
    # logging.info("Params = %.2fMB" % params)
    # logging.info("Mult-Adds = %.2fMB" % mult_adds)
