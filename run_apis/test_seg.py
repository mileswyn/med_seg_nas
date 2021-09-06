import os
import sys
import yaml
import time
import shutil
import argparse
import pprint
from tqdm import tqdm
import random
import torch
from PIL import Image
import torchvision
import numpy as np
import torchsummary
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from tools.metric_seg import *
import cv2
import logging
import tools.utils as utils

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import SimpleITK as sitk
from configs.promise12_train_cfg import cfg as config

sys.path.append('..')
from tensorboardX import SummaryWriter
from tools.loss_seg import SegmentationLosses
from models import model_derived_segment as model_derived
from dataset.Promise12_dataset import Promise12
from tools.augmentations import input_images

def test():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    parser = argparse.ArgumentParser("Test_Params")
    parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
    parser.add_argument('--data_path', type=str, default='/hdd1/wyn/PROMISE2012/', help='location of the data corpus')
    parser.add_argument('--load_path', type=str, default='./model_path', help='model loading path')
    parser.add_argument('--save', type=str, default='/hdd1/wyn/DenseNAS/log', help='experiment name')
    # parser.add_argument('--save_pt', type=str, default=r'E:\CodeRepo\DenseNAS\pt', help='experiment name')
    parser.add_argument('--tb_path', type=str, default='', help='tensorboard output path')
    parser.add_argument('--job_name', type=str, default='Test_res_promise12', help='job_name')
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