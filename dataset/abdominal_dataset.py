import os
import sys
import cv2
import torch
import random
import subprocess
import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
from skimage.exposure import equalize_adapthist
# from .base import BaseDataset
sys.path.append('/hdd1/wyn/DenseNAS')
sys.path.append('/hdd1/wyn/DenseNAS/tools')
from tools.utils import create_exp_dir
from tools.augmentations import smooth_images
from tools.augmentations import *
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as tf
import pickle
from collections import OrderedDict
from sklearn.model_selection import KFold
from batchgenerators.dataloading import SlimDataLoaderBase

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

class Abdominal():
    def __init__(self, fold=3, batch_size=2):
        super(Abdominal, self).__init__()
        self.folder_with_preprocessed_data = '/hdd1/wyn/nnUNetFrame/nnUNet_preprocessed/Task102_TCIAorgan/nnUNetData_plans_v2.1_stage0'
        self.dataset = None
        self.batch_size = batch_size
        self.fold = fold
        self.patch_size = [205, 205, 205]
        self.final_patch_size = [128,128,128]
        
    # def load_dataset(self):
    #     self.dataset = load_dataset(self.folder_with_preprocessed_data)

    # def do_split(self):
    #     """
    #     This is a suggestion for if your dataset is a dictionary (my personal standard)
    #     :return:
    #     """
    #     splits_file = join(self.dataset_directory, "splits_final.pkl")
    #     if not isfile(splits_file):
    #         print("Creating new split...")
    #         splits = []
    #         all_keys_sorted = np.sort(list(self.dataset.keys()))
    #         kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    #         for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
    #             train_keys = np.array(all_keys_sorted)[train_idx]
    #             test_keys = np.array(all_keys_sorted)[test_idx]
    #             splits.append(OrderedDict())
    #             splits[-1]['train'] = train_keys
    #             splits[-1]['val'] = test_keys

    #     splits = load_pickle(splits_file)
    #     tr_keys = splits[self.fold]['train']
    #     val_keys = splits[self.fold]['val']

    #     tr_keys.sort()
    #     val_keys.sort()

    #     self.dataset_tr = OrderedDict()
    #     for i in tr_keys:
    #         self.dataset_tr[i] = self.dataset[i]

    #     self.dataset_val = OrderedDict()
    #     for i in val_keys:
    #         self.dataset_val[i] = self.dataset[i]
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_tr = dataloader3D_abdominal(self.dataset_tr, self.batch_size, self.patch_size, self.final_patch_size)
        dl_val = dataloader3D_abdominal(self.dataset_val, self.batch_size, self.patch_size, self.final_patch_size)
        return dl_tr, dl_val

class dataloader3D_abdominal(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size, final_patch_size):
        super(dataloader3D_abdominal, self).__init__(data, batch_size, None)
        self.dataset_directory = '/hdd1/wyn/nnUNetFrame/nnUNet_preprocessed/Task102_TCIAorgan'
        self.dataset = None
        self.data = data
        self.batchsize = batch_size
        self.patch_size = patch_size
        self.final_patch_size = final_patch_size
        self.oversample_foreground_percent = 0.33
        self.data_shape,self.seg_shape = (self.batchsize, 1, *self.patch_size), (self.batchsize, 1, *self.patch_size)
        self.need_to_pad = [77,77,77]
        self.list_of_keys = list(self._data.keys())
    
    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batchsize *  (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batchsize, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        for j, i in enumerate(selected_keys):

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", 'r')
            for d in range(3):
                if self.need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    self.need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]
            shape = case_all_data.shape[1:]
            lb_x = - self.need_to_pad[0] // 2
            ub_x = shape[0] + self.need_to_pad[0] // 2 + self.need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - self.need_to_pad[1] // 2
            ub_y = shape[1] + self.need_to_pad[1] // 2 + self.need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - self.need_to_pad[2] // 2
            ub_z = shape[2] + self.need_to_pad[2] // 2 + self.need_to_pad[2] % 2 - self.patch_size[2]
            
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                
                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)
                    # print('selected_class', selected_class)
                    voxels_of_that_class = properties['class_locations'][selected_class]
                
                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub,
                            valid_bbox_z_lb:valid_bbox_z_ub])
            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                            (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                        'constant')

            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                            'constant', **{'constant_values': -1})
            
        return  {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}
    
    

def read_npy_npz_pkl(path, num):
    npy_read = np.load(os.path.join(path, 'pancreas_ct'+number+'.npy'))
    npz_read = np.load(os.path.join(path, 'pancreas_ct' + number + '.npz'))
    with open(os.path.join(path, 'pancreas_ct' + number + '.pkl'), 'rb') as f:
        pkl_read = pickle.load(f)
    return None


if __name__ == '__main__':
    npz_path = '/hdd1/wyn/nnUNetFrame/nnUNet_preprocessed/Task102_TCIAorgan/nnUNetData_plans_v2.1_stage0/'
    number = '01'
    read_npy_npz_pkl(npz_path, number)
    # a = Abdominal()