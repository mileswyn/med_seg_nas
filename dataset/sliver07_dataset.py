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
from tools.utils import create_exp_dir
from tools.augmentations import smooth_images
from tools.augmentations import *
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as tf
import pickle

def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def silver07_data2array(base_path, foldchose, store_path,img_rows,img_cols):
    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows / 8), int(img_cols / 8)))
    fileList = os.listdir(os.path.join(base_path, 'training-scans'))
    fileList = sorted((x for x in fileList if '.mhd' in x))
    val_list = foldchose
    train_list = list(set(range(1,21)) - set(val_list))
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []
        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(3) in file]
        for filename in filtered:
            itkimage = sitk.ReadImage(os.path.join(base_path, 'training-scans', filename))
            imgs = sitk.GetArrayFromImage(itkimage)
            if 'seg' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append(imgs)
                print('{} segmentation done'.format(filename))
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs)
                print('{} image done'.format(filename))
        # images: slices x w x h ==> total number x w x h
        images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
        masks = masks.astype(np.uint8)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)
        images = images.astype(np.float32)

        if count == 0:  # no normalize
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu) / sigma
            np.save(os.path.join(store_path, 'image_train.npy'), images)
            np.save(os.path.join(store_path, 'label_train.npy'), masks)
        elif count == 1:
            images = (images - mu) / sigma
            np.save(os.path.join(store_path, 'image_val.npy'), images)
            np.save(os.path.join(store_path,'label_val.npy'), masks)
        count += 1
    fileList = os.listdir(os.path.join(base_path, 'test-scans'))
    fileList = sorted([x for x in fileList if '.mhd' in x])
    n_imgs = []
    images = []
    for filename in fileList:
        itkimage = sitk.ReadImage(os.path.join(base_path, 'test-scans', filename))
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        images.append(imgs)
        n_imgs.append(len(imgs))

    images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols)
    images = smooth_images(images)
    images = images.astype(np.float32)
    images = (images - mu) / sigma

    np.save(os.path.join(store_path,'image_test.npy'), images)
    np.save(os.path.join(store_path, 'test_label_imgs.npy'), np.array(n_imgs)) # no label, label means the length of test images
    print('save file in {}'.format(store_path))


def data_to_array(base_path, store_path, img_rows, img_cols):

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows/8),int(img_cols/8)))

    fileList = os.listdir(os.path.join(base_path, 'TrainingData'))

    fileList = sorted((x for x in fileList if '.mhd' in x))

    val_list = [5, 15, 25, 35, 45]
    train_list = list(set(range(50)) - set(val_list) )
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []

        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file ]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, 'TrainingData', filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append( imgs )
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs)

        # images: slices x w x h ==> total number x w x h
        images = np.concatenate(images , axis=0 ).reshape(-1, img_rows, img_cols)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
        masks = masks.astype(np.uint8)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)
        images = images.astype(np.float32)

        if count==0: # no normalize
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu)/sigma

            #np.save(os.path.join(store_path, 'X_train.npy'), images)
            #np.save(os.path.join(store_path,'y_train.npy'), masks)
        elif count==1:
            images = (images - mu)/sigma
            #np.save(os.path.join(store_path, 'X_val.npy'), images)
            #np.save(os.path.join(store_path,'y_val.npy'), masks)
        count+=1

    fileList =  os.listdir(os.path.join(base_path, 'TestData'))
    fileList = sorted([x for x in fileList if '.mhd' in x])
    n_imgs=[]
    images=[]
    for filename in fileList:
        itkimage = sitk.ReadImage(os.path.join(base_path, 'TestData', filename))
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        images.append(imgs)
        n_imgs.append(len(imgs))

    images = np.concatenate(images , axis=0).reshape(-1, img_rows, img_cols)
    images = smooth_images(images)
    images = images.astype(np.float32)
    images = (images - mu)/sigma

    #np.save(os.path.join(store_path,'X_test.npy'), images)
    #np.save(os.path.join(store_path, 'test_n_imgs.npy'), np.array(n_imgs))
    print('save file in {}'.format(store_path))

def only_train_data_to_array(base_path, store_path, img_rows, img_cols):

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows/8),int(img_cols/8)))

    fileList =  os.listdir(os.path.join(base_path, 'TrainingData'))

    fileList = sorted((x for x in fileList if '.mhd' in x))

    train_list = list(set(range(50)))

    images = []
    masks = []

    filtered = [file for file in fileList for ff in train_list if str(ff).zfill(2) in file]

    for filename in filtered:

        itkimage = sitk.ReadImage(os.path.join(base_path, 'TrainingData', filename))
        imgs = sitk.GetArrayFromImage(itkimage)

        if 'segm' in filename.lower():
            imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
            masks.append( imgs )
        else:
            imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
            images.append(imgs)

    # images: slices x w x h ==> total number x w x h
    images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols)
    masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
    masks = masks.astype(np.uint8)

    # Smooth images using CurvatureFlow
    images = smooth_images(images)
    images = images.astype(np.float32)

    mu = np.mean(images)
    sigma = np.std(images)
    images = (images - mu)/sigma
    np.save(os.path.join(store_path, 'X_train.npy'), images)
    np.save(os.path.join(store_path,'y_train.npy'), masks)


    fileList =  os.listdir(os.path.join(base_path, 'TestData'))
    fileList = sorted([x for x in fileList if '.mhd' in x])
    n_imgs=[]
    images=[]
    for filename in fileList:
        itkimage = sitk.ReadImage(os.path.join(base_path, 'TestData', filename))
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        images.append(imgs)
        n_imgs.append(len(imgs))

    images = np.concatenate(images , axis=0).reshape(-1, img_rows, img_cols)
    images = smooth_images(images)
    images = images.astype(np.float32)
    images = (images - mu)/sigma

    np.save(os.path.join(store_path,'X_test.npy'), images)
    np.save(os.path.join(store_path, 'test_n_imgs.npy'), np.array(n_imgs))
    print('save file in {}'.format(store_path))

def load_train_data(store_path):

    X_train = np.load(os.path.join(store_path, 'image_train.npy'))
    y_train = np.load(os.path.join(store_path, 'label_train.npy'))

    return X_train, y_train

def load_val_data(store_path):

    X_val = np.load(os.path.join(store_path, 'image_val.npy'))
    y_val = np.load(os.path.join(store_path, 'label_val.npy'))
    return X_val, y_val

def load_test_data(store_path):
    X_test = np.load(os.path.join(store_path, 'image_test.npy'))
    x_slice_array = np.load(os.path.join(store_path, 'test_label_imgs.npy'))
    return X_test, x_slice_array

def get_test_list(base_path):
    fileList = os.listdir(os.path.join(base_path, 'test-scans'))
    fileList = sorted([os.path.join(base_path, 'test-scans',x) for x in fileList if '.mhd' in x])
    return fileList

# ce+dice:0.9098(dice)/0.8346(miou)

class Silver07(data.Dataset):
    # IN_CHANNELS = 1
    # BASE_DIR = 'PROMISE2012'
    # TRAIN_IMAGE_DIR = 'TrainingData'
    # VAL_IMAGE_DIR = 'TestData'
    # NUM_CLASS = 1
    # CROP_SIZE = 256
    # CLASS_WEIGHTS = None

    def __init__(self, root, mode):
        super(Silver07, self).__init__()
        self.mode = mode
        #self.joint_transform = joint_transform
        # root = root + '/' + self.BASE_DIR
        self.joint_transform_train = Compose([
            RandomHorizontallyFlip(),
            RandomElasticTransform(alpha=1.5, sigma=0.07, img_type='F'),
        ])
        # self.joint_transform_valid = Compose([
        #     CenterCrop(size=192),
        # ])
        # RandomTranslate(offset=(0.2, 0.1)),
        # RandomVerticallyFlip(),
        self.RET = RandomElasticTransform(alpha=1.5, sigma=0.07, img_type='F')
        self.transform_image = torchvision.transforms.Compose([
            #torchvision.transforms.RandomVerticalFlip(),
            #torchvision.transforms.RandomHorizontalFlip(),
            RandomElasticTransform_image(alpha = 1.5, sigma = 0.07, img_type='F'),
            ])
        self.transform_mask = torchvision.transforms.Compose([
            #torchvision.transforms.RandomVerticalFlip(),
            #torchvision.transforms.RandomHorizontalFlip(),
            RandomElasticTransform_mask(alpha = 1.5, sigma = 0.07, img_type='F'),
            ])

        self.img_normalize = None
        # fold0 = ['Case08', 'Case12', 'Case21', 'Case22', 'Case30', 'Case32', 'Case33', 'Case38', 'Case42', 'Case45']
        # fold1 = ['Case00', 'Case01', 'Case07', 'Case13', 'Case15', 'Case23', 'Case24', 'Case34', 'Case40', 'Case46']
        # fold2 = ['Case02', 'Case05', 'Case09', 'Case17', 'Case18', 'Case25', 'Case28', 'Case35', 'Case39', 'Case44']
        # fold3 = ['Case03', 'Case06', 'Case10', 'Case14', 'Case19', 'Case26', 'Case29', 'Case36', 'Case41', 'Case47']
        # fold4 = ['Case04', 'Case11', 'Case16', 'Case20', 'Case27', 'Case31', 'Case37', 'Case43', 'Case48', 'Case49']
        fold0 = [6,7,12,19]
        fold1 = [1,5,11,17]
        fold2 = [2,9,13,15]
        fold3 = [3,8,14,18]
        fold4 = [4,10,16,20]
        # SECOND
        # store data in the npy file
        data_path = os.path.join(root, 'Silver07_npy_fold0')

        if not os.path.exists(data_path):
            create_exp_dir(data_path)
            silver07_data2array(root, fold0, data_path, 256, 256)
        else:
            print('read the data from: {}'.format(data_path))

        self.test_file_list = get_test_list(root)
        self.blank_layer = []
        # read the data from npy
        if mode == 'train':
            self.X_train, self.y_train = load_train_data(data_path)
            # self.size = self.X_train.shape[0]
            # for i in range(self.X_train.shape[0]):
            #     if np.sum(self.y_train[i,:,:]) == 0:
            #         self.blank_layer.append(i)
            # self.blank_layer.reverse()
            # with open(r'E:\CodeRepo\myUnet\Silver07_blank.pkl', 'wb') as f:
            #     pickle.dump(self.blank_layer, f)



            with open('/hdd1/wyn/DenseNAS/run_apis/Silver07_blank.pkl', 'rb') as f:
                a = pickle.load(f)
            self.X_train_no_blank = self.X_train
            self.y_train_no_blank = self.y_train
            for j in a:
                if random.random() < 0.8:
                    self.X_train_no_blank = np.delete(self.X_train_no_blank, j, 0)
                    self.y_train_no_blank = np.delete(self.y_train_no_blank, j, 0)
            print('blank layer % its size', len(a))
            # self.size = self.X_train.shape[0]
            img_aug = Image.fromarray(self.X_train_no_blank[0], mode='F')
            target_aug = Image.fromarray(self.y_train_no_blank[0], mode='L')
            img2, target2 = self.joint_transform_train(img_aug, target_aug)
            img2 = np.array(img2)
            target2 = np.array(target2)
            img_aug_arr = np.expand_dims(img2, 0)
            target_aug_arr = np.expand_dims(target2, 0)
            for im in range(1, self.X_train_no_blank.shape[0]):
                img_aug = Image.fromarray(self.X_train_no_blank[im], mode='F')
                target_aug = Image.fromarray(self.y_train_no_blank[im], mode='L')
                img2, target2 = self.joint_transform_train(img_aug, target_aug)
                img2 = np.array(img2)
                img2 = np.expand_dims(img2, 0)
                target2 = np.array(target2)
                target2 = np.expand_dims(target2, 0)
                img_aug_arr = np.concatenate((img_aug_arr, img2), axis=0)
                target_aug_arr = np.concatenate((target_aug_arr, target2), axis=0)
            # self.X_train = np.concatenate((self.X_train, img_aug_arr), axis=0)
            # self.y_train = np.concatenate((self.y_train, target_aug_arr), axis=0)
            self.X_train = img_aug_arr
            self.y_train = target_aug_arr
            # self.y_train = np.array(self.y_train == 1).astypy('int64')
            self.size = self.X_train.shape[0]
            print('train set size is: ',self.size)
        elif mode == 'val':
            self.X_val, self.y_val = load_val_data(data_path)
            self.size = self.X_val.shape[0]
        elif mode == 'test':
            self.X_test, self.x_slice_array = load_test_data(data_path)
            self.size = self.X_test.shape[0]

    def __getitem__(self, index):
        # 1. the image already crop
        if self.mode == "train":
            img, target = self.X_train[index], self.y_train[index]
        elif self.mode == 'val':
            img, target = self.X_val[index], self.y_val[index]
            #img, target = self.X_val[index], self.validfilelist
        elif self.mode == 'test': # the test target indicate the number of slice for each case
            img, target = self.X_test[index], self.test_file_list
        img = Image.fromarray(img, mode='F')
        #target = np.array(target, dtype=np.float32)

        if self.mode == 'train':
            target = Image.fromarray(target, mode='L')
            # 2. do joint transform
            if self.joint_transform_train is not None:
                # img, target = self.joint_transform(img, target)
                img = np.array(img)
                target = np.array(target)
                # 3. to tensor
                img = torch.from_numpy(img)
                target = torch.from_numpy(target)
        elif self.mode == 'val':
            # target = Image.fromarray(target, mode='L')
            target = Image.fromarray(target, mode='L')
            # img,target = self.joint_transform_valid(img, target)
            img = np.array(img)
            target = np.array(target)
            img = torch.from_numpy(img)
            target = torch.from_numpy(target)
        else:
            # 3. img to tensor
            # img = tf.to_tensor(img)
            img = torch.from_numpy(img)


        img = img.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        # 4. normalize for img
        if self.img_normalize != None:
            img = self.img_normalize(img)

        return img, target

    def __len__(self):
        return self.size

import torch.utils.data as data

# if __name__ == '__main__':
#     root = r'E:\anyDataset\PROMISE2012'
#     data_path = r'E:\anyDataset\PROMISE2012\npy_image'
#     CROP_SIZE = 256
#     data_to_array(root, data_path, CROP_SIZE, CROP_SIZE)
