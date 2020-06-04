#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import transform

# RGB color for each class
colormap = [[0, 0, 0], [128, 128, 128], [0, 255, 0], [128, 255, 255], [0, 0, 255],
            [255, 128, 0], [128, 0, 255], [255, 255, 0], [255, 0, 0]]

colormap2label = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i, cm in enumerate(colormap):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引

def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(colormap2label[idx], dtype='int64')  # 根据索引得到 label 矩阵

class ECPDataset(data.Dataset):

    class_names = np.array(['Outlier', 'Chimney', 'Shop', 'Sky', 'Roof',
                            'Door', 'Balcony', 'Wall', 'Window'])

    mean_bgr = np.array([132.771101, 133.996078, 138.191973])
    std_bgr = np.array([67.851507, 66.293189, 66.143625])
    norm_mean_bgr = np.array([0.520671, 0.525475, 0.541929])
    norm_std_bgr = np.array([0.266084, 0.259973, 0.259387])

    def __init__(self, root, train=False):
        super(ECPDataset, self).__init__()
        self.root = root
        self.train = train

        self.img_ids = [i_id.strip() for i_id in open(osp.join(self.root, "images_id.txt"))]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s.jpg" % name)
            label_file = osp.join(self.root, "labels/%s_mask.png" % name)
            self.files.append({
                'img':img_file,
                'lbl':label_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file).convert('RGB')
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file).convert('RGB')
        lbl = image2label(lbl)
        # lbl = np.array(lbl, dtype=np.int32)
        # lbl[lbl == 255] = -1

        if self.train:
            return self.transform_train(img, lbl)
        else:
            return self.transform_test(img, lbl)
        # return self.transform(img, lbl)

    def transform_train(self, img, lbl):
        lbl = PIL.Image.fromarray(lbl.astype('uint8'))
        img, lbl = transform.RandomHorizontalFlip()(img,lbl)
        img, lbl = transform.RandomCrop(100)(img, lbl)

        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def transform_test(self, img, lbl):
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class Subset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.class_names=dataset.class_names
        self.untransform = dataset.untransform

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
