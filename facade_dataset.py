#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import transform


# def label2image(lbl):
#     i = lbl.shape[0]
#     j = lbl.shape[1]
#     rgblbl = np.zeros((i,j,3))
#     for index_i in range(i):
#         for index_j in range(j):
#             rgblbl[index_i, index_j, :] = colormap[lbl[index_i, index_j]]
#     return rgblbl


class DatasetBase(data.Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split=split

        self.class_names = np.array(['Outlier', 'Window', 'Wall', 'Balcony', 'Door',
                                'Roof', 'Sky', 'Shop', 'Chimney'])
        self.mean_bgr = np.array([133.375976, 134.527542, 138.631788])
        self.std_bgr = np.array([67.907031, 66.337443, 66.169096])

        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(root, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(root, split,'%s.jpg' % did)
                lbl_file = osp.join(root, split,'%s_mask.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file).convert('RGB')
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file).convert('RGB')
        lbl = self.image2label(lbl)
        # lbl = np.array(lbl, dtype=np.int32)
        # lbl[lbl == 255] = -1

        if self.split=='train':
            return self.transform_train(img, lbl)
        else:
            return self.transform_test(img, lbl)

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

    def image2label(self,im):
        # RGB color for each class
        colormap = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [128, 0, 255],
                    [255, 128, 0], [0, 0, 255], [128, 255, 255], [0, 255, 0], [128, 128, 128]]
        colormap2label = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
        for i, cm in enumerate(colormap):
            colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引

        data = np.array(im, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(colormap2label[idx], dtype='int64')  # 根据索引得到 label 矩阵

class CMP_Dataset(DatasetBase):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split

        self.class_names = np.array(['background', 'facade', 'window', 'door', 'cornice',
                                'sill', 'balcony', 'blind', 'deco', 'molding', 'pillar', 'shop'])
        self.mean_bgr = np.array([105.479640, 115.025420, 121.890368])
        self.std_bgr = np.array([59.603436, 59.9710923, 61.937353])
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(root, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(root, split,'%s.jpg' % did)
                lbl_file = osp.join(root, split,'%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file).convert('RGB')
        # load label
        lbl_file = data_file['lbl']
        lbl = np.array(PIL.Image.open(lbl_file))-1
        if self.split=='train':
            return self.transform_train(img, lbl)
        else:
            return self.transform_test(img, lbl)

# class graz50_Dataset(DatasetBase):





# class etrims_Dataset(DatasetBase):







# class varcity3d_Dataset(DatasetBase):







class ECP_Dataset(DatasetBase):
    def __init__(self, root, split='train'):
        super(ECP_Dataset, self).__init__(root, split=split)



# class ParisArtDeco_Dataset(DatasetBase):





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
