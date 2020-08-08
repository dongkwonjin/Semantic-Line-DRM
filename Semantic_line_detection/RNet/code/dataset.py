import os
import torch
import random
import pickle

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset

from libs.utils import *

class Train_Dataset_RNet(Dataset):

    def __init__(self, cfg):
        # setting
        self.cfg = cfg

        # load datalist
        self.datalist = load_pickle(cfg.dataset_dir + 'data/train')

        # image transform
        self.transform = transforms.Compose([transforms.Resize((self.cfg.height, self.cfg.width), interpolation=2), transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, flip, idx):

        img_path = self.cfg.img_dir + self.datalist['img_path'][idx]

        img = Image.open(img_path)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img.size

        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]


    def get_negative_line(self, flip, idx):

        fname = self.cfg.pickle_dir + 'RNet/pickle/neg/' + \
                    self.datalist['img_path'][idx][:-4]
        data = load_pickle(fname)

        if flip == 1:
            data['outlier_line'][0][:, 0] = (self.width - 1) - data['outlier_line'][0][:, 0]
            data['outlier_line'][0][:, 2] = (self.width - 1) - data['outlier_line'][0][:, 2]
            for i in range(len(data['inlier_line'])):
                data['inlier_line'][i][:, 0] = (self.width - 1) - data['inlier_line'][i][:, 0]
                data['inlier_line'][i][:, 2] = (self.width - 1) - data['inlier_line'][i][:, 2]

        return data


    def get_positive_line(self, flip, idx):

        fname = self.cfg.pickle_dir + 'RNet/pickle/pos/' + \
                    self.datalist['img_path'][idx][:-4]
        data = load_pickle(fname)

        if flip == 1:
            for i in range(len(data['pos_line'])):
                data['pos_line'][i][:, 0] = (self.width - 1) - data['pos_line'][i][:, 0]
                data['pos_line'][i][:, 2] = (self.width - 1) - data['pos_line'][i][:, 2]
                data['offset'][i][:, 2] = -1 * data['offset'][i][:, 2]
                data['offset'][i][:, 2] = -1 * data['offset'][i][:, 2]

        return data


    def __getitem__(self, idx):

        # flip
        flip = random.randint(0, 1)

        # get pre-processed images
        img, img_size, img_name = self.get_image(flip, idx)
        img = self.transform(img)

        # load candidate label
        neg_label = self.get_negative_line(flip, idx)
        pos_label = self.get_positive_line(flip, idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'flip': flip,
                'pos_label': pos_label,
                'neg_label': neg_label}

    def __len__(self):
        return len(self.datalist['img_path'])


class SEL_Test_Dataset(Dataset):

    def __init__(self, cfg):

        self.cfg = cfg

        self.datalist = load_pickle(cfg.dataset_dir + 'data/test')
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.img_dir, self.datalist['img_path'][idx])
        img = Image.open(img_name)

        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_gtlines(self, idx):

        mul_gt = self.datalist['multiple'][idx]
        pri_gt_idx = self.datalist['primary'][idx]
        pri_gt = mul_gt[pri_gt_idx == 1]

        return pri_gt, mul_gt


    def __getitem__(self, idx):

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        img = self.transform(img)

        pri_gt, mul_gt = self.get_gtlines(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'pri_gt': pri_gt,
                'mul_gt': mul_gt}

    def __len__(self):
        return len(self.datalist['img_path'])

class SEL_Hard_Test_Dataset(Dataset):

    def __init__(self, cfg):
        # setting
        self.cfg = cfg
        self.scale = np.float32([cfg.width, cfg.height, cfg.width, cfg.height])

        # load datalist
        self.datalist = load_pickle(cfg.dataset_dir + 'data/SEL_Hard')

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.img_dir, self.datalist['img_path'][idx])

        img = Image.open(img_name).convert('RGB')
        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_gt_endpoints(self, idx):

        pri_gt = self.datalist['primary'][idx]
        mul_gt = self.datalist['multiple'][idx]

        return pri_gt, mul_gt

    def __getitem__(self, idx):

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        img = self.transform(img)

        pri_gt, mul_gt = self.get_gt_endpoints(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'pri_gt': pri_gt,
                'mul_gt': mul_gt}

    def __len__(self):
        return len(self.datalist['img_path'])