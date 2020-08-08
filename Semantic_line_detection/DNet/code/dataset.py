import os
import torch
import random
import pickle

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset

from libs.utils import *

class Train_Dataset_DNet(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(cfg.pickle_dir + 'detector_train_candidates')
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.pos_num = int(cfg.batch_size['train_line'] * cfg.ratio_pos)
        self.neg_num = int(cfg.batch_size['train_line'] * (1 - cfg.ratio_pos))

        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, flip, idx):
        img_name = os.path.join(self.cfg.img_dir, self.datalist[idx]['img_path'])
        img = Image.open(img_name)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist[idx]['img_path']

    def get_data(self, flip, idx):

        data = self.datalist[idx]

        pos_data = np.concatenate((data['pos_line']['endpts'],
                                   data['pos_line']['cls'],
                                   data['pos_line']['offset']), axis=1)
        neg_data = np.concatenate((data['neg_line']['endpts'],
                                   data['neg_line']['cls'],
                                   data['neg_line']['offset']), axis=1)

        # random sampling
        pos_idx = torch.randperm(pos_data.shape[0])[:self.pos_num]
        neg_idx = torch.randperm(neg_data.shape[0])[:self.neg_num]

        pos_data = pos_data[pos_idx]
        neg_data = neg_data[neg_idx]


        if flip == 1:
            pos_data[:, 0] = self.width - 1 - pos_data[:, 0]
            pos_data[:, 2] = self.width - 1 - pos_data[:, 2]
            pos_data[:, 6] = -1 * pos_data[:, 6]
            pos_data[:, 8] = -1 * pos_data[:, 8]

            neg_data[:, 0] = self.width - 1 - neg_data[:, 0]
            neg_data[:, 2] = self.width - 1 - neg_data[:, 2]
            neg_data[:, 6] = -1 * neg_data[:, 6]
            neg_data[:, 8] = -1 * neg_data[:, 8]

        return pos_data, neg_data

    def normalize_point_coord(self, data):
        data[:, 6] = data[:, 6] / self.cfg.width
        data[:, 7] = data[:, 7] / self.cfg.height
        data[:, 8] = data[:, 8] / self.cfg.width
        data[:, 9] = data[:, 9] / self.cfg.height
        return data

    def __getitem__(self, idx):
        flip = random.randint(0, 1)
        # get pre-processed images
        img, img_size, img_name = self.get_image(flip, idx)
        img = self.transform(img)

        # get candidate lines
        pos_data, neg_data = self.get_data(flip, idx)
        pos_data = self.normalize_point_coord(pos_data)

        train_data = np.concatenate((pos_data, neg_data), axis=0)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'flip': flip,
                'train_data': train_data}

    def __len__(self):
        return len(self.datalist)



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
        img_name = os.path.join(self.cfg.img_dir,
                                self.datalist['img_path'][idx])
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
